import logging
import pathlib
import re
import string
from operator import itemgetter

import html2text
import mdformat
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

import connectors.config as cfg

from .agent import db

log = logging.getLogger("tangerine.db.vector")


TXT_SEPARATORS = [
    "\n\n## ",
    "\n\n### ",
    "\n\n#### ",
    "\n\n##### ",
    "\n\n###### ",
    "\n\n",
    "\n",
    " ",
    "",
]


class VectorStoreInterface:
    def __init__(self):
        self.store = None
        self.vector_chunk_size = 2000
        self.vector_chunk_overlap = 0

        self.embeddings = OpenAIEmbeddings(
            model=cfg.EMBED_MODEL_NAME,
            openai_api_base=cfg.EMBED_BASE_URL,
            openai_api_key=cfg.EMBED_API_KEY,
            check_embedding_ctx_length=False,
        )

    def init_vector_store(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection=cfg.DB_URI,
                embeddings=self.embeddings,
            )
        except Exception:
            log.exception("error initializing vector store")

    def combine_small_chunks(self, chunks):
        """
        Combine small chunks into the next chunk

        Sometimes we see the text splitter create a chunk containing only a single line (like
        a header), we will store these small chunks on the next chunk to avoid storing a
        document with small context
        """
        for idx, chunk in enumerate(chunks):
            if len(chunk) < 200:
                # this chunk is less than 200 chars, move it to the next chunk
                try:
                    chunks[idx + 1] = f"{chunk}\n\n{chunks[idx + 1]}"
                except IndexError:
                    # we've reached the end and there is no 'next chunk', just give up
                    break
                # make note of its index and pop it later...
                chunks[idx] = "<<removed>>"

        chunks = list(filter(lambda val: val != "<<removed>>", chunks))

        return chunks

    def split_to_docs(self, text, metadata):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            separators=TXT_SEPARATORS,
        )

        chunks = text_splitter.split_text(text)
        chunks = self.combine_small_chunks(chunks)

        # find title if possible and add to metadata
        first_line_of_first_chunk = chunks[0].splitlines()[0]
        if first_line_of_first_chunk.startswith("# "):
            # we found the title header, add it to metadata
            metadata["title"] = first_line_of_first_chunk.strip("# ")

        documents = []
        for chunk in chunks:
            if cfg.EMBED_DOCUMENT_PREFIX:
                chunk = f"{cfg.EMBED_DOCUMENT_PREFIX}: {chunk}"
            documents.append(Document(page_content=chunk, metadata=metadata))

        return documents

    def remove_large_code_blocks(self, text):
        lines = []
        code_lines = []
        in_code_block = False
        for line in text.split("\n"):
            if line.strip() == "```" and not in_code_block:
                in_code_block = True
                code_lines = []
                code_lines.append(line)
            elif line.strip() == "```" and in_code_block:
                code_lines.append(line)
                in_code_block = False
                if len(code_lines) > 9:
                    code_lines = ["```", "<large code block, visit documentation to view>", "```"]
                lines.extend(code_lines)
            elif in_code_block:
                code_lines.append(line)
            else:
                lines.append(line)

        return "\n".join(lines)

    def html_to_md(self, text):
        """
        Parse a .html page that has been composed with mkdocs

        It is assumed that the page contains 'md-content' which was compiled based on .md

        TODO: possibly handle html pages not built with mkdocs
        """
        md = ""
        soup = BeautifulSoup(text, "lxml")
        # extract 'md-content', this ignores nav/header/footer/etc.
        md_content = soup.find("div", class_="md-content")
        if md_content:
            # remove "Edit this page" button
            edit_button = md_content.find("a", title="Edit this page")
            if edit_button:
                edit_button.decompose()

            # remove line numbers from code blocks
            linenos_columns = md_content.find_all("td", class_="linenos")
            for linenos_column in linenos_columns:
                linenos_column.decompose()

            h = html2text.HTML2Text()
            h.ignore_images = True
            h.mark_code = True
            h.body_width = 0
            h.ignore_emphasis = True
            h.wrap_links = False
            h.ignore_tables = True

            html2text_output = h.handle(str(md_content))

            md_lines = []
            in_code_block = False

            for line in html2text_output.split("\n"):
                # remove non printable chars (like paragraph markers)
                line = "".join(filter(lambda char: char in string.printable, line))

                # replace html2text code block start/end with standard md
                if "[code]" in line:
                    in_code_block = True
                    line = line.replace("[code]", "```")
                elif "[/code]" in line:
                    in_code_block = False
                    line = line.replace("[/code]", "```")
                if in_code_block:
                    # also fix indent, html2text indents all code content by 4 spaces
                    if line.startswith("```"):
                        # start of code block and the txt may be on the same line...
                        #   eg: ```    <text>
                        line = f"```\n{line[8:]}"
                    else:
                        line = line[4:]
                md_lines.append(line)

            md = "\n".join(md_lines)

            # strip empty newlines before end of code blocks
            md = re.sub(r"\n\n+```", "\n```", md)
            # strip empty newlines after the start of a code block
            md = re.sub(r"```\n\n+", "```\n", md)
            # finally, use opinionated formatter
            md = mdformat.text(md)
        else:
            log.error("no 'md-content' div found")

        md = self.remove_large_code_blocks(md)

        return md

    def create_documents(self, text, agent_id, source, full_path):
        log.debug("processing %s", full_path)
        if full_path.lower().endswith(".html"):
            text = self.html_to_md(text)

        if not text:
            raise ValueError("no document text provided")

        metadata = {
            "agent_id": str(agent_id),
            "source": source,
            "full_path": full_path,
            "filename": pathlib.Path(full_path).name,
        }

        chunked_docs = self.split_to_docs(text, metadata)

        log.debug("document chunks: %s", chunked_docs)

        return chunked_docs

    def add_document(self, text, agent_id, source, full_path):
        try:
            documents = self.create_documents(text, agent_id, source, full_path)
            if documents:
                self.store.add_documents(documents)
                log.debug(
                    "added %d documents to agent %d from source %s",
                    len(documents),
                    agent_id,
                    source,
                )
        except Exception:
            log.exception("error adding documents")

    def search(self, query, agent_id):
        filter = {"agent_id": str(agent_id)}
        if cfg.EMBED_QUERY_PREFIX:
            query = f"{cfg.EMBED_QUERY_PREFIX}: {query}"

        # return 4 chunks using MMR
        results = self.store.max_marginal_relevance_search_with_score(
            query=query,
            filter=filter,
            lambda_mult=0.7,
            k=4,
        )

        # return 2 chunks using sentence similarity
        results.extend(self.store.similarity_search_with_score(query=query, filter=filter, k=2))

        # sort by score lowest to highest, lower is "less distance" which is better
        results = sorted(results, key=itemgetter(1))
        # drop the score
        results = [result[0] for result in results]

        # de-dupe, 'Document' is unhashable so check page content
        unique_results = []
        for new_result in results:
            present = False
            for existing_result in unique_results:
                if new_result.page_content == existing_result.page_content:
                    # this one is already in the list, don't add it
                    present = True
                    break
            if not present:
                unique_results.append(new_result)

        return unique_results

    def delete_documents_by_metadata(self, metadata: dict) -> dict:
        if not metadata:
            raise ValueError("empty metadata")

        filter_stmts = []

        for key, val in metadata.items():
            if not isinstance(val, str):
                raise ValueError("metadata values must be of type 'str'")
            # use parameterized query
            filter_stmt = f"cmetadata->>'{key}' = :{key}"
            filter_stmts.append(filter_stmt)

        filter_ = " AND ".join(filter_stmts)

        query = text(f"SELECT id, cmetadata FROM langchain_pg_embedding WHERE {filter_};")
        results = db.session.execute(query, metadata).all()

        matching_docs = []
        for result in results:
            # add document id into each result
            result[1]["id"] = result[0]
            matching_docs.append(result[1])

        log.debug(
            "found %d doc(s) from vector DB matching filter: %s", len(matching_docs), metadata
        )

        self.delete_documents([doc["id"] for doc in matching_docs])

        return matching_docs

    def delete_documents(self, ids):
        log.debug("deleting %d doc(s) from vector store", len(ids))
        self.store.delete(ids)


vector_db = VectorStoreInterface()
