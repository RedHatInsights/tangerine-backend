import logging
import pathlib
import re
import string

import html2text
import mdformat
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

import connectors.config as cfg

log = logging.getLogger("tangerine.db")

db = SQLAlchemy()


class Agents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    filenames = db.Column(db.ARRAY(db.String), default=[], nullable=True)

    def __repr__(self):
        return f"<Agents {self.id}>"


class VectorStoreInterface:
    def __init__(self):
        self.store = None
        self.vector_chunk_size = 1024
        self.vector_chunk_overlap = int(self.vector_chunk_size * 0.3)

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

    def split_docs_to_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

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
                line = "".join(filter(lambda x: x in string.printable, line))
                if line.strip() == "[code]":
                    in_code_block = True
                    # replace html2text code block start with standard md
                    line = "```"
                elif line.strip() == "[/code]":
                    in_code_block = False
                    # replace html2text code block end with standard md
                    line = "```"
                elif in_code_block:
                    # html2text indents all code blocks, un-indent first level
                    line = line[4:]
                md_lines.append(line)

            md = "\n".join(md_lines)

            # strip empty newlines before end of code blocks
            md = re.sub(r"\n\n```", "\n```", md)
            # finally, use opinionated formatter
            md = mdformat.text(md)
        else:
            log.error("no 'md-content' div found")

        return md

    def create_documents(self, text, agent_id, source, full_path):
        log.debug("processsing %s", full_path)
        if full_path.lower().endswith(".html"):
            text = self.html_to_md(text)

        if not text:
            raise ValueError("no document text provided")

        documents = [
            Document(
                page_content=text,
                metadata={
                    "agent_id": str(agent_id),
                    "source": source,
                    "full_path": full_path,
                    "filename": pathlib.Path(full_path).name,
                },
            )
        ]

        chunked_docs = self.split_docs_to_chunks(documents)
        return chunked_docs

    def add_document(self, text, agent_id, source, full_path):
        try:
            self.store.add_documents(self.create_documents(text, agent_id, source, full_path))
        except Exception:
            log.exception("error adding documents")

    def search(self, query, agent_id):
        results = self.store.max_marginal_relevance_search(
            query=query, filter={"agent_id": str(agent_id)}, k=6
        )

        return results

    def delete_documents(self, ids):
        self.store.delete(ids)


vector_interface = VectorStoreInterface()
