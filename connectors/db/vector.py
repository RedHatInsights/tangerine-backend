import json
import logging
from operator import itemgetter

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

import connectors.config as cfg

from .agent import db
from .file import File

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

    def split_to_document_chunks(self, text, metadata):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            separators=TXT_SEPARATORS,
        )

        # find title if possible and add to metadata
        for line in text.splitlines():
            if line.startswith("# "):
                # we found a title header, add it to metadata
                metadata["title"] = line.strip("# ")
                break

        chunks = text_splitter.split_text(text)
        chunks = self.combine_small_chunks(chunks)

        documents = []
        for chunk in chunks:
            if cfg.EMBED_DOCUMENT_PREFIX:
                chunk = f"{cfg.EMBED_DOCUMENT_PREFIX}: {chunk}"
            documents.append(Document(page_content=chunk, metadata=metadata))

        return documents

    def create_document_chunks(self, file: File, agent_id: int):
        log.debug("processing %s", file)

        text = file.extract_text()

        if not text:
            log.error("file %s: empty text", file)
            return []

        metadata = {
            "agent_id": str(agent_id),
        }
        metadata.update(file.metadata)

        chunks = self.split_to_document_chunks(text, metadata)

        log.debug("document chunks: %s", chunks)

        return chunks

    def add_file(self, file: File, agent_id: int):
        try:
            chunks = self.create_document_chunks(file, agent_id)
            if chunks:
                self.store.add_documents(chunks)
                log.debug(
                    "added %d document chunks to agent %s",
                    len(chunks),
                    agent_id,
                )
        except Exception:
            log.exception("error adding documents")

    def search(self, query, agent_id: int):
        filter = {"agent_id": str(agent_id), "active": "True"}
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

    def _build_metadata_filter(self, metadata):
        filter_stmts = []

        metadata_as_str = {key: str(val) for key, val in metadata.items()}

        for key, val in metadata_as_str.items():
            # use parameterized query
            filter_stmt = f"cmetadata->>'{key}' = :{key}"
            filter_stmts.append(filter_stmt)

        filter_ = " AND ".join(filter_stmts)

        return metadata_as_str, filter_

    def get_distinct_cmetadata(self, filter):
        if not filter:
            raise ValueError("empty metadata")

        metadata_as_str, filter_ = self._build_metadata_filter(filter)
        query = text(
            f"SELECT distinct on (cmetadata) cmetadata from langchain_pg_embedding where {filter_}"
        )
        results = db.session.execute(query, metadata_as_str).all()

        return [row.cmetadata for row in results]

    def get_ids_and_cmetadata(self, filter):
        if not filter:
            raise ValueError("empty metadata")

        metadata_as_str, filter_ = self._build_metadata_filter(filter)
        query = text(f"SELECT id, cmetadata FROM langchain_pg_embedding WHERE {filter_}")
        results = db.session.execute(query, metadata_as_str).all()

        return results

    def delete_document_chunks_by_id(self, ids):
        log.debug("deleting %d document chunks from vector store", len(ids))
        self.store.delete(ids)

    def delete_document_chunks(self, filter: dict) -> dict:
        results = self.get_ids_and_cmetadata(filter)

        matching_docs = []
        for result in results:
            # add document id into each result
            result.cmetadata["id"] = result.id
            matching_docs.append(result.cmetadata)

        log.debug("found %d doc(s) from vector DB matching filter: %s", len(matching_docs), filter)

        self.delete_document_chunks_by_id([doc["id"] for doc in matching_docs])

        return matching_docs

    def set_doc_states(self, active: bool, pending_removal: bool, filter: dict):
        metadata_as_str, filter_ = self._build_metadata_filter(filter)
        data = {"active": str(active), "pending_removal": str(pending_removal)}
        update = (
            "UPDATE langchain_pg_embedding "
            f"SET cmetadata = cmetadata || '{json.dumps(data)}' "
            f"WHERE {filter_}"
        )
        db.session.execute(text(update), metadata_as_str)
        db.session.commit()


vector_db = VectorStoreInterface()
