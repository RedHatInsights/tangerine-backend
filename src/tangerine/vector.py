import importlib
import itertools
import json
import logging
import math
import re

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_postgres.vectorstores import PGVector
from sqlalchemy import text

import tangerine.config as cfg

from .db import db
from .embeddings import embeddings
from .file import File, QualityDetector

log = logging.getLogger("tangerine.vector")

class VectorStoreInterface:
    def __init__(self):
        self.store = None
        self.splitter_chunk_size = 2000
        self.max_chunk_size = 2300
        self.chunk_overlap = 200
        self.batch_size = 32
        self.db = db
        self.search_providers = []
        self.quality_detector = QualityDetector()
        self._embeddings = embeddings

    def initialize(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection=cfg.DB_URI,
                embeddings=self._embeddings,
            )
            self._ensure_fts_vector_column()
        except Exception:
            log.exception("error initializing vector store")

        self.quality_detector.initialize_model()

    def combine_small_chunks(self, chunks):
        """
        Keep merging small chunks into the next one until the result is large enough.
        """
        merged_chunks = []
        buffer = ""

        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) == 0:  # Ignore empty chunks
                continue

            combined_size = len(buffer) + len(chunk)
            if combined_size > self.max_chunk_size:
                # the buffer is too big to add this additional chunk, append the buffer and reset
                merged_chunks.append(buffer)
                buffer = chunk
                continue

            # the buffer is small enough, combine this chunk into it...
            buffer += "\n\n" + chunk
            buffer = buffer.strip()

        # if anything is still left over in the buffer, append it
        if buffer:
            merged_chunks.append(buffer)
        return merged_chunks

    def _load_sql_file(self, sql_file):
        """Load a SQL file by name"""
        sql_query = (
            importlib.resources.files("tangerine.sql").joinpath(sql_file).read_text()
        )
        return text(sql_query)

    def _ensure_fts_vector_column(self):
        """Ensure the 'fts_vector' column exists in the 'langchain_pg_embedding' table."""
        with self.db.engine.connect() as conn:
            # Check if the column already exists
            check_col = conn.execute(self._load_sql_file("tsvector_check.sql")).fetchone()
            if not check_col:
                # Add the computed tsvector column
                log.info("Adding fts_vector column to langchain_pg_embedding table")
                conn.execute(self._load_sql_file("add_tsvector_column.sql"))
                conn.execute(self._load_sql_file("index_tsvector_column.sql"))
                conn.commit()

    def has_markdown_headers(self, text):
        """Checks if a document contains markdown headers."""
        return bool(re.search(r"^#{1,6} ", text, re.MULTILINE))

    def split_to_document_chunks(self, text, metadata) -> list[Document]:
        """Split documents into chunks. Use markdown-aware splitter first if text is markdown."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.splitter_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", ". ", "? ", "! ", "\n", " ", ""],
        )

        md_splitter = MarkdownHeaderTextSplitter(
            strip_headers=False,
            headers_to_split_on=[
                ("#", "H1"),
                ("##", "H2"),
                ("###", "H3"),
                ("####", "H4"),
                ("#####", "H5"),
                ("######", "H6"),
            ],
        )

        if self.has_markdown_headers(text):
            markdown_documents = md_splitter.split_text(text)
            chunks = text_splitter.split_documents(markdown_documents)
            # convert back to list[str] so we can filter and combine them
            # TODO: figure out how to combine but also retain md header metadata?
            chunks = [chunk.page_content for chunk in chunks]
        else:
            chunks = text_splitter.split_text(text)

        if cfg.ENABLE_QUALITY_DETECTION:
            desired_quality = "prose"
            chunks = self.quality_detector.filter_by_quality(chunks, desired_quality)

        chunks = self.combine_small_chunks(chunks)

        # filter out any empty chunks
        len_pre_filter = len(chunks)
        chunks = list(filter(lambda c: c.strip() != "", chunks))
        len_post_filter = len(chunks)
        len_diff = len_pre_filter - len_post_filter

        if len_diff:
            log.debug("dropped %d empty chunks", len_diff)

        # Convert back to Document objects for call to 'embed_documents'
        documents = []
        for chunk in chunks:
            if cfg.EMBED_DOCUMENT_PREFIX:
                chunk = f"{cfg.EMBED_DOCUMENT_PREFIX}: {chunk}"
            documents.append(Document(page_content=chunk, metadata=metadata))

        return documents

    def create_document_chunks(self, file: File, agent_id: int) -> list[Document]:
        log.debug("creating doc chunks for %s", file)

        text = file.extract_text()

        if not text:
            log.error("file %s: empty text", file)
            return []

        metadata = {
            "agent_id": str(agent_id),
        }
        metadata.update(file.metadata)

        documents = self.split_to_document_chunks(text, metadata)

        if documents:
            doc_sizes = ",".join([str(len(d.page_content)) for d in documents])
            log.debug("file %s split into %d chunks of sizes %s", file, len(documents), doc_sizes)
        else:
            log.debug("file %s split into 0 chunks")

        return documents

    def add_file(self, file: File, agent_id: int):
        documents = []
        try:
            documents = self.create_document_chunks(file, agent_id)
        except Exception:
            log.exception("error creating document chunks")
            return

        total = len(documents)
        batch_size = self.batch_size
        total_batches = math.ceil(total / batch_size)
        for idx, batch in enumerate(itertools.batched(documents, batch_size)):
            size = 0
            for doc in batch:
                size += len(doc.page_content)
            current_batch = idx + 1
            log.debug(
                "adding batch %d/%d for file %s to agent %s (%d chunks, total size: %d chars)",
                current_batch,
                total_batches,
                file,
                agent_id,
                len(batch),
                size,
            )
            try:
                self.store.add_documents(batch)
            except Exception:
                log.exception(
                    "error on batch %d/%d for file %s", current_batch, total_batches, file
                )

    def _build_metadata_filter(self, metadata):
        filter_stmts = []

        metadata_as_str = {key: str(val) for key, val in metadata.items()}

        for key in metadata_as_str.keys():
            # use parameterized query
            filter_stmt = f"cmetadata->>'{key}' = :{key}"
            filter_stmts.append(filter_stmt)

        filter_ = " AND ".join(filter_stmts)

        return metadata_as_str, filter_

    def get_distinct_cmetadata(self, search_filter):
        if not search_filter:
            raise ValueError("empty metadata")

        metadata_as_str, filter_ = self._build_metadata_filter(search_filter)
        query = text(
            f"SELECT distinct on (cmetadata) cmetadata from langchain_pg_embedding where {filter_}"
        )
        results = db.session.execute(query, metadata_as_str).all()

        return [row.cmetadata for row in results]

    def get_ids_and_cmetadata(self, search_filter):
        if not search_filter:
            raise ValueError("empty metadata")

        metadata_as_str, filter_ = self._build_metadata_filter(search_filter)
        query = text(f"SELECT id, cmetadata FROM langchain_pg_embedding WHERE {filter_}")
        results = db.session.execute(query, metadata_as_str).all()

        return results

    def delete_document_chunks_by_id(self, ids):
        log.debug("deleting %d document chunks from vector store", len(ids))
        self.store.delete(ids)

    def delete_document_chunks(self, search_filter: dict) -> dict:
        results = self.get_ids_and_cmetadata(search_filter)

        matching_docs = []
        for result in results:
            # add document id into each result
            result.cmetadata["id"] = result.id
            matching_docs.append(result.cmetadata)

        log.debug(
            "found %d doc(s) from vector DB matching filter: %s", len(matching_docs), search_filter
        )

        self.delete_document_chunks_by_id([doc["id"] for doc in matching_docs])

        return matching_docs

    def update_cmetadata(self, metadata: dict, search_filter: dict, commit: bool = True):
        metadata_as_str, filter_ = self._build_metadata_filter(search_filter)
        data = {key: str(val) for key, val in metadata.items()}
        update = (
            "UPDATE langchain_pg_embedding "
            f"SET cmetadata = cmetadata || '{json.dumps(data)}' "
            f"WHERE {filter_}"
        )
        db.session.execute(text(update), metadata_as_str)
        if commit:
            db.session.commit()

    def set_doc_states(self, active: bool, pending_removal: bool, search_filter: dict):
        metadata = {"active": str(active), "pending_removal": str(pending_removal)}
        self.update_cmetadata(metadata, search_filter)

    def get_search_filter(self, agent_id):
        return {"agent_id": str(agent_id), "active": "True"}


vector_db = VectorStoreInterface()
