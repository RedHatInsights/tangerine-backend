import io
import itertools
import json
import logging
import math
from operator import itemgetter

import httpx
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sqlalchemy import text

import connectors.config as cfg
from resources.metrics import get_counter

from .agent import db
from .file import File

log = logging.getLogger("tangerine.db.vector")
embed_prompt_tokens_metric = get_counter(
    "embed_prompt_tokens", "Embedding model prompt tokens usage"
)

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

# because we currently cannot access usage_metadata for embedding calls nor use
# get_openai_callback() in the same way we can for chat model calls...
# (see https://github.com/langchain-ai/langchain/issues/945)
#
# we use a work-around inspired by https://github.com/encode/httpx/discussions/3073


class CustomResponse(httpx.Response):
    def iter_bytes(self, *args, **kwargs):
        content = io.BytesIO()

        # copy the chunk into our own buffer but yield same chunk to caller
        for chunk in super().iter_bytes(*args, **kwargs):
            content.write(chunk)
            yield chunk

        # check to see if content can be loaded as json and look for 'usage' key
        content.seek(0)
        try:
            usage = json.load(content).get("usage")
        except json.JSONDecodeError:
            usage = {}

        try:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
        except ValueError:
            prompt_tokens = 0

        log.debug("embedding prompt tokens: %d", prompt_tokens)
        embed_prompt_tokens_metric.inc(prompt_tokens)


class CustomTransport(httpx.BaseTransport):
    def __init__(self, transport: httpx.BaseTransport):
        self.transport = transport

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = self.transport.handle_request(request)

        return CustomResponse(
            status_code=response.status_code,
            headers=response.headers,
            stream=response.stream,
            extensions=response.extensions,
        )


class VectorStoreInterface:
    def __init__(self):
        self.store = None
        self.vector_chunk_size = 2000
        self.vector_chunk_overlap = 0
        self.batch_size = 32
        self.db = db

        self.embeddings = OpenAIEmbeddings(
            http_client=httpx.Client(transport=CustomTransport(httpx.HTTPTransport())),
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
        # find title if possible and add to metadata
        for line in text.splitlines():
            if line.startswith("# "):
                # we found a title header, add it to metadata
                metadata["title"] = line.lstrip("# ").strip()
                break
        else:
            metadata["title"] = metadata["full_path"]

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            separators=TXT_SEPARATORS,  # todo: split on more doc types
        )

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

        return chunks

    def add_file(self, file: File, agent_id: int):
        chunks = []
        try:
            chunks = self.create_document_chunks(file, agent_id)
        except Exception:
            log.exception("error creating document chunks")
            return

        total = len(chunks)
        batch_size = self.batch_size
        total_batches = math.ceil(total / batch_size)
        for idx, batch in enumerate(itertools.batched(chunks, batch_size)):
            current_batch = idx + 1
            log.debug(
                "adding %d document chunks to agent %s, batch %d/%d",
                len(batch),
                agent_id,
                current_batch,
                total_batches,
            )
            try:
                self.store.add_documents(batch)
            except Exception:
                log.exception("error adding documents to vector store for batch %d", current_batch)

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

        for key in metadata_as_str.keys():
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

    def update_cmetadata(self, metadata: dict, filter: dict, commit: bool = True):
        metadata_as_str, filter_ = self._build_metadata_filter(filter)
        data = {key: str(val) for key, val in metadata.items()}
        update = (
            "UPDATE langchain_pg_embedding "
            f"SET cmetadata = cmetadata || '{json.dumps(data)}' "
            f"WHERE {filter_}"
        )
        db.session.execute(text(update), metadata_as_str)
        if commit:
            db.session.commit()

    def set_doc_states(self, active: bool, pending_removal: bool, filter: dict):
        metadata = {"active": str(active), "pending_removal": str(pending_removal)}
        self.update_cmetadata(metadata, filter)


vector_db = VectorStoreInterface()
