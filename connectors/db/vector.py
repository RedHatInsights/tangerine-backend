import io
import itertools
import json
import logging
import math
from abc import ABC, abstractmethod

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


class SearchResult:
    """Class to hold search results with document and score."""
    def __init__(self, document: str, score: float):
        self.document = document
        self.score = score

# Search providers let us swap out different search algorithms
# without changing the interface
class SearchProvider(ABC):
    """Abstract base class for search providers."""

    RETRIEVAL_METHOD = None

    def __init__(self, store):
            if self.RETRIEVAL_METHOD is None:
                raise TypeError("Subclasses must set RETRIEVAL_METHOD to a non-None value")
            self.store = store

    @abstractmethod
    def search(self, query, search_filter) -> list[SearchResult]:
        """Runs the search and returns results with normalized scores."""
        pass

    def _add_retrieval_method(self, docs):
        """Add retrieval method to each document's metadata."""
        for doc, _score in docs:
            doc.metadata["retrieval_method"] = self.RETRIEVAL_METHOD
        return docs
    
    def _add_relevance_score(self, docs):
        """Add relevance score to each document's metadata."""
        for doc, score in docs:
            doc.metadata["relevance_score"] = score
        return docs
    
    def _process_results(self, results):
        """Process the results and return a list of SearchResult."""
        results = self._add_retrieval_method(results)
        results = self._add_relevance_score(results)
        return results


class MMRSearchProvider(SearchProvider):
    """Maximal Marginal Relevance (MMR) Search Provider."""

    RETRIEVAL_METHOD = "mmr"

    def search(self, query, search_filter) -> list[SearchResult]:
        """Runs MMR search and normalizes scores."""
        results = self.store.max_marginal_relevance_search_with_score(
            query=query,
            filter=search_filter,
            lambda_mult=0.7,
            k=4,
        )
        results = self._process_results(results)
        # Normalize (Invert distance-based scores)
        return [SearchResult(doc, 1 - score) for doc, score in results]


class SimilaritySearchProvider(SearchProvider):
    """Cosine Similarity Search Provider."""

    RETRIEVAL_METHOD = "similarity"

    def search(self, query, search_filter) -> list[SearchResult]:
        """Runs similarity search and ensures scores are normalized."""
        results = self.store.similarity_search_with_score(
            query=query,
            filter=search_filter,
            k=2,
        )
        results = self._process_results(results)
        # Assume scores are already in 0-1 range (cosine similarity)
        return [SearchResult(doc, score) for doc, score in results]



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
        self.search_providers = []

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
        self.search_providers = [
            MMRSearchProvider(self.store),
            SimilaritySearchProvider(self.store),
        ]

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

    def split_to_document_chunks(self, text_to_split, metadata):
        # find title if possible and add to metadata
        for line in text_to_split.splitlines():
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

        chunks = text_splitter.split_text(text_to_split)
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
        results = []
        search_filter = {"agent_id": str(agent_id), "active": "True"}
        if cfg.EMBED_QUERY_PREFIX:
            query = f"{cfg.EMBED_QUERY_PREFIX}: {query}"

        for provider in self.search_providers:
            results.extend(provider.search(query, search_filter))

        # Sort the results by relevance score
        results.sort(key=lambda result: result.score, reverse=True)

        # de-dupe, 'Document' is unhashable so check page content
        unique_results = []
        seen_pages = set()
        for new_result in results:
            page_text = new_result.page_content
            if page_text not in seen_pages:
                seen_pages.add(page_text)
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
        results = self.get_ids_and_cmetadata(filter)

        matching_docs = []
        for result in results:
            # add document id into each result
            result.cmetadata["id"] = result.id
            matching_docs.append(result.cmetadata)

        log.debug("found %d doc(s) from vector DB matching filter: %s", len(matching_docs), search_filter)

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


vector_db = VectorStoreInterface()
