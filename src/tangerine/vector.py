import importlib.resources
import io
import itertools
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from decimal import Decimal
from typing import Optional

import httpx
from httpx_retries import Retry, RetryTransport
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

import tangerine.config as cfg
import tangerine.llm as llm
from tangerine.metrics import get_counter

from .file import File, QualityDetector
from .models.agent import db

log = logging.getLogger("tangerine.vector")
embed_prompt_tokens_metric = get_counter(
    "embed_prompt_tokens", "Embedding model prompt tokens usage"
)


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

    def __init__(self, store: VectorStore):
        if self.RETRIEVAL_METHOD is None:
            raise TypeError("Subclasses must set RETRIEVAL_METHOD to a non-None value")
        self.store = store
        log.debug("initializing search provider %s", self.__class__.__name__)

    @abstractmethod
    def search(self, query, search_filter, query_embedding=None) -> list[SearchResult]:
        """Runs the search and returns results with normalized scores."""
        pass

    def _add_retrieval_method(self, docs):
        """Add retrieval method to each document's metadata."""
        for doc, _ in docs:
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

    def search(self, query, search_filter, query_embedding) -> list[SearchResult]:
        """Runs MMR search and normalizes scores."""
        results = self.store.max_marginal_relevance_search_with_score_by_vector(
            embedding=query_embedding,
            filter=search_filter,
            lambda_mult=0.85,
            k=3,
        )
        results = self._process_results(results)

        # Normalize (Invert distance-based scores)
        return [SearchResult(doc, 1 - score) for doc, score in results]


class SimilaritySearchProvider(SearchProvider):
    """Cosine Similarity Search Provider."""

    RETRIEVAL_METHOD = "similarity"

    def search(self, query, search_filter, query_embedding) -> list[SearchResult]:
        """Runs similarity search and ensures scores are normalized."""
        results = self.store.similarity_search_with_score_by_vector(
            embedding=query_embedding,
            filter=search_filter,
            k=3,
        )
        results = self._process_results(results)

        # Assume scores are already in 0-1 range (cosine similarity)
        return [SearchResult(doc, score) for doc, score in results if score >= 0.3]


class HybridSearchProvider(SearchProvider):
    """Hybrid Search combining Vector Similarity and Full-Text BM25 Search in PGVector."""

    RETRIEVAL_METHOD = "hybrid"
    QUERY_FILE = "hybrid_search.sql"

    def __init__(self, store):
        super().__init__(store)
        self.sql_loaded = False
        self.sql_query = ""
        self._load_sql_file()

    def _load_sql_file(self):
        """Loads an SQL file into memory."""
        try:
            self.sql_query = (
                importlib.resources.files("tangerine.sql").joinpath(self.QUERY_FILE).read_text()
            )
            self.sql_loaded = True
            log.debug("hybrid search SQL query loaded")
        except Exception:
            self.sql_loaded = False
            log.exception("Error loading SQL file %s", self.QUERY_FILE)

    def search(self, query, search_filter, query_embedding) -> list[SearchResult]:
        """Hybrid search provider combining vector similarity and full-text BM25 search.

        Based on:
        * https://github.com/pgvector/pgvector-python/tree/master/examples/hybrid_search
        * https://python.langchain.com/docs/how_to/hybrid/
        """

        if not self.sql_loaded:
            log.error("SQL file not loaded, cannot run hybrid search")
            return []

        try:
            # Convert list to vectors
            query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            hybrid_search_sql = text(self.sql_query)

            results = db.session.execute(
                hybrid_search_sql,
                {
                    "query": query,
                    "embedding": query_embedding_str,
                    "agent_id": search_filter["agent_id"],
                },
            ).fetchall()

            # Process results into LangChain's SearchResult format
            processed_results = []
            for row in results:
                doc = Document(
                    page_content=row[2], metadata={"retrieval_method": self.RETRIEVAL_METHOD}
                )
                # Convert decimal to float to preserve compatibility with the other search providers
                score = float(row[1]) if isinstance(row[1], Decimal) else row[1]
                processed_results.append((doc, score))

            results = self._process_results(processed_results)
            return [SearchResult(doc, score) for doc, score in results]
        except Exception:
            log.exception("Error running hybrid search")
            return []


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

        if not usage:
            log.debug("no 'usage' in embedding response")
            return

        try:
            prompt_tokens = int(usage.get("prompt_tokens", 0))
        except ValueError:
            log.debug("invalid 'usage' content in embedding response")
            return

        log.debug("embedding prompt tokens: %d", prompt_tokens)
        embed_prompt_tokens_metric.inc(prompt_tokens)


# base this on top of RetryTransport so that we can configure http backoff timers
class CustomTransport(RetryTransport):
    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)

        return CustomResponse(
            status_code=response.status_code,
            headers=response.headers,
            stream=response.stream,
            extensions=response.extensions,
        )


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

        self._embeddings = OpenAIEmbeddings(
            http_client=httpx.Client(
                transport=CustomTransport(
                    retry=Retry(
                        total=3,
                        backoff_factor=0.5,
                        max_backoff_wait=30,
                        status_forcelist=[429, 502, 503],  # intentionally omitting 504
                    )
                )
            ),
            max_retries=0,  # disable openai client's built-in retry mechanism
            model=cfg.EMBED_MODEL_NAME,
            openai_api_base=cfg.EMBED_BASE_URL,
            openai_api_key=cfg.EMBED_API_KEY,
            check_embedding_ctx_length=False,
        )

    def initialize(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection=cfg.DB_URI,
                embeddings=self._embeddings,
            )
        except Exception:
            log.exception("error initializing vector store")

        self.search_providers = [
            MMRSearchProvider(self.store),
            SimilaritySearchProvider(self.store),
        ]

        if cfg.ENABLE_HYBRID_SEARCH:
            self.search_providers.append(HybridSearchProvider(self.store))

        self.quality_detector.initialize_model()

    def embed_query(self, query: str) -> Optional[Embeddings]:
        if cfg.EMBED_QUERY_PREFIX:
            query = f"{cfg.EMBED_QUERY_PREFIX}: {query}"
        return self._embeddings.embed_query(query)

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

    def deduplicate_results(self, results, threshold=0.90):
        """
        Removes near-duplicate search results based on text similarity.

        - `threshold=0.90` means chunks with >= 90% text similarity are considered duplicates.
        - Keeps only the highest-ranked unique chunk.
        """
        if not results:
            return []

        vectorizer = TfidfVectorizer().fit_transform([r.document.page_content for r in results])
        similarities = cosine_similarity(vectorizer)

        unique_results = []
        indicies_of_dups = set()

        for i, result in enumerate(results):
            if i in indicies_of_dups:
                continue  # Skip already marked duplicates

            # Mark similar results as duplicates
            for j in range(i + 1, len(results)):
                if similarities[i, j] > threshold:
                    indicies_of_dups.add(j)

            unique_results.append(result)

        return unique_results

    def _rerank_results(self, query, search_results):
        """
        Uses the LLM to rank search results based on relevance.
        """
        response = llm.rerank(query, search_results)

        valid_rankings = list(range(0, len(search_results)))
        rankings = [int(num) - 1 for num in response.split(",")]
        log.debug("model response rankings: %s, valid rankings: %s", rankings, valid_rankings)
        if not rankings or not all([r in valid_rankings for r in rankings]):
            raise ValueError(
                f"Invalid model rankings: {rankings}, "
                f"valid rankings: {valid_rankings}, "
                f"model response: {response}"
            )

        # Sort results based on LLM ranking
        sorted_results = [search_results[i] for i in rankings]

        return sorted_results

    def search(self, query, embedding, agent_id: int):
        results = []
        search_filter = {"agent_id": str(agent_id), "active": "True"}

        query_embedding = embedding or self.embed_query(query)
        for provider in self.search_providers:
            results.extend(provider.search(query, search_filter, query_embedding))

        # Remove duplicates based on text similarity
        deduped_results = self.deduplicate_results(results)

        # Rank the results using LLM if enabled, otherwise by score
        sorted_results = None
        if cfg.ENABLE_RERANKING:
            try:
                sorted_results = self._rerank_results(query, deduped_results)
            except Exception:
                log.exception("model re-ranking failed")

        if not sorted_results:
            log.debug("sorting results by score")
            sorted_results = sorted(deduped_results, key=lambda r: r.score, reverse=True)

        return sorted_results

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


vector_db = VectorStoreInterface()
