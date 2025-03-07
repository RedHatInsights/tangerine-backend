import io
import os
import itertools
import json
import logging
import math
import re
from abc import ABC, abstractmethod
from decimal import Decimal


import httpx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from sqlalchemy import text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import connectors.config as cfg
from resources.metrics import get_counter

from .agent import db
from .file import File, quality_detector

log = logging.getLogger("tangerine.db.vector")
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
            lambda_mult=0.85,
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
            k=4,
        )
        results = self._process_results(results)
        # Assume scores are already in 0-1 range (cosine similarity)
        return [SearchResult(doc, score) for doc, score in results]


class HybridSearchProvider(SearchProvider):
    """Hybrid Search combining Vector Similarity and Full-Text BM25 Search in PGVector."""

    RETRIEVAL_METHOD = "hybrid"
    QUERY_FILE = os.path.join(os.path.dirname(__file__), "../../sql/hybrid_search.sql")

    def __init__(self, store):
        super().__init__(store)
        self.sql_loaded = False
        self.sql_query = ""
        self._load_sql_file()
        self.embeddings_model = OpenAIEmbeddings(
            http_client=httpx.Client(transport=CustomTransport(httpx.HTTPTransport())),
            model=cfg.EMBED_MODEL_NAME,
            openai_api_base=cfg.EMBED_BASE_URL,
            openai_api_key=cfg.EMBED_API_KEY,
            check_embedding_ctx_length=False,
        )

    def _load_sql_file(self):
        """Loads an SQL file into memory."""
        try:
            with open(self.QUERY_FILE, "r", encoding="utf-8") as file:
                self.sql_query = file.read()
                self.sql_loaded = True
        except Exception:
            self.sql_loaded = False
            log.exception("Error loading SQL file %s", self.QUERY_FILE)


    # This is based on a couple of different sources
    # First, the PGVector hybrid search example project https://github.com/pgvector/pgvector-python/tree/master/examples/hybrid_search
    # Second, LangChain's hybrid search docs https://python.langchain.com/docs/how_to/hybrid/
    def search(self, query, search_filter) -> list[SearchResult]:
        """Hybrid search provider combining vector similarity and full-text BM25 search."""

        if not self.sql_loaded:
            log.error("SQL file not loaded, cannot run hybrid search")
            return []

        try:
            # Get embedding for the quyery
            query_embedding = self.embeddings_model.embed_query(query)

            # Convert list to vectors
            query_embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

            # I cannot claim to know exactly what this does and how and why
            # I largely copied this from the PGVector hybrid search example project
            hybrid_search_sql = text(self.sql_query)

            results = db.session.execute(hybrid_search_sql, {"query": query, "embedding": query_embedding_str, "agent_id": search_filter["agent_id"]}).fetchall()

            # Process results into LangChain's SearchResult format
            processed_results = []
            for row in results:
                doc = Document(page_content=row[2], metadata={"retrieval_method": self.RETRIEVAL_METHOD})
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
        self.splitter_chunk_size = 2000
        self.max_chunk_size = 2300
        self.chunk_overlap = 200
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
            HybridSearchProvider(self.store),
        ]

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

    def split_to_document_chunks(self, text, metadata):
        """Split documents into chunks. Use markdown-aware splitter first if text is markdown."""

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.splitter_chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=["\n\n", ". ", "? ", "! "]
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
            ]
        )

        if self.has_markdown_headers(text):
            markdown_documents = md_splitter.split_text(text)
            chunks = text_splitter.split_documents(markdown_documents)
        else:
            chunks = text_splitter.split_text(text)

        # convert back to plain text so we can combine them
        # TODO: figure out how to combine but also retain md header metadata?
        chunks = [chunk.page_content for chunk in chunks]
        
        desired_quality = "prose"
        chunks = quality_detector.filter_by_quality(chunks, desired_quality)

        chunks = self.combine_small_chunks(chunks)

        # Convert back to Document objects for call to 'embed_documents'
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

        # Check for 

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

    def rank_results_with_llm(self, query, search_results):
        """
        Uses the LLM to rank search results based on relevance.
        Falls back to score-based ranking if LLM fails.
        """
        if len(search_results) <= 1:
            return search_results  # No need to rank if there's only one result

        # Construct prompt
        document_list = "\n".join([f"{i+1}. {result.document.page_content[:300]}" for i, result in enumerate(search_results)])
        prompt = f"""
        You are an AI search assistant. Rank the following search results from most to least relevant to the given query.

        ### Query:
        "{query}"

        ### Documents:
        {document_list}

        ### Instructions for Ranking:
        1. **Prioritize well-written prose** that directly answers the query.
        2. **Do NOT rank tables of contents, lists of links, or navigation menus highly**, as they are not meaningful responses.
        3. **Prefer documents that provide clear, informative, and explanatory content.**
        4. **Ignore documents that only contain a collection of links, bullet points, or raw lists with no explanation.**
        5. **If a document is highly repetitive or contains mostly boilerplate text, rank it lower.**
        6. **Only return a comma-separated list of numbers corresponding to the ranking order. Do NOT include explanations or extra formatting.**
        7. **If you are unsure about a document, you can skip it.**
        8. **Skip any document that starts with the string "Skip to content"**

        ### Example Output:
        1, 3, 5, 2, 4
        """
        reranker = ChatOpenAI(
                    model=cfg.LLM_MODEL_NAME,
                    openai_api_base=cfg.LLM_BASE_URL,
                    openai_api_key=cfg.LLM_API_KEY,
                    temperature=cfg.LLM_TEMPERATURE,
                    )
        try:
            # Send to LLM and get response
            llm_response = reranker.invoke(prompt)
        except Exception as e:
            print(f"model ranking failed: {e}. Falling back to score-based ranking.")
            # Fallback: Sort by raw score (descending)
            return sorted(search_results, key=lambda r: r.score, reverse=True)
        ranking = [int(num.strip()) - 1 for num in llm_response.content.split(",")]

        # Validate ranking output
        if not ranking or max(ranking) >= len(search_results):
            raise ValueError("Invalid model ranking response")

        # Sort results based on LLM ranking
        sorted_results = [search_results[i] for i in ranking if i < len(search_results)]
        return sorted_results

    def search(self, query, agent_id: int):
        results = []
        search_filter = {"agent_id": str(agent_id), "active": "True"}
        if cfg.EMBED_QUERY_PREFIX:
            query = f"{cfg.EMBED_QUERY_PREFIX}: {query}"

        for provider in self.search_providers:
            results.extend(provider.search(query, search_filter))

        # Remove duplicates based on text similarity
        deduped_results = self.deduplicate_results(results)

        # Rank the results using LLM if enabled, otherwise by score
        if cfg.ENABLE_MODEL_RANKING:
            sorted_results = self.rank_results_with_llm(query, deduped_results)
        else:
            sorted_results = sorted(deduped_results, key=lambda r: r.score, reverse=True)

        # de-dupe, 'Document' is unhashable so check page content
        unique_results = []
        seen_pages = set()
        for new_result in sorted_results:
            page_text = new_result.document.page_content
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
