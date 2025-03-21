import importlib.resources
import logging
from abc import ABC, abstractmethod
from decimal import Decimal

from langchain_core.documents import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import text

import tangerine.config as cfg
import tangerine.llm as llm

from .db import db
from .embeddings import embed_query
from .vector import vector_db

log = logging.getLogger("tangerine.search")


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

    def __init__(self):
        if self.RETRIEVAL_METHOD is None:
            raise TypeError("Subclasses must set RETRIEVAL_METHOD to a non-None value")
        log.debug("initializing search provider %s", self.__class__.__name__)

    @abstractmethod
    def search(self, agent_id, query, embedding) -> list[SearchResult]:
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

    def search(self, agent_id, query, embedding) -> list[SearchResult]:
        """Runs MMR search and normalizes scores."""
        search_filter = vector_db.get_search_filter(agent_id)

        results = vector_db.store.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
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

    def search(self, agent_id, query, embedding) -> list[SearchResult]:
        """Runs similarity search and ensures scores are normalized."""
        search_filter = vector_db.get_search_filter(agent_id)

        results = vector_db.store.similarity_search_with_score_by_vector(
            embedding=embedding,
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

    def __init__(self):
        super().__init__()
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

    def search(self, agent_id, query, embedding) -> list[SearchResult]:
        """Hybrid search provider combining vector similarity and full-text BM25 search.

        Based on:
        * https://github.com/pgvector/pgvector-python/tree/master/examples/hybrid_search
        * https://python.langchain.com/docs/how_to/hybrid/
        """

        if not self.sql_loaded:
            log.error("SQL file not loaded, cannot run hybrid search")
            return []

        search_filter = vector_db.get_search_filter(agent_id)

        try:
            # Convert list to vectors
            query_embedding_str = "[" + ",".join(map(str, embedding)) + "]"

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


class SearchEngine:
    def __init__(self):
        self.search_providers = [
            MMRSearchProvider(),
            SimilaritySearchProvider(),
        ]

        if cfg.ENABLE_HYBRID_SEARCH:
            self.search_providers.append(HybridSearchProvider())

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

    def search(self, agent_id, query, embedding=None):
        results = []

        embedding = embedding or embed_query(query)

        for provider in self.search_providers:
            results.extend(provider.search(agent_id, query, embedding))

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


search_engine = SearchEngine()
