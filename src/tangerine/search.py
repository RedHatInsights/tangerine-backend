import importlib.resources
import logging
from abc import ABC, abstractmethod

from langchain_core.documents import Document
from pgvector.sqlalchemy import Vector
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import bindparam, text
from sqlalchemy.types import ARRAY, String

import tangerine.config as cfg
import tangerine.llm as llm

from .db import db
from .embeddings import embed_query
from .vector import vector_db

log = logging.getLogger("tangerine.search")

DEFAULT_FILTER = {
    "active": "True",
}


class SearchResult:
    """Class to hold search results with document and scores."""

    def __init__(
        self, document: Document, score: int | float, rank: int = 0, rrf_score: float = 0.0
    ):
        self.document = document
        self.score = float(score)
        self.rank = 0
        self.rrf_score = 0

    def to_json(self):
        return {
            "document": {
                "content": self.document.page_content,
                "metadata": self.document.metadata,
            },
            "score": self.score,
            "rank": self.rank,
            "rrf_score": self.rrf_score,
        }


# Search providers let us swap out different search algorithms
# without changing the interface
class SearchProvider(ABC):
    """Abstract base class for search providers."""

    RETRIEVAL_METHOD = None
    QUERY_FILE = None

    def __init__(self):
        self.sql_loaded = False
        self.sql_query = ""
        if self.RETRIEVAL_METHOD is None:
            raise TypeError("Subclasses must set RETRIEVAL_METHOD to a non-None value")
        log.debug("initializing search provider %s", self.__class__.__name__)

    @abstractmethod
    def search(self, assistant_ids, query, embedding) -> list[SearchResult]:
        """Runs the search and returns results with normalized scores."""
        pass

    def _set_ranks(self, results: list[SearchResult]) -> list[SearchResult]:
        """
        Sets integer rank on SearchResults

        Default assumes that lower score is a better result, override this func if needed
        """
        sorted_by_best = sorted(results, key=lambda r: r.score, reverse=True)
        for idx, r in enumerate(sorted_by_best):
            r.rank = idx

    def _process_results(self, results: list[SearchResult]) -> list[SearchResult]:
        """Process the results and return a sorted list of SearchResult with ranks."""
        # normalize scores from 0-1
        max_score = max(r.score for r in results)
        min_score = min(r.score for r in results)
        for r in results:
            if max_score == min_score:
                r.score = 1.0
            else:
                r.score = (r.score - min_score) / (max_score - min_score)
            # update metadata
            r.document.metadata["retrieval_method"] = self.RETRIEVAL_METHOD

        # add rank to the results, to be used later for RRF
        self._set_ranks(results)

        sorted_and_ranked_results = sorted(results, key=lambda r: r.rank)

        return sorted_and_ranked_results

    def _load_sql_file(self):
        """Loads an SQL file into memory."""
        try:
            self.sql_query = (
                importlib.resources.files("tangerine.sql").joinpath(self.QUERY_FILE).read_text()
            )
            self.sql_loaded = True
            log.debug("SQL query loaded from file: %s", self.QUERY_FILE)
        except Exception:
            self.sql_loaded = False
            log.exception("Error loading SQL file %s", self.QUERY_FILE)


class FTSPostgresSearchProvider(SearchProvider):
    """PostgreSQL full-text search using the fts_vector column."""

    RETRIEVAL_METHOD = "fts_postgres"
    QUERY_FILE = "fts_tsvector.sql"

    def __init__(self):
        super().__init__()
        self._load_sql_file()

    def _set_ranks(self, results):
        # we handle setting rank in _process_results below, higher score = better result
        pass

    def _process_results(self, results):
        search_results = []

        for idx, row in enumerate(results):
            score = row.score
            doc = Document(id=row.id, page_content=row.document, metadata=row.cmetadata)
            search_results.append(SearchResult(document=doc, score=score, rank=idx))

        return super()._process_results(search_results)

    def _execute_query(self, assistant_ids, query, _embedding):
        fts_query = text(self.sql_query).bindparams(
            bindparam("query", value=query),
            bindparam("assistant_ids", value=assistant_ids, type_=ARRAY(String)),
        )
        results = db.session.execute(fts_query).fetchall()
        return results

    def search(self, assistant_ids, query, embedding) -> list[SearchResult]:
        """Run full-text search over langchain_pg_embedding table."""
        results = None
        if not isinstance(assistant_ids, list):
            assistant_ids = [assistant_ids]
        try:
            results = self._execute_query(assistant_ids, query, embedding)
        except Exception:
            log.exception("error running fts search")

        if not results:
            return []

        return self._process_results(results)


class MMRSearchProvider(SearchProvider):
    """Maximal Marginal Relevance (MMR) Search Provider."""

    RETRIEVAL_METHOD = "mmr"

    def search(self, assistant_ids, query, embedding) -> list[SearchResult]:
        search_filter = vector_db.get_search_filter(assistant_ids)
        results = vector_db.store.max_marginal_relevance_search_with_score_by_vector(
            embedding=embedding,
            filter=search_filter,
            lambda_mult=0.85,
            k=3,
        )
        search_results = [SearchResult(document=doc, score=score) for doc, score in results]
        return self._process_results(search_results)


class SimilaritySearchProvider(SearchProvider):
    """Cosine Similarity Search Provider."""

    RETRIEVAL_METHOD = "similarity"

    def search(self, assistant_ids, query, embedding) -> list[SearchResult]:
        search_filter = vector_db.get_search_filter(assistant_ids)

        results = vector_db.store.similarity_search_with_score_by_vector(
            embedding=embedding,
            filter=search_filter,
            k=3,
        )
        search_results = [SearchResult(document=doc, score=score) for doc, score in results]
        return self._process_results(search_results)


class HybridSearchProvider(SearchProvider):
    """Hybrid Search combining Vector Similarity and Full-Text Search."""

    RETRIEVAL_METHOD = "hybrid"
    QUERY_FILE = "hybrid_search.sql"

    def __init__(self):
        super().__init__()
        self._load_sql_file()

    def _execute_query(self, assistant_ids, query, embedding):
        if not isinstance(assistant_ids, list):
            assistant_ids = [assistant_ids]

        hybrid_search_sql = text(self.sql_query).bindparams(
            bindparam("query", value=query),
            bindparam("assistant_ids", value=assistant_ids, type_=ARRAY(String)),
            bindparam("embedding", value=embedding, type_=Vector()),
        )

        results = db.session.execute(hybrid_search_sql).fetchall()

        return results

    def _set_ranks(self, results):
        # we handle setting rank in _process_results below, higher score = better result
        pass

    def search(self, assistant_ids, query, embedding) -> list[SearchResult]:
        """Hybrid search provider combining vector similarity and full-text BM25 search.

        Based on:
        * https://github.com/pgvector/pgvector-python/tree/master/examples/hybrid_search
        * https://python.langchain.com/docs/how_to/hybrid/
        """

        if not self.sql_loaded:
            log.error("SQL file not loaded, cannot run hybrid search")
            return []

        try:
            results = self._execute_query(assistant_ids, query, embedding)

            # Process results into LangChain's SearchResult format
            search_results = []
            for idx, row in enumerate(results):
                doc = Document(id=row.id, page_content=row.document, metadata=row.cmetadata)
                score = row.rrf_score
                search_results.append(SearchResult(document=doc, score=score, rank=idx))

            return self._process_results(search_results)
        except Exception:
            log.exception("Error running hybrid search")
            return []


class SearchEngine:
    def __init__(self):
        self.search_providers = self._get_search_providers()

    def _get_search_providers(self):
        search_providers = []
        if cfg.ENABLE_MMR_SEARCH:
            search_providers.append(MMRSearchProvider())
        if cfg.ENABLE_SIMILARITY_SEARCH:
            search_providers.append(SimilaritySearchProvider())
        if cfg.ENABLE_FULL_TEXT_SEARCH:
            search_providers.append(FTSPostgresSearchProvider())
        if cfg.ENABLE_HYBRID_SEARCH:
            search_providers.append(HybridSearchProvider())
        return search_providers

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
        rankings = [int(num.strip()) - 1 for num in response.split(",")]
        log.debug("model response rankings: %s, valid rankings: %s", rankings, valid_rankings)
        if not rankings or not all([r in valid_rankings for r in rankings]):
            raise ValueError(
                f"Invalid model rankings: {rankings}, "
                f"valid rankings: {valid_rankings}, "
                f"model response: {response}"
            )

        # Sort results based on LLM ranking
        sorted_results = []
        for idx, doc_num in enumerate(rankings):
            search_result = search_results[doc_num]
            search_result.rank = idx
            search_result.rrf_score = 1 / (1 + search_result.rank)  # for compatability
            sorted_results.append(search_result)

        return sorted_results

    def _sort_using_rrf(self, results: list[SearchResult]):
        log.debug("sorting results with rrf")

        # TODO: incorporate weighted RRF here depending on provider?
        aggregated_results = {}
        for r in results:
            document_id = r.document.id
            if not document_id:
                raise ValueError("document id cannot be 'None'")
            if document_id not in aggregated_results:
                aggregated_results[document_id] = SearchResult(document=r.document, score=0)
            aggregated_results[document_id].rrf_score += 1 / (1 + r.rank)

        aggregated_results = [search_result for _, search_result in aggregated_results.items()]
        # de-dupe after aggregation in case any are highly similar
        deduped_results = self.deduplicate_results(aggregated_results)
        sorted_results = sorted(deduped_results, key=lambda r: r.rrf_score, reverse=True)

        return sorted_results

    def search(self, assistant_ids, query, embedding=None):
        results = []
        embedding = embedding or embed_query(query)
        if not isinstance(assistant_ids, list):
            assistant_ids = [assistant_ids]
        for provider in self.search_providers:
            results.extend(provider.search(assistant_ids, query, embedding))

        return self._finalize_results(query, results)

    def _finalize_results(self, query, results):
        sorted_results = []

        # Rank the results using LLM if enabled, otherwise by score
        if cfg.ENABLE_RERANKING:
            try:
                # de-dupe first before sending to model
                deduped_results = self.deduplicate_results(results)
                sorted_results = self._rerank_results(query, deduped_results)
            except Exception:
                log.exception("model re-ranking failed")

        if not sorted_results:
            sorted_results = self._sort_using_rrf(results)

        for r in sorted_results:
            r.document.metadata["relevance_score"] = r.rrf_score

        # return top 5 results
        return sorted_results[:5]


search_engine = SearchEngine()
