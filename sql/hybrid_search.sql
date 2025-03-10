-- based on: https://github.com/pgvector/pgvector-python/blob/7e7a851d413d07cd7701ba7c411ed3093b531934/examples/hybrid_search/rrf.py#L24-L46
WITH semantic_search AS (
    SELECT id, document, cmetadata,
        -- Normalize vector similarity to [0,1]
        1.0 - (embedding <=> CAST(:embedding AS vector)) AS norm_vector_score,
        RANK() OVER (ORDER BY embedding <=> CAST(:embedding AS vector)) AS rank
    FROM langchain_pg_embedding
    WHERE cmetadata->>'active' = 'True' AND
        cmetadata->>'agent_id' = :agent_id
    ORDER BY embedding <=> CAST(:embedding AS vector)
    LIMIT 20
),
keyword_search AS (
    SELECT id, document,
        -- Normalize BM25 scores by scaling to the highest score in the batch
        ts_rank_cd(to_tsvector('english', document), plainto_tsquery(:query)) /
        (SELECT MAX(ts_rank_cd(to_tsvector('english', document), plainto_tsquery(:query))) FROM langchain_pg_embedding) AS norm_bm25_score,
        RANK() OVER (ORDER BY ts_rank_cd(to_tsvector('english', document), plainto_tsquery(:query)) DESC) AS rank
    FROM langchain_pg_embedding, plainto_tsquery('english', :query) query
    WHERE to_tsvector('english', document) @@ query
    ORDER BY ts_rank_cd(to_tsvector('english', document), query) DESC
    LIMIT 20
)
SELECT
    COALESCE(semantic_search.id, keyword_search.id) AS id,
    -- Use normalized scores directly instead of reciprocal ranking
    COALESCE(semantic_search.norm_vector_score, 0.0) * 0.5 +
    COALESCE(keyword_search.norm_bm25_score, 0.0) * 0.5 AS score,
    COALESCE(semantic_search.document, keyword_search.document) AS document
FROM semantic_search
FULL OUTER JOIN keyword_search ON semantic_search.id = keyword_search.id
ORDER BY score DESC
LIMIT 4;
