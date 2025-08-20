WITH fts_results AS (
    SELECT
        id,
        document,
        cmetadata,
        RANK() OVER (
            ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) DESC
        ) AS rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'knowledgebase_id' = ANY(:knowledgebase_ids)
        AND cmetadata->>'active' = 'True'
        AND fts_vector @@ plainto_tsquery('english', :query)
    ORDER BY
        ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) DESC
    LIMIT 10
),
vector_results AS (
    SELECT
        id,
        document,
        cmetadata,
        RANK() OVER (
            ORDER BY -(embedding <#> :embedding) DESC
        ) AS rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'knowledgebase_id' = ANY(:knowledgebase_ids)
        AND cmetadata->>'active' = 'True'
    ORDER BY
        -(embedding <#> :embedding) DESC
    LIMIT 10
)
SELECT
    COALESCE(fts_results.id, vector_results.id) AS id,
    COALESCE(fts_results.document, vector_results.document) AS document,
    COALESCE(fts_results.cmetadata, vector_results.cmetadata) AS cmetadata,
    -- Weighing 70% of the score to fts results, 30% of the score to vector results
    (
        COALESCE(1 / (1 + fts_results.rank)::numeric, 0.0) * 0.7
    ) + (
        COALESCE(1 / (1 + vector_results.rank)::numeric, 0.0) * 0.3
    ) AS rrf_score
FROM
    fts_results
FULL OUTER JOIN
    vector_results
    ON fts_results.id = vector_results.id
ORDER BY
    rrf_score DESC
LIMIT 6;
