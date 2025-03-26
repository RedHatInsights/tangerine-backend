WITH fts_results AS (
    SELECT
        id,
        document,
        cmetadata,
        RANK() OVER (ORDER BY ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) DESC) AS rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'agent_id' = :agent_id
        AND cmetadata->>'active' = 'True'
        AND fts_vector @@ plainto_tsquery('english', :query)
    ORDER BY
        ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) DESC
    LIMIT 20
),
vector_results AS (
    SELECT
        id,
        document,
        cmetadata,
        RANK() OVER (ORDER BY -(embedding <#> :embedding) DESC) AS rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'agent_id' = :agent_id
        AND cmetadata->>'active' = 'True'
    ORDER BY
        -(embedding <#> :embedding) DESC
    LIMIT 20
)
SELECT
    COALESCE(fts_results.id, vector_results.id) AS id,
    COALESCE(fts_results.document, vector_results.document) AS document,
    COALESCE(fts_results.cmetadata, vector_results.cmetadata) AS cmetadata,
    COALESCE(1 / (50 + fts_results.rank) * 0.7, 0.0) + COALESCE(1 / (50 + vector_results.rank) * 0.3, 0.0) AS rrf_score -- k = 50, weighting fts results higher
FROM
    fts_results
FULL OUTER JOIN
    vector_results
    ON fts_results.id = vector_results.id
ORDER BY
    rrf_score DESC
LIMIT 5;
