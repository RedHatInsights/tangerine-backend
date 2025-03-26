WITH fts_results AS (
    SELECT
        id,
        document,
        cmetadata,
        ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) AS fts_rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'agent_id' = :agent_id AND cmetadata->>'active' = 'True' AND fts_vector @@ plainto_tsquery('english', :query)
),
vector_results AS (
    SELECT
        id,
        document,
        cmetadata,
        -(embedding <#> :embedding) AS vector_rank
    FROM
        langchain_pg_embedding
    WHERE
        cmetadata->>'agent_id' = :agent_id AND cmetadata->>'active' = 'True'
    ORDER BY
        vector_rank DESC
    LIMIT 100  -- Limit the number of candidates to refine the query performance
)
SELECT
    fts_results.id,
    fts_results.document,
    fts_results.cmetadata,
    (1 / (1 + fts_results.fts_rank)) + (1 / (1 + vector_results.vector_rank)) AS rrf_score
FROM
    fts_results
JOIN
    vector_results
    ON fts_results.id = vector_results.id
ORDER BY
    rrf_score DESC
LIMIT 4;
