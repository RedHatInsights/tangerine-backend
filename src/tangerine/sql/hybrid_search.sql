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
    LIMIT 20
)
SELECT
    COALESCE(fts_results.id, vector_results.id) AS id,
    COALESCE(fts_results.document, vector_results.document) AS document,
    COALESCE(fts_results.cmetadata, vector_results.cmetadata) AS cmetadata,
    (1 / (1 + COALESCE(fts_results.fts_rank, 0))) + (1 / (1 + COALESCE(vector_results.vector_rank, 0))) AS rrf_score
FROM
    fts_results
FULL OUTER JOIN
    vector_results
    ON fts_results.id = vector_results.id
ORDER BY
    rrf_score DESC
LIMIT 4;
