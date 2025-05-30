SELECT
    id,
    document,
    cmetadata,
    ts_rank_cd(fts_vector, plainto_tsquery('english', :query)) AS score
FROM
    langchain_pg_embedding
WHERE
    cmetadata->>'assistant_id' = ANY(:assistant_ids)
    AND cmetadata->>'active' = 'True'
    AND fts_vector @@ plainto_tsquery('english', :query)
ORDER BY
    score DESC
LIMIT 4;
