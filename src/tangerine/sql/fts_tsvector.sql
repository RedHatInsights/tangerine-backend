SELECT id, document, cmetadata, ts_rank_cd(fts_vector, plainto_tsquery(:q)) AS rank
FROM langchain_pg_embedding
WHERE fts_vector @@ plainto_tsquery(:q)
    AND cmetadata->>'agent_id' = :agent_id
ORDER BY rank DESC
LIMIT 4;