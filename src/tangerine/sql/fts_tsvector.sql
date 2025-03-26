SELECT id, document, cmetadata, ts_rank_cd(fts_vector, plainto_tsquery('english', :q)) AS rank
FROM langchain_pg_embedding
WHERE cmetadata->>'agent_id' = :agent_id AND cmetadata->>'active' = 'True' AND fts_vector @@ plainto_tsquery('english', :q)
ORDER BY rank DESC
LIMIT 4;
