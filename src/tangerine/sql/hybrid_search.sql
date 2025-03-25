SELECT
  id,
  document,
  cmetadata,
  ts_rank_cd(fts_vector, plainto_tsquery(:query)) AS fts_score,
  1 - (embedding <#> :embedding) AS vector_score,
  (
    0.7 * ts_rank_cd(fts_vector, plainto_tsquery(:query)) + -- weighting for FTS score
    0.3 * (1 - (embedding <#> :embedding))              -- weighting for vector score
  ) AS hybrid_score
FROM langchain_pg_embedding
WHERE cmetadata->>'agent_id' = :agent_id
ORDER BY hybrid_score DESC
LIMIT 4;
