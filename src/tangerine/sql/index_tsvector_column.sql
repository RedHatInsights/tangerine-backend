CREATE INDEX idx_langchain_pg_embedding_fts
ON langchain_pg_embedding USING GIN (fts_vector);