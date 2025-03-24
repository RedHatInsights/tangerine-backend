ALTER TABLE langchain_pg_embedding
ADD COLUMN fts_vector tsvector
GENERATED ALWAYS AS (to_tsvector('english', document)) STORED;
