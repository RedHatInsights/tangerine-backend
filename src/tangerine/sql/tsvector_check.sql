SELECT column_name FROM information_schema.columns
WHERE table_name = 'langchain_pg_embedding' AND column_name = 'fts_vector';
