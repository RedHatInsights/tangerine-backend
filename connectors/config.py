import os

DB_USERNAME = os.getenv("DB_USERNAME", "citrus")
DB_PASSWORD = os.getenv("DB_PASSWORD", "citrus")
DB_NAME = os.getenv("DB_NAME", "citrus")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URI = f"postgresql+psycopg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "collection")


LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "EMPTY")
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "mistral")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "EMPTY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")

# for nomic: 'search_query'
# for snowflake-arctic-embed-m-long: 'Represent this sentence for searching relevant passages'
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query")

# for nomic: 'search_document'
# for snowflake-arctic-embed-m-long: ''
EMBED_DOCUMENT_PREFIX = os.getenv("EMBED_DOCUMENT_PREFIX", "")
