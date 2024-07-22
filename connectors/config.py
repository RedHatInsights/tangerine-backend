import os

DB_USERNAME = os.getenv("DB_USERNAME", "citrus")
DB_PASSWORD = os.getenv("DB_PASSWORD", "citrus")
DB_NAME = os.getenv("DB_NAME", "citrus")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URI = f"postgresql://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "collection")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "mistral")

EMBEDDING_MODEL_SOURCE = os.getenv("EMBEDDING_MODEL_SOURCE", "openai")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "False").lower() in ('true', '1')
