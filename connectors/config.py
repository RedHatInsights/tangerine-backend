import os

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "EMPTY")

CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "mistral")

EMBEDDING_MODEL_SOURCE = os.getenv("EMBEDDING_MODEL_SOURCE", "ollama")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "nomic-embed-text")
TRUST_REMOTE_CODE = os.getenv("TRUST_REMOTE_CODE", "False").lower() in ('true', '1')
