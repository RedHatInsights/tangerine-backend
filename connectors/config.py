import os


def _is_true(env_var):
    return str(os.getenv(env_var, "false")).lower() in [
        "1",
        "t",
        "true",
    ]


LOG_LEVEL_GLOBAL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_LEVEL_APP = os.getenv("LOG_LEVEL", "DEBUG").upper()

DB_USERNAME = os.getenv("DB_USERNAME", "citrus")
DB_PASSWORD = os.getenv("DB_PASSWORD", "citrus")
DB_NAME = os.getenv("DB_NAME", "citrus")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_URI = f"postgresql+psycopg://{DB_USERNAME}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
VECTOR_COLLECTION_NAME = os.getenv("VECTOR_COLLECTION_NAME", "collection")

SQLALCHEMY_POOL_SIZE = int(os.getenv("SQLALCHEMY_POOL_SIZE", 30))
SQLALCHEMY_MAX_OVERFLOW = int(os.getenv("SQLALCHEMY_MAX_OVERFLOW", 10))

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

S3_SYNC_CONFIG_FILE = os.getenv("S3_SYNC_CONFIG_FILE", "s3.yaml")
S3_SYNC_EXPORT_METRICS = _is_true("S3_SYNC_EXPORT_METRICS")
S3_SYNC_EXPORT_METRICS_SLEEP_SECS = int(os.getenv("S3_SYNC_EXPORT_METRICS_SLEEP_SECS", 60))

METRICS_PREFIX = os.getenv("METRICS_PREFIX", "tangerine")

USER_PROMPT_TEMPLATE = """
[INST]
Question: {question}

Answer the above question using the below search results as context:

{context}
[/INST]
""".lstrip(
    "\n"
).rstrip(
    "\n"
)

_prompt = """
<s>[INST] You are a helpful assistant that helps software developers quickly find answers to their
questions by reviewing technical documents. You will be provided with a question and search results
that are relevant for answering the question. The start marker for each search result is similar to
this: <<Search result 1>>. If the title of the document is known, then the start marker result is
similar to this: <<Search result 1, Document title: An Example Title>>. The end marker of each
search result is similar to this: <<Search result 1 END>>. The content of the search result is
found between the start marker and the end marker and is a snippet of technical documentation in
markdown format. The search results are ordered according to relevance with the most relevant
search result listed first. Answer the question using the search results as context. Answer as
concisely as possible. If the first search result provides enough information to answer the
question, just use that single search result as context and discard the others. Your answers must
be based solely on the content found in the search results. Format your answers in markdown for
easy readability. If you are not able to answer a question, you should say "I do not have enough
information available to be able to answer your question." Answers must consider chat history.
[/INST]
"""

DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", _prompt).lstrip("\n").replace("\n", " ")
