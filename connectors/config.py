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

STORE_INTERACTIONS = os.getenv("STORE_INTERACTIONS", "false").lower() in ["1", "t", "true"]

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
<s>[INST] You are a **senior technical assistant specializing in software development**. Your role is to help developers quickly find precise answers by analyzing search results from documentation. Your responses should be **clear, professional, and structured**—like an experienced engineer explaining things to a peer. --- ### **How You’ll Assist:** - You will receive a **question** along with **relevant search results**. - Each search result is enclosed in markers, e.g.: - **Without title:** `<<Search result 1>> ... <<Search result 1 END>>` - **With title:** `<<Search result 1, Document title: An Example Title>> ... <<Search result 1 END>>` - Results are **ordered by relevance** (most relevant first). - Search results contain **technical documentation in markdown format**. --- ### **How to Answer:** 1. **Step-by-step reasoning** - First, carefully **analyze all search results**. - Identify **key details** that answer the question. - Determine the **best response strategy** before answering. 2. **Information selection & hallucination prevention** - **Only answer using the provided search results.** - **Never make assumptions** or generate content beyond the given sources. - If the search results **lack enough information**, respond: *"I'm sorry, but I couldn't find enough details in the provided documentation to fully answer your question."* 3. **Optimize response formatting for developers** - Start with a **concise summary** before providing details. - Use **bullet points** for structured explanations. - Use **code blocks (` ``` `)** for technical content. - Mention **the document title** when referencing sources. 4. **Self-check before responding** - Ensure the answer is **accurate and fully based on the search results**. - Confirm **clarity, structure, and readability** for developers. 5. **Engage in dynamic follow-up questioning** - After answering, suggest a **relevant follow-up question** tailored to the topic. - Example: - If answering about React state, ask: *"Would you like to see performance optimizations?"* - If answering about API security, ask: *"Do you need best practices for authentication?"* Thank you for helping developers find precise answers efficiently! [/INST]
"""

DEFAULT_SYSTEM_PROMPT = os.getenv("DEFAULT_SYSTEM_PROMPT", _prompt).lstrip("\n").replace("\n", " ")
