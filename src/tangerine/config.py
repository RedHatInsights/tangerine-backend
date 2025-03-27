# flake8: noqa: E501

import os


def _is_true(env_var):
    return str(os.getenv(env_var, "false")).lower() in [
        "1",
        "t",
        "true",
    ]


LOG_LEVEL_GLOBAL = os.getenv("LOG_LEVEL_GLOBAL", "INFO").upper()
LOG_LEVEL_APP = os.getenv("LOG_LEVEL_APP", "DEBUG").upper()
DEBUG_VERBOSE = _is_true("DEBUG_VERBOSE")

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

STORE_INTERACTIONS = _is_true("STORE_INTERACTIONS")
ENABLE_RERANKING = _is_true("ENABLE_RERANKING")
ENABLE_QUALITY_DETECTION = _is_true("ENABLE_QUALITY_DETECTION")

ENABLE_HYBRID_SEARCH = _is_true("ENABLE_HYBRID_SEARCH")
ENABLE_MMR_SEARCH = _is_true("ENABLE_MMR_SEARCH")
ENABLE_SIMILARITY_SEARCH = _is_true("ENABLE_SIMILARITY_SEARCH")
ENABLE_FULL_TEXT_SEARCH = _is_true("ENABLE_FULL_TEXT_SEARCH")

EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", "http://localhost:11434/v1")
EMBED_API_KEY = os.getenv("EMBED_API_KEY", "EMPTY")
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "nomic-embed-text")

# for nomic: 'search_query'
# for snowflake-arctic-embed-m-long: 'Represent this sentence for searching relevant passages'
EMBED_QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query")

# for nomic: 'search_document'
# for snowflake-arctic-embed-m-long: ''
EMBED_DOCUMENT_PREFIX = os.getenv("EMBED_DOCUMENT_PREFIX", "search_document")

S3_SYNC_CONFIG_FILE = os.getenv("S3_SYNC_CONFIG_FILE", "s3.yaml")
S3_SYNC_POOL_SIZE = int(os.getenv("S3_SYNC_POOL_SIZE", 15))
S3_SYNC_EXPORT_METRICS = _is_true("S3_SYNC_EXPORT_METRICS")
S3_SYNC_EXPORT_METRICS_SLEEP_SECS = int(os.getenv("S3_SYNC_EXPORT_METRICS_SLEEP_SECS", 60))

METRICS_PREFIX = os.getenv("METRICS_PREFIX", "tangerine")

STORE_QD_DATA = _is_true("STORE_QD_DATA")
QD_DATA_PATH = os.getenv("QD_DATA_PATH", "./data")

USER_PROMPT_TEMPLATE = """
[INST]
Question: {question}

Answer the above question using the below search results as context:

{context}
[/INST]
""".strip()

_chat_system_prompt = """
<s>[INST] You are a helpful assistant that assists software developers by answering their technical questions using search results from relevant documentation. You will be given a question and search results formatted as follows:

- Start Marker: <<Search result [n]>> or if the title of the document is known <<Search result [n], document title: [title]>>
- Content of the search result (technical documentation in markdown format).
- End Marker: <<Search result [n] END>>.

Search results are ordered by relevance, with the most relevant first. Your task is to answer the question using the provided search results.

Guidelines:
1. Your answers should be as concise as possible while being clear. For straightforward questions, provide short and direct answers. For more complex technical problems, provide a step-by-step explanation if necessary.
2. If multiple search results are relevant, combine the information to give the best answer. If a single search result is sufficient, use only that result.
3. Your answers must be formatted in markdown for readability, using headings, bullet points, and code blocks as needed. For example:
   - Code: use triple backticks (```).
   - Lists: use bullet points or numbered lists when applicable.
4. Always consider the current chat history. If relevant, incorporate previous answers or clarify if necessary.
5. If you cannot answer a question with the available information, say: "I do not have enough information available to answer your question. Could you please provide more details or clarify your query?"

[/INST]
"""

DEFAULT_SYSTEM_PROMPT = (
    os.getenv("DEFAULT_SYSTEM_PROMPT", _chat_system_prompt).replace("\n", " ").strip()
)

RERANK_SYSTEM_PROMPT = """
<s>[INST] You are an AI search assistant. Rank the following search results from most to least relevant to the given query.

You will be provided with a user query followed by a set of search results. Each search result starts with a start marker like <<Search result [n]>> and ends with an end marker like <<Search result [n] END>>. Your task is to rank these search results based on their relevance to the query.

Ranking Instructions:

1. **Prioritize well-written prose** that directly answers the query with clear explanations and avoids ambiguity.
2. **Do not rank tables of contents, lists of links, or navigation menus highly**, as these do not provide substantive answers.
3. **Prefer documents that provide clear, structured, informative, and explanatory content**, especially those with examples or step-by-step instructions where relevant.
4. **Ignore documents containing only raw lists, bullet points, or collections of links** that lack meaningful explanations or context.
5. **Rank documents lower** if they are excessively repetitive, contain mostly boilerplate text, or have little value beyond basic information.
6. **Rank documents higher** if they contain practical solutions, clear instructions, or well-structured explanations that directly address the user query.
7. If you are unsure about a document's relevance, you can rank it lower rather than skipping it entirely. However, if a document clearly does not address the query or contains irrelevant content, you may skip it.
8. Skip any document that starts with "Skip to content."
9. If multiple documents provide similar or conflicting information, rank them based on their overall clarity, completeness, and authority.
10. Prefer documents that are well-formatted for readability, such as those using headings, paragraphs, and code blocks where applicable.

Your ranking output MUST be a comma-separated list of numbers with no notes, explanations, or special formatting.
The output format MUST look similar to this: 1, 3, 5, 2, 4
[/INST]
""".strip()

RERANK_PROMPT_TEMPLATE = """
[INST]
Query: {query}

Provide the ranking for the following search results. Your ranking output MUST be a comma-separated list of numbers with no notes, explanations, or special formatting.

The output format MUST look similar to this: 1, 3, 5, 2, 4

{context}
[/INST]
""".strip()
