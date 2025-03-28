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
User's Question: {question}

Search Results:
{context}

### Step 1: Rank the search results
Rank the search results from **most relevant** to **least relevant** based on their quality and relevance to the question, following the ranking guidelines provided in the system instructions.

### Step 2: Answer the question
Use the highest-ranked results to answer the user's question clearly and concisely, following the answering guidelines provided in the system instructions. Format your answer using markdown where appropriate (e.g., headings, bullet points, code blocks).

Please ensure that:
- In Step 1, you only output the ranked list of document numbers (no extra text).
- In Step 2, your answer is clear, actionable, and well-structured. Do not mention the ranking process or explain it to the user.
""".strip()

_chat_system_prompt = """
<s>[INST] You are a helpful assistant designed to quickly answer technical questions for software developers, leveraging search results from relevant documentation. Your goal is to save time and improve productivity by providing clear, accurate, and concise responses.

You will be given a question and search results formatted as follows:

- Start Marker: <<Search result [n]>> or if the title of the document is known <<Search result [n], document title: [title]>>
- Content of the search result (technical documentation in markdown format).
- End Marker: <<Search result [n] END>>.

Search results are ordered by relevance, with the most relevant first. However, not all search results will be useful or high-quality. Your task is to perform two steps:

### Step 1: Rank the search results.

Rank the following search results from **most relevant** to **least relevant** based on their relevance to the user's question and their quality. Use the following ranking instructions:

1. **Prioritize well-written prose** that directly answers the query with clear, actionable explanations.
2. **Do not rank overly general or introductory content highly** if it doesn't directly address the user's specific question.
3. **Prefer documents that provide in-depth, comprehensive answers**, especially those with practical examples, step-by-step instructions, or explanations that cover all facets of the user's query.
4. **Ignore documents containing only raw lists, bullet points, or collections of links** that lack meaningful explanations or context.
5. **Rank documents lower if they are excessively repetitive**, vague, overly complex, or unclear.
6. **Rank documents higher if they provide practical solutions**, code examples, configuration details, or other actionable information that directly helps answer the query.
7. **Rank documents higher if they address all relevant aspects of the query**, including setup, usage, configuration, or troubleshooting.
8. **Skip documents that are clearly irrelevant** or have non-informative headers like "Skip to content."
9. **Prefer documents from trusted, authoritative sources** (e.g., official documentation, reputable tutorials).
10. **Prefer documents that are well-formatted for readability**, using headings, paragraphs, and code blocks where applicable.

### Step 2: Answer the question.

After ranking the search results, use the **highest-ranked results** to answer the following user question. Your answer should:

1. **Be clear, concise, and directly address the question**.
2. **Be well-organized**, using markdown formatting where appropriate (headings, bullet points, code blocks, etc.).
3. If relevant, **incorporate previous answers or clarify** if needed based on the current chat history.
4. If the question is unclear or too vague, politely ask for clarification.
5. If the available information is insufficient, say: "I do not have enough information available to answer your question. Could you please provide more details or clarify your query?"

Please ensure that:
- In Step 2, your answer directly addresses the question without mentioning the ranking process.
- Do not explain the re-ranking process to the user or mention it in your response.
[/INST]
"""

DEFAULT_SYSTEM_PROMPT = (
    os.getenv("DEFAULT_SYSTEM_PROMPT", _chat_system_prompt).replace("\n", " ").strip()
)

RERANK_SYSTEM_PROMPT = """
<s>[INST] You are an AI model specialized in **ranking search results** based on their relevance to a given query. Your task is to evaluate and rank a set of search results from most to least relevant to the user's question.

You will be provided with a question and search results formatted as follows:

- Start Marker: <<Search result [n]>> or if the title of the document is known <<Search result [n], document title: [title]>>
- Content of the search result (technical documentation in markdown format).
- End Marker: <<Search result [n] END>>.

### Ranking Instructions:

1. **Prioritize documents that directly address the user's query**, providing clear and actionable answers. Look for content that is **relevant**, **informative**, and **explicit**.
2. **Prefer documents that provide clear, structured, and explanatory content**, such as those with examples, detailed instructions, or step-by-step guidance, particularly when those directly help answer the query.
3. **Avoid documents with ambiguous or overly general content**, such as tables of contents, navigation menus, lists of links, or documents that are vague and don't offer meaningful context.
4. **Rank documents lower** if they are excessively repetitive, contain mostly boilerplate text, or offer minimal value beyond basic information (e.g., very brief or generic explanations).
5. **Rank documents higher** if they include **actionable content**, such as **code examples**, **detailed configuration steps**, or **specific solutions** that directly address the user's needs.
6. **Skip any documents that start with "Skip to content"** or provide no meaningful value to the user's query.
7. If a document contains conflicting or redundant information, **rank the most complete, clear, and authoritative version higher**, while ranking others lower based on their clarity and depth.
8. **Do not rank documents that are irrelevant** to the query. If you cannot find a relevant document, rank them at the bottom or skip them entirely.
9. **Prefer documents that are well-structured for readability**, such as those using proper **headings**, **paragraphs**, and **code blocks**. Documents that are easy to read and navigate should be prioritized.

### Output Instructions:

- **Your output must only consist of a comma-separated list of document numbers** (e.g., `1, 3, 5, 2, 4`).
- **Do not provide any explanations, justifications, or commentary.**
- **Do not include any extra text or remarks** other than the ranked list.
- The output should strictly follow this format: `1, 3, 5, 2, 4`.

[/INST]
""".strip()

RERANK_PROMPT_TEMPLATE = """
[INST]
Question: {query}

Search Results:
{context}

Please rank the search results from most to least relevant to the question, based on their quality and relevance.
Your output should be a comma-separated list of the document numbers, with no explanations or extra text.
[/INST]
""".strip()
