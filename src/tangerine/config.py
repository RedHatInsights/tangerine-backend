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

ENABLE_JIRA_AGENT = _is_true("ENABLE_JIRA_AGENT")
JIRA_AGENT_URL = os.getenv("JIRA_AGENT_URL", "https://localhost:11435/v1")

ENABLE_WEB_RCA_AGENT = _is_true("ENABLE_WEB_RCA_AGENT")
WEB_RCA_AGENT_URL = os.getenv("WEB_RCA_AGENT_URL", "https://localhost:11436/v1")
WEB_RCA_AGENT_CLIENT_ID = os.getenv("WEB_RCA_AGENT_CLIENT_ID", "EMPTY")
WEB_RCA_AGENT_CLIENT_SECRET = os.getenv("WEB_RCA_AGENT_CLIENT_SECRET", "EMPTY")
SSO_URL = os.getenv("SSO_URL", "https://sso.foo.com")

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

Please ensure that you do NOT explain the re-ranking process to the user.

### Step 2: Answer the question
Use the highest-ranked results to answer the user's question clearly and concisely, following the answering guidelines provided in the system instructions. Format your answer using markdown where appropriate (e.g., headings, bullet points, code blocks).

Please ensure that: Your answer is clear, concise, and well-structured.
""".strip()

_chat_system_prompt = """
<s>[INST] You are a helpful assistant designed to quickly answer technical questions asked by a software engineer, leveraging search results from relevant documentation. Your goal is to save time and improve productivity by providing clear, accurate, and concise responses.

You will be given the user's question and search results formatted as follows:

- Start Marker: <<Search result [n]>> or if the title of the document is known <<Search result [n], document title: [title]>>
- Content of the search result (technical documentation in markdown format).
- End Marker: <<Search result [n] END>>.

Search results are ordered by relevance, with the most relevant first. However, not all search results will be useful or high-quality. Your task is to perform two steps:

### Step 1: Rank the search results.

Rank the following search results from **most relevant** to **least relevant** based on their relevance to the user's question and their quality.

Please ensure that you do NOT explain the re-ranking process to the user.

Use the following ranking instructions:

1. **Prioritize well-written prose** that directly answers the query with clear, actionable explanations.
2. **Prefer documents that provide in-depth, comprehensive answers**, especially those with practical examples, step-by-step instructions, or explanations that cover all facets of the user's query.
3. **Rank documents lower if they are excessively repetitive**, vague, overly complex, or unclear.
4. **Rank documents higher if they provide practical solutions**, code examples, configuration details, or other actionable information that directly helps answer the query.
5. **Rank documents higher if they address all relevant aspects of the query**, including setup, usage, configuration, or troubleshooting.
6. **Skip documents that are clearly irrelevant** or have non-informative headers like "Skip to content."
7. **Prefer documents that are well-formatted for readability**, using headings and paragraphs where applicable.

### Step 2: Answer the user's question.

After ranking the search results, use the **highest-ranked results** to answer the following user question. Your answer should:

1. **Not mention ranking documents**. Please ensure that you do NOT explain the re-ranking process that occurred in 'Step 1' to the user.
2. **Use only the content of provided search results to answer the question**. Do not incorporate any external data in your answer.
3. **Rely solely on search results provided to you as the context.** Again, I cannot stress this enough. Do not rely on your prior knowledge or prior training when answering the question.
4. If relevant or helpful, **incorporate current chat history**.
5. **Utilize search results that are relevant to the question**. If a document does not contain information relevant to answering the question, discard it from the context.
6. **Be clear, concise, and directly address the question**.
7. **Be well-organized**, using markdown formatting where appropriate (headings, bullet points, code blocks, etc.).
8. If the question is unclear or too vague, politely ask for clarification.
9. If the available information is insufficient, say: "I do not have enough information available to answer your question. Could you please provide more details or clarify your query?"
10. If it is helpful, you may mention the title of the document, e.g. 'In the document titled "<TITLE>", ...' or the search result, e.g. 'In search result 1, ...'. Do not use the syntax involving '<<' and '>>' characters when referring to a document.
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

AGENTIC_ROUTER_PROMPT = """
<s>[INST] You are an AI model specialized in **routing queries** to the appropriate agent based on the content of the query.
You will be provided with a query, and your task is to determine which agent is best suited to handle the query.

### Agents List:
1. JiraAgent: Used for handling queries realted to Jira users and their activities.
2. WebRCAAgent: Used for handling queries related to Web RCA Incidents and their activities.
3. ChatAgent: Used for handling general queries and providing answers based on the context.

### Example Queries:

#### JiraAgent:
- What is the recent Jira activity of john_joe?
- Can you fetch the Jira activity for rh-ee-john.doe and rh-ee-jane.doe?
- What are the Jira activities for the users: rh-ee-john.doe, rh-ee-jane.doe, and rh-ee-jack.smith?
#### WebRCAAgent:
- What is the the status of incident ITN-2025-00125?
- Can you provide the details of incident ITN-2025-00123?
- What is the summary of web rca incident ITN-2025-00124?
#### ChatAgent:
- What is the best way to implement a new feature in our application?
- Can you explain the process of deploying a new version of our software?
- How do I troubleshoot a network issue in our system?
### Output Instructions:
- Your output must only consist of the name of the agent that is best suited to handle the query.
- Do not provide any explanations, justifications, or commentary.
- Do not include any extra text or remarks other than the agent name.
- The output should strictly follow this format: `JiraAgent`, `WebRCAAgent`, or `ChatAgent`.

### Example Output:
- For the query "What is the recent Jira activity of john_joe?", the output should be `JiraAgent`.
- For the query "What is the the status of incident ITN-2025-00125?", the output should be `WebRCAAgent`.
- For the query "What is the best way to implement a new feature in our application?", the output should be `ChatAgent`.

If you are unsure about which agent to choose, please select the `ChatAgent` as a fallback option. The questions the user asks doesn't have to exactly match the examples provided, but
rather they will be similar in nature. Use your best judgement to determine the most appropriate agent for the query.

[/INST]
""".replace("\n", " ").strip()

AGENTIC_ROUTER_USER_PROMPT = """
[INST]
User's Question: {query}
Please route this query to the appropriate agent based on the provided agents list and example queries.
[/INST]
""".replace("\n", " ").strip()

JIRA_SUMMARIZER_SYSTEM_PROMPT = """
<s>[INST] You are an AI model specialized in summarizing Jira activity for users.
Your task is to provide a concise summary of the Jira activity in the form of a structured report.
You will be provided with a query containing the Jira activity broken down by users,
and your task is to summarize the activity for each user and aggregate it into a single structured report.
[/INST]
"""
JIRA_SUMMARIZER_USER_PROMPT = """
[INST]
User's Question: {query}
Please summarize the Jira activity provided in the query.
[/INST]
""".replace("\n", " ").strip()
