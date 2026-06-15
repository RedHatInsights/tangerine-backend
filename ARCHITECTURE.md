# Architecture

This document describes the internal design of tangerine-backend: module responsibilities,
data flow through the RAG pipeline, database schema, and key integration points.

## High-Level Design

Tangerine is a multi-tenant RAG system built on [Flask][flask] and [Flask-RESTful][flask-restful].
Each tenant is modeled as an **Assistant** that owns one or more **KnowledgeBases**. A knowledge
base contains document chunks stored as vector embeddings in [PostgreSQL with pgvector][pgvector].
When a user asks a question, the system retrieves relevant chunks, optionally re-ranks them, and
streams an LLM-generated answer back to the caller.

The system is designed around three distinct operational modes:

1. **API server** -- serves chat, search, and management endpoints on port 8000.
2. **S3 sync job** -- a CLI command (`flask s3sync`) that pulls documents from S3, chunks and
   embeds them, and writes the results into the vector store.
3. **Agentic routing** -- an LLM-based router that can delegate queries to external agents (Jira,
   WebRCA) instead of the standard RAG pipeline.

## Package Structure

All source code lives under `src/tangerine/`. The package is organized by concern:

| Module | Responsibility |
|---|---|
| `__init__.py` | Flask application factory (`create_app`), CLI registration, startup initialization |
| `config.py` | All environment variable parsing, model registry, prompt templates |
| `db.py` | SQLAlchemy and Flask-Migrate initialization, migration table exclusions |
| `vector.py` | `VectorStoreInterface` -- document chunking, embedding storage, metadata queries |
| `search.py` | `SearchEngine` and pluggable `SearchProvider` implementations |
| `llm.py` | LLM interaction: prompt execution, streaming, re-ranking, agentic routing |
| `embeddings.py` | Embedding client with custom HTTP transport for token tracking and retry |
| `file.py` | `File` class for document representation, format-specific text extraction, `QualityDetector` |
| `utils.py` | Thin utility wrappers for embedding and removing files from knowledge bases |
| `metrics.py` | Prometheus counter/gauge factory functions, RESTful metrics exporter |
| `nltk.py` | NLTK corpus initialization (English word list for agent username extraction) |
| `resources/` | Flask-RESTful resource classes (API layer) |
| `models/` | SQLAlchemy ORM models |
| `agents/` | External agent integrations (Jira, WebRCA) |
| `sync/` | S3 synchronization pipeline |
| `sql/` | Raw SQL queries for full-text and hybrid search |
| `data/` | Bundled training data for the quality detection model |

## Data Flow

### Chat Request Pipeline

The following sequence describes what happens when a client sends a chat request to
`/api/assistants/<id>/chat` or `/api/assistants/chat`:

1. **Request parsing** -- `AssistantChatApi` (or `AssistantAdvancedChatApi`) extracts the query,
   session ID, streaming preference, and optional overrides (model, system prompt, injected chunks).
   Conversation history is auto-reconstructed from the database rather than relying on the client
   to send `prevMsgs`.

2. **Embedding** -- The query is embedded via `embed_query()`, which prepends a configurable prefix
   (default: `search_query`) and calls the OpenAI-compatible embedding endpoint.

3. **Retrieval** -- `SearchEngine.search()` fans the query out to all enabled `SearchProvider`
   instances in parallel. Each provider returns ranked `SearchResult` objects with normalized
   scores.

4. **Fusion and deduplication** -- Results from all providers are aggregated using Reciprocal Rank
   Fusion (RRF). Near-duplicate chunks (>90% TF-IDF cosine similarity) are removed.

5. **Optional LLM re-ranking** -- When `ENABLE_RERANKING` is true, the deduplicated results are
   sent to the LLM with a specialized ranking prompt. The LLM returns a comma-separated ordering
   that replaces the RRF ranking. If re-ranking fails, the system falls back to RRF.

6. **Agentic routing** (advanced API only) -- Unless `disable_agentic` is set, `llm.identify_agent`
   asks the LLM to classify the query as `JiraAgent`, `WebRCAAgent`, or `ChatAgent`. If the
   classified agent is enabled, the request is forwarded to the external agent service and the
   standard RAG path is skipped.

7. **LLM generation** -- `llm.ask()` builds a `ChatPromptTemplate` with the system prompt,
   conversation history, and search context. It streams tokens from the LLM via
   `ChatOpenAI.stream()`, recording processing and completion rate metrics.

8. **Response delivery** -- In streaming mode, tokens are sent as `data:` SSE chunks. The final
   chunk contains `search_metadata`. In non-streaming mode, the full response is returned as JSON.

9. **Post-response bookkeeping** -- The interaction is optionally stored (question, response,
   source chunks, question embedding, relevance scores). The conversation history is upserted with
   the new Q&A pair.

### Document Ingestion Pipeline

The `flask s3sync` CLI command orchestrates document ingestion:

1. **Configuration** -- An `s3.yaml` file defines knowledge bases, S3 bucket paths, file
   extensions, and citation URL templates. The config is parsed into Pydantic models
   (`SyncConfig`, `KnowledgeBaseConfig`, `AssistantConfig`).

2. **Comparison** -- For each knowledge base, the sync process compares S3 object listings against
   document chunk metadata in the vector store. It identifies files to add, update (hash changed),
   or delete (removed from S3 or prefix no longer configured).

3. **Download and embed** -- New files are downloaded concurrently to a temporary directory, then
   embedded concurrently using a configurable thread pool (`S3_SYNC_POOL_SIZE`, default 15).

4. **Atomic swap** -- New chunks are initially inserted as `active=False`. Old chunks marked for
   removal are set to `pending_removal=True`. After embedding succeeds, new chunks are activated
   and old chunks are deleted. This prevents serving partial updates.

5. **Assistant association** -- Assistants defined in the sync config are created or updated, then
   associated with their configured knowledge bases via the join table.

### Text Processing Pipeline

The `File` class handles format-specific text extraction:

| Format | Processing |
|---|---|
| HTML | Parsed with BeautifulSoup (lxml), stripped of nav/header/footer, converted to Markdown via html2text, then cleaned |
| Markdown | Formatted with mdformat, large code blocks replaced with placeholders, tables condensed to `header: value` format, relative links resolved |
| PDF | Pages extracted with PyPDF2 |
| Plain text / RST | Returned as-is |

After extraction, `VectorStoreInterface.split_to_document_chunks()` splits text into chunks:

1. If the text has Markdown headers, a `MarkdownHeaderTextSplitter` is applied first to respect
   document structure, followed by `RecursiveCharacterTextSplitter` (2000-char target, 200-char
   overlap).
2. If quality detection is enabled, chunks are filtered through a TF-IDF + Logistic Regression
   classifier trained on labeled examples to discard "junk" content (navigation fragments, boilerplate).
3. Small adjacent chunks are merged up to a 2300-character maximum to avoid overly fragmented
   embeddings.

## Search Architecture

Search is built on the Strategy pattern. `SearchEngine` holds a list of `SearchProvider`
instances, selected at startup based on feature flags:

| Provider | Flag | Mechanism |
|---|---|---|
| `SimilaritySearchProvider` | `ENABLE_SIMILARITY_SEARCH` | Cosine similarity via pgvector |
| `MMRSearchProvider` | `ENABLE_MMR_SEARCH` | Maximal Marginal Relevance (lambda=0.85, k=3) |
| `FTSPostgresSearchProvider` | `ENABLE_FULL_TEXT_SEARCH` | PostgreSQL tsvector full-text search |
| `HybridSearchProvider` | `ENABLE_HYBRID_SEARCH` | Combined vector + FTS with weighted RRF (70% FTS, 30% vector) |

All providers normalize scores to [0, 1] and assign integer ranks. When multiple providers are
enabled, their results are fused using Reciprocal Rank Fusion. The top 5 results are returned.

The hybrid search provider uses a raw SQL query (`sql/hybrid_search.sql`) that runs both a
`ts_rank_cd` full-text search and a pgvector inner-product similarity search in parallel CTEs,
then joins and weights them using RRF.

## Database Schema

Tangerine uses PostgreSQL with the pgvector extension. Tables managed by Flask-Migrate:

| Table | Purpose |
|---|---|
| `assistant` | Chatbot assistant definitions (name, description, system_prompt, model) |
| `knowledgebase` | Document collection definitions (name, description, timestamps) |
| `assistant_knowledgebase` | Many-to-many join table linking assistants to knowledge bases |
| `interactions` | Logged RAG interactions (question, LLM response, source chunks, session, user) |
| `question_embeddings` | 768-dimensional vector embeddings of user questions, linked to interactions |
| `relevance_scores` | Per-chunk retrieval method and score, linked to interactions |
| `user_feedback` | Like/dislike feedback with optional text, linked to interactions |
| `conversations` | Persisted conversation history (session-based, with auto-generated LLM titles) |

Two tables are managed by LangChain and excluded from Alembic autogeneration:

| Table | Purpose |
|---|---|
| `langchain_pg_collection` | Vector store collection metadata |
| `langchain_pg_embedding` | Document chunks with embeddings, full-text search vectors, and JSON metadata |

The `langchain_pg_embedding` table carries a `cmetadata` JSONB column that stores per-chunk
metadata including `knowledgebase_id`, `source`, `full_path`, `hash`, `citation_url`, `active`,
and `pending_removal` flags. This metadata drives the S3 sync comparison logic and search
filtering.

### Migration Approach

Migrations are managed with Flask-Migrate (Alembic). The `include_object` callback in `db.py`
excludes the two LangChain-managed tables from autogenerated migrations. A `fts_vector` tsvector
column was added to `langchain_pg_embedding` via a dedicated migration to support full-text search
without modifying LangChain internals.

## Model Registry and Multi-Model Support

The `config.py` module maintains a `MODELS` dictionary that maps model names to their connection
parameters (endpoint URL, API key, model identifier, temperature). The `default` model is always
registered. Additional models (e.g., `llama4_scout`) are conditionally added based on feature
flags.

Model selection follows a priority chain:

1. Explicit `model` parameter in the API request (advanced API only)
2. The assistant's configured `model` field
3. The `DEFAULT_MODEL` environment variable (defaults to `"default"`)

All LLM calls go through `llm.get_response()`, which creates a `ChatOpenAI` instance from the
resolved model config and streams the response through a LangChain chain (`prompt | chat`).

## Embeddings

The embedding client (`embeddings.py`) wraps `OpenAIEmbeddings` with a custom HTTP transport stack:

- `CustomTransport` extends `RetryTransport` (from httpx-retries) to add automatic retries with
  exponential backoff on 429, 502, and 503 errors. 504 is intentionally excluded.
- `CustomResponse` intercepts the response stream to extract `usage.prompt_tokens` from the JSON
  body and feed it into a Prometheus counter, working around a LangChain limitation where embedding
  token usage is not exposed through standard callbacks.
- The OpenAI client's built-in retry mechanism is disabled (`max_retries=0`) to avoid double-retry.

Query embeddings are prefixed with a configurable string (default: `search_query` for the nomic
model). Document embeddings use a separate prefix (default: `search_document`).

## Agentic Routing

The advanced chat API supports an optional agentic layer that classifies queries before running the
standard RAG pipeline:

- `llm.identify_agent()` sends the user query to the LLM with a routing prompt. The LLM returns
  one of `JiraAgent`, `WebRCAAgent`, or `ChatAgent`.
- `JiraAgent` extracts usernames from the query using NLTK word filtering, calls an external Jira
  service, and optionally summarizes multi-user results via the LLM.
- `WebRCAAgent` extracts incident IDs (pattern `ITN-YYYY-NNNNN`) via regex, authenticates with an
  SSO token, and fetches incident summaries from a Web RCA service.
- If the agent is not enabled or routing returns `ChatAgent`, the standard RAG pipeline runs.

## Configuration Architecture

All configuration is centralized in `config.py` as module-level constants parsed from environment
variables with sensible defaults for local development (Ollama on localhost). Configuration is
grouped by concern:

| Group | Key variables |
|---|---|
| Database | `DB_HOST`, `DB_PORT`, `DB_USERNAME`, `DB_PASSWORD`, `DB_NAME` |
| LLM | `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL_NAME`, `LLM_TEMPERATURE` |
| Embeddings | `EMBED_BASE_URL`, `EMBED_API_KEY`, `EMBED_MODEL_NAME`, `EMBED_QUERY_PREFIX`, `EMBED_DOCUMENT_PREFIX` |
| Search features | `ENABLE_HYBRID_SEARCH`, `ENABLE_MMR_SEARCH`, `ENABLE_SIMILARITY_SEARCH`, `ENABLE_FULL_TEXT_SEARCH`, `ENABLE_RERANKING` |
| Agents | `ENABLE_JIRA_AGENT`, `JIRA_AGENT_URL`, `ENABLE_WEB_RCA_AGENT`, `WEB_RCA_AGENT_URL` |
| S3 sync | `S3_SYNC_CONFIG_FILE`, `S3_SYNC_POOL_SIZE`, `FORCE_RESYNC`, `FORCE_RESYNC_UNTIL` |
| Quality detection | `ENABLE_QUALITY_DETECTION`, `STORE_QD_DATA`, `QD_DATA_PATH` |
| Observability | `LOG_LEVEL_GLOBAL`, `LOG_LEVEL_APP`, `DEBUG_VERBOSE`, `METRICS_PREFIX` |
| Interactions | `STORE_INTERACTIONS` |

Boolean flags use a helper `_is_true()` that accepts `1`, `t`, or `true` (case-insensitive).
Prompt templates (system prompt, user prompt, reranking prompt, agentic router prompt) are defined
as module-level string constants and can be overridden via environment variables or per-assistant
configuration.

## API Route Organization

Routes are registered in `resources/routes.py` using Flask-RESTful's `api.add_resource()`:

| Path | Resource | Methods | Purpose |
|---|---|---|---|
| `/api/assistants` | `AssistantsApi` | GET, POST | List and create assistants |
| `/api/assistants/<id>` | `AssistantApi` | GET, PUT, DELETE | CRUD for a single assistant |
| `/api/assistants/<id>/chat` | `AssistantChatApi` | POST | Single-assistant RAG chat |
| `/api/assistants/chat` | `AssistantAdvancedChatApi` | POST | Multi-assistant chat with model/prompt overrides and agentic routing |
| `/api/assistants/<id>/search` | `AssistantSearchApi` | POST | Search without LLM generation |
| `/api/assistants/<id>/knowledgebases` | `AssistantKnowledgeBasesApi` | GET, POST, DELETE | Manage assistant-knowledgebase associations |
| `/api/knowledgebases` | `KnowledgeBasesApi` | GET, POST | List and create knowledge bases |
| `/api/knowledgebases/<id>` | `KnowledgeBaseApi` | GET, PUT, DELETE | CRUD for a single knowledge base |
| `/api/knowledgebases/<id>/documents` | `KnowledgeBaseDocuments` | POST, DELETE | Upload and remove documents |
| `/api/conversations/list` | `ConversationListApi` | POST | List conversations for a user |
| `/api/conversations/load` | `ConversationRetrievalApi` | POST | Load a conversation by session ID |
| `/api/conversations/upsert` | `ConversationUpsertApi` | POST | Create or update a conversation |
| `/api/conversations/delete` | `ConversationDeleteApi` | POST | Delete a conversation |
| `/api/feedback` | `FeedbackApi` | POST | Submit like/dislike feedback for an interaction |
| `/ping` | `PingApi` | GET | Health check |

`AssistantAdvancedChatApi` extends `AssistantChatApi`, adding support for multiple assistants,
external chunk injection, model selection, custom user/system prompts, and agentic workflow
control.

## Observability

Prometheus metrics are exposed through [prometheus-flask-exporter][prometheus-exporter] with a
configurable prefix (default: `tangerine`). Key custom metrics:

- `assistant_response_counter` -- total responses per assistant (labels: assistant_id, assistant_name)
- `user_interaction_counter` -- total user interactions (labels: user, client, assistant_id, assistant_name)
- `llm_completion_tokens` / `llm_prompt_tokens` -- LLM token usage counters
- `llm_completion_rate` / `llm_processing_rate` -- tokens/sec gauges for the most recent request
- `llm_no_answer` -- counter for queries with zero search results
- `embed_prompt_tokens` -- embedding model token usage

User IDs are anonymized (SHA-256, truncated to 16 characters) before being stored in metrics
labels.

<!-- reference-style link definitions -->
[flask]: https://flask.palletsprojects.com/
[flask-restful]: https://flask-restful.readthedocs.io/
[pgvector]: https://github.com/pgvector/pgvector
[prometheus-exporter]: https://github.com/rycus86/prometheus_flask_exporter
