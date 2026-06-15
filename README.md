# tangerine-backend

A slim, lightweight RAG (Retrieval Augmented Generation) system for creating and managing chatbot
assistants. Each assistant answers questions from a curated set of documents known as a knowledge
base.

![Demo video][demo-gif]

Maintained by [Red Hat Insights][rhi]. Licensed under [Apache 2.0][license].

## Table of Contents

- [Overview](#overview)
  - [How It Works](#how-it-works)
  - [Document Processing](#document-processing)
  - [Related Projects](#related-projects)
  - [Use of Hosted AI Services](#use-of-hosted-ai-services)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Local Environment Setup](#local-environment-setup)
  - [With Docker Compose](#with-docker-compose)
  - [Without Docker Compose](#without-docker-compose)
- [Development](#development)
  - [Installing Development Packages](#installing-development-packages)
  - [Linting and Formatting](#linting-and-formatting)
  - [Using Pre-commit](#using-pre-commit)
  - [Running Tests](#running-tests)
  - [Database Migrations](#database-migrations)
  - [Debugging in VSCode](#debugging-in-vscode)
  - [Mac Development Tips](#mac-development-tips)
- [Synchronizing Documents from S3](#synchronizing-documents-from-s3)
  - [Continuous Synchronization](#continuous-synchronization)
- [Deploying to OpenShift](#deploying-to-openshift)
- [Running the Frontend](#running-the-frontend)
- [API Reference](#api-reference)
  - [Available Endpoints](#available-endpoints)
  - [Advanced Chat API](#advanced-chat-api)
  - [Prompt Template Variables](#prompt-template-variables)
  - [Conversation History](#conversation-history)
- [Multiple Model Support](#multiple-model-support)
- [License](#license)

## Overview

tangerine implements a basic RAG architecture that relies on four key components:

- **Vector database** -- PostgreSQL with the [pgvector][pgvector] extension
- **Large language model (LLM)** -- any [OpenAI-compatible API][openai-api] service; locally, you
  can use [Ollama][ollama]
- **Embedding model** -- any OpenAI-compatible API service; locally, you can use Ollama
- **Document source** -- files uploaded via the API, or (optionally) synced from an S3 bucket

This project is maintained by Red Hat's Hybrid Cloud Management Engineering Productivity Team. It
was born out of a hackathon and is still a work in progress. Some areas of the code are well
developed while others need attention to become fully production-ready. That said, tangerine is
currently in good enough shape to provide a working chatbot system.

### How It Works

**Data preparation:**

1. Documents are uploaded to the backend service (or synced from an S3 bucket).
2. Documents are processed, converted, and cleaned up (see
   [Document Processing](#document-processing) below).
3. Documents are split into text chunks of approximately 2000 characters.
4. Embeddings are created for each chunk and inserted into the vector database.

**Retrieval Augmented Generation:**

1. A user asks a question through the chat interface.
2. Embeddings are created for the query using the embedding model.
3. A similarity search and a max marginal relevance search are performed against the vector
   database to find the most relevant document chunks (scoped to the assistant's knowledge bases).
4. The LLM is prompted to answer the question using only the retrieved context.
5. The response is streamed back to the user along with citation metadata.

### Document Processing

Document processing is arguably the most important part of a good RAG solution. The quality of data
stored in each text chunk is key to yielding accurate search results for the LLM.

The documentation set has initially focused on pages compiled with [MkDocs][mkdocs] or
[Antora][antora]. Supported formats:

| Format | Status |
| ------ | ------ |
| `.md`  | Well supported |
| `.html` (MkDocs / Antora) | Well supported |
| `.pdf`, `.txt`, `.rst` | Supported but not fully optimized |
| `.adoc` | Work in progress (via [Docling][docling]) |

**Processing logic for Markdown:**

1. Replace very large code blocks with a placeholder directing the reader to the original
   documentation (large code blocks tend to fill chunks with unhelpful content).
2. Convert tables into plain text with `header: value` rows to preserve context across chunks.
3. Rewrite relative links as absolute URLs so citation snippets remain navigable.
4. Remove extra whitespace, non-printable characters, and other formatting noise.

**Processing logic for HTML:**

1. Detect whether the page was created with MkDocs or Antora. If so, extract only the content body
   (strip header, footer, navigation, etc.).
2. Convert the page to Markdown with `html2text`, then process it as described above.

**Chunking strategy:**

Documents are split into chunks of about 2000 characters with no overlap. Very small chunks are
"rolled" into the next chunk so that each chunk carries as much useful content as possible.

### Related Projects

- [tangerine-frontend][tangerine-frontend] -- the accompanying web UI
- [backstage-plugin-ai-search-frontend][backstage-plugin] -- a related plugin for
  [Red Hat Developer Hub][rhdh]

### Use of Hosted AI Services

tangerine can be configured to use any OpenAI-compatible API service hosting an LLM or embedding
model. The model names and instruction prompts are fully customizable via environment variables:

```sh
LLM_BASE_URL=https://3rd-party-ai-service:443/v1
LLM_MODEL_NAME=mistral-7b-instruct
LLM_API_KEY=your-secret-key
DEFAULT_SYSTEM_PROMPT="Your LLM system prompt here"
EMBED_BASE_URL=https://3rd-party-ai-service:443/v1
EMBED_MODEL_NAME=snowflake-arctic-embed-m-v1.5
EMBED_API_KEY=your-secret-key
EMBED_QUERY_PREFIX="Represent this sentence for searching relevant passages"
EMBED_DOCUMENT_PREFIX=""
```

## Architecture

The backend manages assistants, knowledge bases, document ingestion, vector search, and LLM
interaction through a Flask REST API backed by PostgreSQL with pgvector.

See [Architecture][architecture] for internal design details.

## Prerequisites

- Python 3.12
- [pipenv][pipenv]
- [Docker][docker] (or Podman for standalone database setup)
- [Ollama][ollama] (for local LLM and embedding model hosting)
- (on Mac) [Homebrew][homebrew]

## Local Environment Setup

A development environment can be set up with or without Docker Compose. In both cases, Ollama can
make use of your NVIDIA or AMD GPU (see the [Ollama GPU guide][ollama-gpu] for details). On macOS,
Ollama must run as a standalone application outside Docker since Docker Desktop does not support GPU
passthrough.

### With Docker Compose

The Compose file spins up all components. Ollama hosts the LLM and embedding model, PostgreSQL
persists data, and pgAdmin lets you query the database.

1. Install Docker by following the [official guide][docker-install].

   > **Note:** The Compose file does not currently work with Podman.

2. On Linux, complete the [post-install steps][docker-postinstall].

3. Create a data directory for local persistence:

   ```sh
   mkdir data
   ```

4. Start the stack (PostgreSQL data persists in `data/postgres`):

   ```sh
   docker compose up --build
   ```

5. Pull the default models (data persists in `data/ollama`):

   ```sh
   docker exec tangerine-ollama ollama pull mistral
   docker exec tangerine-ollama ollama pull nomic-embed-text
   ```

6. Verify the API is running on port 8000:

   ```sh
   curl -s http://127.0.0.1:8000/api/assistants
   ```

7. (Optional) Start the [frontend with Docker Compose][frontend-docker].

pgAdmin is available at `http://localhost:5050`.

#### Using HuggingFace text-embeddings-inference (deprecated)

Ollama previously lacked an OpenAI-compatible `/v1/embeddings` endpoint. If you need to test
embedding models not supported by Ollama, you can use the HuggingFace
[text-embeddings-inference][tei] server instead:

1. Install [Git LFS][git-lfs]:

   - Fedora: `sudo dnf install git-lfs`
   - macOS: `brew install git-lfs`

   Then activate it:

   ```sh
   git lfs install
   ```

2. Download the embedding model:

   ```sh
   mkdir data/embeddings
   git clone https://huggingface.co/nomic-ai/nomic-embed-text-v1.5 \
     data/embeddings/nomic-embed-text
   ```

3. Search for `uncomment to use huggingface text-embeddings-inference` in
   [docker-compose.yml][compose-file] and uncomment the relevant lines.

### Without Docker Compose

1. Ensure the following are installed and working:

   - `pipenv`
   - `pyenv`
   - `docker` or `podman`
   - (on Mac) `brew`

2. Install Ollama from the [download page][ollama-download] (or via Homebrew on Mac):

   ```sh
   brew install ollama
   ```

3. Start Ollama:

   ```sh
   ollama serve
   ```

4. Pull the language and embedding models:

   ```sh
   ollama pull mistral
   ollama pull nomic-embed-text
   ```

5. (on Mac) Install the PostgreSQL C library:

   ```sh
   brew install libpq
   ```

   For Apple Silicon, export these environment variables to avoid C library errors:

   ```sh
   export PATH="/opt/homebrew/opt/libpq/bin:$PATH"
   export LDFLAGS="-L/opt/homebrew/opt/libpq/lib"
   export CPPFLAGS="-I/opt/homebrew/opt/libpq/include"
   ```

6. Create the data directory:

   ```sh
   mkdir data
   ```

7. Start the vector database:

   ```sh
   docker run -d \
       -e POSTGRES_PASSWORD="citrus" \
       -e POSTGRES_USER="citrus" \
       -e POSTGRES_DB="citrus" \
       -e POSTGRES_HOST_AUTH_METHOD=trust \
       -v data/postgres:/var/lib/postgresql/data \
       -p 5432:5432 \
       pgvector/pgvector:pg16
   ```

8. Set up the Python virtual environment:

   ```sh
   pipenv install --dev
   pipenv shell
   ```

9. Start the backend:

   ```sh
   flask db upgrade && flask run
   ```

10. Verify the API is running on port 8000:

    ```sh
    curl -s http://127.0.0.1:8000/api/assistants
    ```

11. (Optional) Start the [frontend without Docker Compose][frontend-standalone].

## Development

### Installing Development Packages

Install development dependencies before contributing:

```sh
pipenv install --dev
```

### Linting and Formatting

This project uses [ruff][ruff] for linting, formatting, and import sorting. Configuration lives in
`pyproject.toml`:

- Line length: 100 characters
- Target: Python 3.12

Ruff is the authoritative linter and formatter for this project and is enforced through pre-commit
hooks and CI.

### Using Pre-commit

This project uses [pre-commit][pre-commit] to run formatting and linting checks automatically.

Run all checks manually:

```sh
pre-commit run --all
```

Or install as a Git hook so checks run on every commit:

```sh
pre-commit install
```

### Running Tests

Tests live in the `tests/` directory and are run with [pytest][pytest]:

```sh
pipenv run pytest -v -s
```

This is the same command used in CI.

### Database Migrations

tangerine uses [Flask-Migrate][flask-migrate] to manage database schema changes. After modifying any
SQLAlchemy model, generate a new migration:

```sh
flask db migrate -m "Your migration message"
```

To apply migrations locally:

1. Start the database (if using Docker Compose, start only PostgreSQL):

   ```sh
   docker compose start postgres
   ```

2. Run migrations:

   ```sh
   flask db upgrade
   ```

3. After migrations are applied, start the full stack with `docker compose up --build` as usual.

When deploying to OpenShift, the backend deploy template runs `flask db upgrade` in an init
container before the application pod starts.

### Debugging in VSCode

Run PostgreSQL and Ollama (locally or in containers) but do not start the backend container. In
VSCode, open **Run & Debug** and launch the "Debug Tangerine Backend" target. You can set
breakpoints and inspect runtime state. A second debug target is available for unit tests.

### Mac Development Tips

Ollama running inside Docker on Apple Silicon cannot use hardware acceleration, making LLM responses
very slow. Running Ollama natively on macOS does use acceleration and is significantly faster.

Recommended Mac development setup:

- Run tangerine-backend in the VSCode debugger
- Run Ollama directly on the host
- Run PostgreSQL and pgAdmin in Docker

Comment out or stop the `ollama` container in the Compose file and run `ollama serve` in your shell
instead.

## Synchronizing Documents from S3

You can configure assistants to sync their knowledge base from an S3 bucket.

1. Export your S3 credentials:

   ```sh
   export AWS_ACCESS_KEY_ID="MYKEYID"
   export AWS_DEFAULT_REGION="us-east-1"
   export AWS_ENDPOINT_URL_S3="https://s3.us-east-1.amazonaws.com"
   export AWS_SECRET_ACCESS_KEY="MYACCESSKEY"
   export BUCKET="mybucket"
   ```

   If using Docker Compose, store these in a `.env` file:

   ```sh
   echo 'AWS_ACCESS_KEY_ID=MYKEYID' >> .env
   echo 'AWS_DEFAULT_REGION=us-east-1' >> .env
   echo 'AWS_ENDPOINT_URL_S3=https://s3.us-east-1.amazonaws.com' >> .env
   echo 'AWS_SECRET_ACCESS_KEY=MYACCESSKEY' >> .env
   echo 'BUCKET=mybucket' >> .env
   ```

2. Create an `s3.yaml` file describing your assistants and their documents. See
   [s3-example.yaml][s3-example] for reference.

   If using Docker Compose, copy the config into the container:

   ```sh
   docker cp s3.yaml tangerine-backend:/opt/app-root/src/s3.yaml
   ```

3. Run the sync job:

   With Docker Compose:

   ```sh
   docker exec -ti tangerine-backend flask s3sync
   ```

   Without Docker Compose:

   ```sh
   flask s3sync
   ```

The sync creates assistants and ingests the configured documents. On subsequent runs it checks the
S3 bucket for updates and only re-ingests files whose content has changed.

### Continuous Synchronization

The S3 sync can run on a schedule for continuous updates. The sync logic works as follows:

1. When documents are ingested, the S3 ETag is stored as metadata on the document chunks.
2. On each sync run, bucket objects are checked for ETag changes.
3. If an object has a new ETag, its chunks are replaced using an active/standby strategy:
   1. Existing chunks are marked `active: true` (used for RAG lookups).
   2. New chunks are inserted with `active: false`.
   3. Once all new chunks are ready, metadata is flipped so new chunks become `active: true` and
      old chunks become `active: false`.
   4. Chunks with `active: false` are removed from the vector database.

The [OpenShift templates][openshift-templates] include a CronJob configuration for running the sync
on a schedule.

## Deploying to OpenShift

This repository provides [OpenShift templates][openshift-templates] for all infrastructure
components (except the LLM hosting server). You may not need the PostgreSQL or
text-embeddings-inference templates if you provide those services through other infrastructure.

## Running the Frontend

The API can be used directly to create, manage, and chat with assistants. For a graphical interface,
see [tangerine-frontend][tangerine-frontend].

## API Reference

### Available Endpoints

| Path | Method | Description |
| ---- | ------ | ----------- |
| `/api/assistants` | `GET` | List all assistants |
| `/api/assistants` | `POST` | Create a new assistant |
| `/api/assistants/<id>` | `GET` | Get an assistant |
| `/api/assistants/<id>` | `PUT` | Update an assistant |
| `/api/assistants/<id>` | `DELETE` | Delete an assistant |
| `/api/assistants/<id>/chat` | `POST` | Chat with an assistant |
| `/api/assistants/chat` | `POST` | Advanced chat API |
| `/api/assistants/<id>/knowledgebases` | `GET` | Get knowledge bases for an assistant |
| `/api/assistants/<id>/knowledgebases` | `POST` | Associate knowledge bases with an assistant |
| `/api/assistants/<id>/knowledgebases` | `DELETE` | Disassociate knowledge bases from an assistant |
| `/api/assistants/<id>/search` | `POST` | Search an assistant's knowledge base |
| `/api/knowledgebases` | `GET` | List all knowledge bases |
| `/api/knowledgebases` | `POST` | Create a new knowledge base |
| `/api/knowledgebases/<id>` | `GET` | Get a knowledge base |
| `/api/knowledgebases/<id>` | `PUT` | Update a knowledge base |
| `/api/knowledgebases/<id>` | `DELETE` | Delete a knowledge base |
| `/api/knowledgebases/<id>/documents` | `POST` | Upload documents to a knowledge base |
| `/api/knowledgebases/<id>/documents` | `DELETE` | Delete documents from a knowledge base |
| `/api/assistantDefaults` | `GET` | Get assistant default settings |
| `/ping` | `GET` | Health check |

### Advanced Chat API

The default assistant chat API (`/api/assistants/<id>/chat`) is optimized for simple chat
experiences such as GUI-based chatbots. The advanced chat API (`/api/assistants/chat`) supports more
complex workflows.

**Basic usage** (equivalent to the standard chat API):

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs"],
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false
  }'
```

**Multi-assistant queries** -- combine results from multiple assistants:

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs", "knowledge base"],
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false
  }'
```

**Bring your own chunks** -- skip the search and answer from provided content:

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "chunks": [
      "To deploy your app run `app deploy`",
      "Apps are deployed with the app command",
      "The app command is available from our private repo."
    ]
  }'
```

**Non-persisted chunks** -- use sensitive content without storing it in the database:

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "chunks": ["CONFIDENTIAL: Internal deployment process requires..."],
    "no_persist_chunks": true
  }'
```

When `no_persist_chunks` is `true`, chunks are used for LLM processing but are not stored in
interaction logs. All other interaction data (query, response, metadata) continues to be logged
normally.

**Model selection** -- specify a model at query time (requires [multiple model support](#multiple-model-support)):

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs"],
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "model": "granite"
  }'
```

**Custom system prompt** -- override the default system prompt (both `prompt` and `system_prompt`
parameters are supported for backward compatibility):

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs"],
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "prompt": "<s>[INST] You are a helpful assistant. Keep responses under 3 sentences. [/INST]"
  }'
```

**Custom user prompt template** -- override the default user prompt while keeping the system prompt
unchanged. The template supports `{question}` and `{context}` variables:

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs"],
    "query": "How do I deploy my app?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "userPrompt": "[INST]\nQuestion: {question}\n\nContext: {context}\n\nProvide a brief answer.\n[/INST]"
  }'
```

You can combine `userPrompt` with `prompt`/`system_prompt` to fully customize both parts of the
conversation template.

**Disable agentic workflow** -- bypass intent analysis and route all queries directly to the
standard chat agent:

```sh
curl -X POST http://localhost:8000/api/assistants/chat \
  -H "Content-Type: application/json" \
  -d '{
    "assistants": ["support docs"],
    "query": "What is the recent Jira activity?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000",
    "interactionId": "137e6543-2f1b-12d3-b456-526614174999",
    "client": "curl",
    "stream": false,
    "disable_agentic": true
  }'
```

### Prompt Template Variables

When customizing prompts, the following variables are available for interpolation using Python's
`{variable_name}` syntax:

| Variable | Description |
| -------- | ----------- |
| `{question}` | The user's original query |
| `{context}` | Formatted search results from the knowledge base |

The `{context}` variable contains search results formatted with markers:

```
<<Search result 1, document title: 'Example Document'>>
Content of the first search result...
<<Search result 1 END>>
```

**Guidelines:**

- **System prompts** define the AI's role and behavior. They should not use template variables.
- **User prompts** should include both `{question}` and `{context}` and provide instructions on how
  to process them.
- Use the `<s>[INST]...[/INST]` format for compatibility with chat models.

### Conversation History

tangerine automatically maintains conversation history for multi-turn conversations. History is
reconstructed from the database -- clients do not need to track conversation state.

Both chat APIs automatically:

- Look up existing conversation history using the `sessionId`
- Provide the last 10 question-answer pairs as context to the LLM
- Store the complete conversation history in the database

Simply provide a `sessionId` and tangerine handles the rest:

```sh
curl -X POST http://localhost:8000/api/assistants/123/chat \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What was my previous question about?",
    "sessionId": "433e4567-8e9b-22d3-a456-626614174000"
  }'
```

The `prevMsgs` parameter is ignored if provided. Conversation history is always auto-reconstructed
from the database for consistency across all clients.

**Session ownership:** When a `user` is provided, the system verifies session ownership to prevent
unauthorized access to conversation history. Anonymous sessions (no `user` provided) can access any
session by `sessionId`.

> **Security note:** The current implementation uses the `user` field from the request body for
> session ownership verification. In production, this should be replaced with authenticated user
> identity (JWT, OAuth/OIDC, API key, etc.) to prevent session hijacking.

## Multiple Model Support

tangerine can be extended to support multiple LLM models for use with the advanced chat API. In the
`MODELS` dictionary in `src/tangerine/config.py`, add entries for each model:

```python
MODELS = {
    "default": {
        "base_url": config.LLM_BASE_URL,
        "name": config.LLM_MODEL_NAME,
        "api_key": config.LLM_API_KEY,
        "temperature": config.LLM_TEMPERATURE,
    },
    "granite": {
        "base_url": config.GRANITE_BASE_URL,
        "name": config.GRANITE_MODEL_NAME,
        "api_key": config.GRANITE_API_KEY,
        "temperature": config.GRANITE_TEMPERATURE,
    },
}
```

Add the corresponding environment variables to `config.py` and your deployment environment. Then
specify the model at query time using the `model` parameter in the
[advanced chat API](#advanced-chat-api).

## License

This project is licensed under the [Apache License 2.0][license].

<!-- Reference-style links -->

[demo-gif]: docs/demo.gif
[rhi]: https://github.com/RedHatInsights
[license]: LICENSE
[pgvector]: https://github.com/pgvector/pgvector
[openai-api]: https://platform.openai.com/docs/api-reference
[ollama]: https://ollama.com
[mkdocs]: https://www.mkdocs.org
[antora]: https://antora.org
[docling]: https://ds4sd.github.io/docling/
[tangerine-frontend]: https://github.com/RedHatInsights/tangerine-frontend
[backstage-plugin]: https://github.com/RedHatInsights/backstage-plugin-ai-search-frontend
[rhdh]: https://developers.redhat.com/rhdh/overview
[architecture]: ARCHITECTURE.md
[pipenv]: https://pipenv.pypa.io
[docker]: https://docs.docker.com/engine/install/
[homebrew]: https://brew.sh
[ollama-gpu]: https://github.com/ollama/ollama/blob/main/docs/gpu.md
[docker-install]: https://docs.docker.com/engine/install/
[docker-postinstall]: https://docs.docker.com/engine/install/linux-postinstall/
[frontend-docker]: https://github.com/RedHatInsights/tangerine-frontend#with-docker-compose
[frontend-standalone]: https://github.com/RedHatInsights/tangerine-frontend#without-docker-compose
[tei]: https://github.com/huggingface/text-embeddings-inference
[git-lfs]: https://git-lfs.com/
[compose-file]: docker-compose.yml
[s3-example]: s3-example.yaml
[openshift-templates]: openshift/
[ollama-download]: https://ollama.com/download
[ruff]: https://docs.astral.sh/ruff/
[pre-commit]: https://pre-commit.com
[pytest]: https://docs.pytest.org
[flask-migrate]: https://flask-migrate.readthedocs.io
