# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Development Commands

### Environment Setup
```bash
# Install dependencies
pipenv install --dev
pipenv shell

# Database setup (with Docker)
docker compose up --build
docker exec tangerine-ollama ollama pull mistral
docker exec tangerine-ollama ollama pull nomic-embed-text

# Database setup (standalone)
flask db upgrade && flask run
```

### Development Workflow
```bash
# Run the application
flask run

# Database migrations (start database first if using Docker)
docker compose start postgres
flask db migrate -m "Your migration message"
flask db upgrade

# S3 document sync
flask s3sync

# Pre-commit hooks (formatting/linting)
pre-commit run --all
# Or install as git hook:
pre-commit install

# Run tests
pytest
```

### Linting and Formatting
- Use `ruff` for linting and formatting (configured in pyproject.toml)
- Use `flake8` for additional linting
- Line length: 100 characters
- Target Python version: 3.12

### Exception Handling Best Practices
- Keep `try` blocks as small as possible - only include the specific operation that might fail
- Use specific exception types in `except` clauses rather than broad `Exception` catches
- When multiple operations could fail differently, use separate try/except blocks for each
- Import specific exception types (e.g., `SQLAlchemyError`) at the top of the file for cleaner code
- Validate all inputs before performing operations to fail fast with clear error messages

## Project Architecture

### Core Components

**RAG System**: The application implements a Retrieval Augmented Generation (RAG) architecture with these key components:
- PostgreSQL with pgvector extension for vector storage
- LLM integration (default: mistral via ollama)
- Embedding model (default: nomic-embed-text via ollama)
- Optional S3 document synchronization

**Application Structure**:
- `src/tangerine/` - Main application package
- `src/tangerine/models/` - SQLAlchemy database models (Assistant, Conversation, Interactions)
- `src/tangerine/resources/` - Flask-RESTful API endpoints
- `src/tangerine/agents/` - Specialized agents (Jira, WebRCA)
- `migrations/` - Database migration files

### Key Modules

**Database & Models** (`db.py`, `models/`):
- Uses Flask-SQLAlchemy with Flask-Migrate
- Main models: Assistant, Conversation, Interactions
- Vector database operations handled by `vector.py`

**Document Processing** (`file.py`, `embeddings.py`):
- Supports .md, .html, .pdf, .txt, .rst, .adoc formats
- Special processing for mkdocs/antora-generated content
- Text chunking (~2000 characters) with embedding generation

**Search & Retrieval** (`search.py`, `vector.py`):
- Hybrid search combining similarity and full-text search
- Max marginal relevance (MMR) search
- Configurable search strategies via environment variables

**LLM Integration** (`llm.py`, `config.py`):
- OpenAI-compatible API support
- Multiple model configuration in `MODELS` dict
- Configurable prompts and temperature settings

**Agents** (`agents/`):
- JiraAgent: Handles Jira-related queries
- WebRCAAgent: Handles incident management queries
- Routing system determines which agent to use

### Configuration

**Environment Variables** (see `config.py`):
- Database: `DB_*` variables
- LLM: `LLM_BASE_URL`, `LLM_MODEL_NAME`, `LLM_API_KEY`
- Embeddings: `EMBED_BASE_URL`, `EMBED_MODEL_NAME`, `EMBED_API_KEY`
- Features: `ENABLE_*` flags for various capabilities
- S3 sync: `AWS_*` and `S3_SYNC_*` variables

**Multiple Model Support**:
- Configure additional models in `config.py` `MODELS` dict
- Use advanced chat API (`/api/assistants/chat`) to specify model at query time

### API Structure

**Core Endpoints**:
- `/api/assistants` - Assistant CRUD operations
- `/api/assistants/<id>/chat` - Simple chat interface
- `/api/assistants/chat` - Advanced chat API (multi-assistant, custom prompts)
- `/api/assistants/<id>/documents` - Document upload/management
- `/api/assistants/<id>/search` - Search functionality

**Advanced Chat API**: Supports multiple assistants, custom chunks, model selection, and custom prompts in a single request.

### Document Processing Pipeline

1. **Upload/Sync**: Documents uploaded via API or synced from S3
2. **Processing**: Format-specific parsing and cleanup (see README for details)
3. **Chunking**: Split into ~2000 character chunks with overlap handling
4. **Embedding**: Generate embeddings and store in vector database
5. **Search**: Similarity + optional hybrid/MMR search during queries
6. **Generation**: LLM generates response using retrieved context

### Development Tips

- **Docker Compose**: Preferred for local development (includes postgres, pgadmin, ollama)
- **Mac Development**: Run ollama natively (not in Docker) for GPU acceleration
- **Testing**: Uses pytest, test files in `tests/` directory
- **Pre-commit**: Automatically formats code with ruff and runs linting
- **Database**: pgadmin available at localhost:5050 in Docker setup
- **Metrics**: Prometheus metrics available via flask-prometheus-exporter
