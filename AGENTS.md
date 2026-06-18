# Tangerine

## Project Overview

Tangerine is a slim, lightweight RAG (Retrieval Augmented Generation) system for creating and
managing chatbot assistants. Each assistant answers questions from a curated set of documents known
as a knowledge base. The system is built on Flask and Flask-RESTful, uses PostgreSQL with pgvector
for vector storage, and integrates with OpenAI-compatible LLM providers. It is maintained by
[Red Hat Insights][rhi] and licensed under [Apache 2.0][license]. The project is deployed as a
containerized Flask service on OpenShift.

## Dependencies

**Runtime:** Python 3.12, Flask, Flask-RESTful, SQLAlchemy, Flask-Migrate, PostgreSQL with pgvector,
LangChain, pipenv for dependency management.

**Dev/Test:** pytest, ruff (lint + format), pre-commit, pipenv dev packages.

## Development Commands

See [Development][readme-dev] in the README for the full local environment setup, Docker Compose
configuration, and database migration workflow.

Commands relevant to agent-driven workflows:

```bash
# Install all dependencies (including dev)
pipenv install --dev

# Run tests (matches CI)
pipenv run pytest -v -s

# Lint and format (matches CI)
pre-commit run --all

# Run the application
pipenv run flask run

# Database migrations
pipenv run flask db migrate -m "migration message"
pipenv run flask db upgrade
```

**Python environment:** This project uses pipenv for dependency management. Always prefix Python
commands with `pipenv run` (e.g., `pipenv run pytest`, `pipenv run flask`). Alternatively, activate
the virtualenv first with `pipenv shell`.

## Architecture

All source code lives under `src/tangerine/`. The application factory is in `__init__.py`, and
configuration is centralized in `config.py` (environment variables, model registry, prompt
templates). API endpoints are defined as Flask-RESTful resources in `resources/`, database models
live in `models/`, and the RAG pipeline spans `embeddings.py`, `vector.py`, `search.py`, and
`llm.py`. An agent routing layer in `agents/` delegates specialized queries to external systems.

For module-level detail, data flow diagrams, database schema, and configuration reference, see
[ARCHITECTURE.md][architecture].

## Code Style

- **Linter/Formatter:** [ruff][ruff], configured in `pyproject.toml`. Runs in CI via pre-commit
  (lint, import sorting, and format as separate hooks). A legacy `[tool.flake8]` section exists in
  `pyproject.toml` but is **not used** in CI or pre-commit — ignore it.
- **Line length:** 100 characters.
- **Indent width:** 4 spaces.
- **Target Python version:** 3.12.
- **Import sorting:** Handled by ruff with the `I` rule set.
- **Excluded paths:** `data/*` (ruff), `migrations/` (pre-commit).

## Testing

- **Framework:** pytest.
- **Command:** `pipenv run pytest -v -s` (matches the CI `unit-tests` job).
- **Test location:** `tests/` directory. Test files follow the `test_*.py` naming convention.
- **pytest config:** `pyproject.toml` sets `addopts = ["--ignore=data/"]`.
- **CI:** [GitHub Actions][gh-actions-workflow] runs two jobs on every push and PR to `main`:
  `pre-commit` (lint/format) and `unit-tests` (pytest).

## Deployment

Tangerine is deployed as a container on OpenShift. The Dockerfile uses a multi-stage build
(UBI9 build stage, UBI10 runtime stage) and exposes port 8000. Tekton pipelines handle OpenShift
builds. See [Deploying to OpenShift][readme-deploy] in the README.

## Common Mistakes

1. **Running commands outside pipenv.** Running `pytest`, `flask`, or `python` directly instead of
   `pipenv run pytest` will use the system Python and miss project dependencies. Always use
   `pipenv run` or activate the shell with `pipenv shell` first.

2. **Using flake8 instead of ruff.** A `[tool.flake8]` section exists in `pyproject.toml`, but
   flake8 is not in pre-commit or CI. Ruff is the sole linter and formatter. Configuring or running
   flake8 is wasted effort.

3. **Broad exception handling.** Keep `try` blocks minimal — wrap only the specific operation that
   might fail. Use specific exception types (e.g., `SQLAlchemyError`) rather than bare `Exception`.
   When multiple operations can fail independently, use separate `try`/`except` blocks.

4. **Forgetting the migrations exclusion.** Pre-commit excludes the `migrations/` directory
   (`exclude: ^migrations/`). Auto-generated migration files should not be reformatted or linted.
   Do not remove this exclusion or manually lint migration files.

5. **Wrong test command flags.** CI runs `pipenv run pytest -v -s`, not plain `pipenv run pytest`.
   Omitting `-v -s` can mask test output and make failures harder to diagnose.

6. **Editing config.py without understanding the model registry.** LLM and embedding model
   configuration is centralized in `config.py` through the `MODELS` dict and environment variables.
   Adding new model support requires changes there, not in individual resource files.

[rhi]: https://github.com/RedHatInsights
[license]: https://github.com/RedHatInsights/tangerine-backend/blob/main/LICENSE
[readme-dev]: README.md#development
[readme-deploy]: README.md#deploying-to-openshift
[architecture]: ARCHITECTURE.md
[ruff]: https://docs.astral.sh/ruff/
[gh-actions-workflow]: .github/workflows/gh-actions.yml
