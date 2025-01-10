# 🍊 tangerine (backend)

tangerine is a slim and light-weight RAG (Retieval Augmented Generated) system used to create and manage chat bot agents.

Each agent is intended to answer questions related to a set of documents known as a knowledge base (KB).

It relies on 4 key components:

* A vector database (PostgresQL with the pgvector extension)
* A large language model (LLM) hosted on any OpenAI-compatible API service.
* An embedding model hosted on any OpenAI-compatible API service.
* (optional) An S3 bucket that you wish to sync documentation from.

The backend service manages:
* Creating/updating/deleting chat bot agents
* Uploading documents to be used as context to assist the agents in answering questions
* Document ingestion including cleanup/conversion, chunking, and embedding into the vector database.
* Document chunk retrieval from the vector database.
* Interfacing with the LLM to prompt it and stream responses
* (optional) Interfacing with S3 to provide continuous document sync.

tangerine will work with any deployed instance of PostgresQL+pgvector and can be configured to use any OpenAI-compliant API service that is hosting a large language model or embedding model.

This repository provides Open Shift templates for all infrastructure (except for the model hosting service) as well as a docker compose file that allows you to spin it up locally and use [ollama](https://ollama.com/).

The accompanying frontend service is [tangerine-frontend](https://github.com/RedHatInsights/tangerine-frontend) and a related plugin for [Red Hat Developer Hub](https://developers.redhat.com/rhdh/overview) can be found [here](https://github.com/RedHatInsights/backstage-plugin-ai-search-frontend)

This project is currently used by Red Hat's Hybrid Cloud Management Engineering Productivity Team. It was born out of a hack-a-thon and is still a work in progress. You will find some areas of code well developed while others are in need of attention and some tweaks to make it production-ready are needed (with that said, the project *is* currently in good enough shape to provide a working chat bot system).

## Getting started

The project can be deployed to a local development environment using ollama to host the LLM and huggingface's [text-embeddings-inference](https://github.com/huggingface/text-embeddings-inference) server to host the embedding model.


### Local Environment Setup for Linux / Intel Macs

You may require further tweaks to properly make use of your GPU. Refer to the [ollama docker image documentation](https://hub.docker.com/r/ollama/ollama).

1. Make sure [git-lfs](https://git-lfs.com/) is installed:

    * Fedora: `sudo dnf install git-lfs`
    * MacOS: `brew install git-lfs`

    Then, activate it globally with:

    ```text
    git lfs install
    ```

2. Create the directory which will house the local environment data:

    ```text
    mkdir data
    ```

3. Create a directory to house the embedding model and download the `snowflake-arctic-embed-m-long` model:

    ```text
    mkdir data/embeddings
    git clone https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long \
      data/embeddings/snowflake-arctic-embed-m-long
    ```

4. Invoke docker compose (postgres data will persist in `data/postgres`):

    ```text
    docker compose up --build
    ```

5. Pull the mistral LLM (data will persist in `data/ollama`):

    ```text
    docker exec tangerine-ollama ollama pull mistral
    ```

6. The API can now be accessed on `http://localhost:5000`


### Local Environment Setup for Apple Silicon Macs

Some of the images used in the `docker-compose.yml` are unsupported on Apple silicon. In order to develop on those systems you will need to start some of the processes manually.

1. You'll need to have the following installed and working before proceeding:

   * brew
   * pipenv
   * pyenv
   * docker or podman

2. Install ollama

    ```text
    brew install ollama
    ```

2. Start ollama

    ```sh
    ollama serve
    ```

3. Pull the language and embedding models

    ```text
    ollama pull mistral
    ollama pull nomic-embed-text
    ```

4. Install the C API for Postgres (libpq)

    ```sh
    brew install libpq
    ```

    For Apple Silicon Macs, you'll need to export the following environment variables to avoid C library errors:

    ```text
    export PATH="/opt/homebrew/opt/libpq/bin:$PATH"
    export LDFLAGS="-L/opt/homebrew/opt/libpq/lib"
    export CPPFLAGS="-I/opt/homebrew/opt/libpq/include"
    ```

5. Start the vector database

    ```text
    docker run \
        -e POSTGRES_PASSWORD="citrus" \
        -e POSTGRES_USER="citrus" \
        -e POSTGRES_DB="citrus" \
        -e POSTGRES_HOST_AUTH_METHOD=trust \
        -p 5432:5432 \
        pgvector/pgvector:pg16
    ```

6. Prepare your python virtual environment:

   ```sh
   pipenv install
   pipenv shell
   ```

7. Start Tangerine Backend

    > [!NOTE]
    > The default tangerine port, 5000, is already claimed by Bonjour on Macs, so we need to use a different port instead.

    ```sh
    flask run --host=127.0.0.1 --port=8000
    ```

   You can now communicate with the API on port `8000`

   ```sh
   curl -XGET 127.0.0.1:8000/api/agents
   {
       "data": []
   }
   ```

### Available API Paths

| Path                               | Method   | Description                |
| ---------------------------------- | -------- | -------------------------- |
| `/api/agents`                      | `GET`    | Get a list of all agents   |
| `/api/agents`                      | `POST`   | Create a new agent         |
| `/api/agents/<id>`                 | `GET`    | Get an agent               |
| `/api/agents/<id>`                 | `PUT`    | Update an agent            |
| `/api/agents/<id>`                 | `DELETE` | Delete an agent            |
| `/api/agents/<id>/chat`            | `POST`   | Chat with an agent         |
| `/api/agents/<id>/documents`       | `POST`   | Agent document uploads     |
| `/api/agents/<id>/documents`       | `DELETE` | Delete agent documents     |
| `/api/agentDefaults`               | `GET`    | Get agent default settings |
| `/ping`

## Syncrhonizing Documents from S3

You can configure a set of agents and continually sync their knowledge base via documents stored in an S3 bucket.

To do so you'll need to do the following:

1. Export environment variables that contain your S3 bucket auth info:

   ```sh
   export AWS_ACCESS_KEY_ID="MYKEYID"
   export AWS_DEFAULT_REGION="us-east-1"
   export AWS_ENDPOINT_URL_S3="https://s3.us-east-1.amazonaws.com"
   export AWS_SECRET_ACCESS_KEY="MYACCESSKEY"
   export BUCKET="mybucket"
   ```

2. Create an `s3.yaml` file that describes your agents and the documents they should ingest. See [s3-example.yaml](s3-example.yaml) for an example.

3. Run the S3 sync job:

    ```sh
    flask s3sync
    ```

The sync creates agents and ingests the configured documents for each agent. After initial creation, when the task is run it checks the S3 bucket for updates and will only re-ingest files into the vector DB when it detects file changes.

The OpenShift templates contain a CronJob configuration that is used to run this document sync repeatedly.

## Run Tangerine Frontend Locally

The API can be used to create/manage/update agents, upload documents, and to chat with each agent. However, the frontend provides a simpler interface to manage the service with. To run the UI in a development environment, see [tangerine-frontend](https://github.com/RedHatInsights/tangerine-frontend)
