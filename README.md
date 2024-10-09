# tangerine-backend

🍊

A work in progress

## Local Environment Setup for Linux / Intel Macs

The local dev environment uses ollama to serve the LLM.

You may require further tweaks to properly make use of your GPU. Refer to the [ollama docker image documentation](https://hub.docker.com/r/ollama/ollama).

1. Make sure [git-lfs](https://git-lfs.com/) is installed:
    ```
    Fedora: `sudo dnf install git-lfs`
    MacOS: `brew install git-lfs`

    git lfs install
    ```
1. Create the directory which will house the local environment data:
    ```
    mkdir data
    ```
1. Create a directory to house the embedding model and download the `snowflake-arctic-embed-m-long` model:
    ```
    mkdir data/embeddings
    git clone https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long \
      data/embeddings/snowflake-arctic-embed-m-long
    ```
1. Invoke docker compose (postgres data will persist in `data/postgres`):
    ```
    docker compose up --build
    ```
1. Pull the mistral LLM (data will persist in `data/ollama`):
    ```
    docker exec tangerine-ollama ollama pull mistral
    ```
1. The API can now be accessed on `http://localhost:5000`

## Local Environment Setup for Apple Silicon Macs

Some of the images used in the `docker-compose.yml` wont work on Apple Silicon Macs. In order to develop on those system you will need to start some of the processes manually.

You'll ned to have the following installed and working before proceeding on:
* Brew
* Pipenv
* Pyenv
* Docker or Podman

1. Install Ollama

```sh
brew install ollama
```

2. Start Ollama

```sh
ollama serve
```

3. Pull the language and embedding models

```sh
ollama pull mistral
ollama pull nomic-embed-text
```

3. Start the Vector database

```sh
docker run -e POSTGRES_PASSWORD="citrus" -e POSTGRES_USER="citrus" -e POSTGRES_DB="citrus" -e POSTGRES_HOST_AUTH_METHOD=trust -p 5432:5432 pgvector/pgvector:pg16
```

4. Prepare the Python virtual environment:

```sh
pipenv --python=3.11
pipenv install
pipenv shell
```

5. Start Tangerine Backend

**Note:* The default tangerine port, 5000, is already claimed by Bonjour on Macs, so we need to use a different port instead.*
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

## Run Tangerine Frontend Locally

To run the UI in a development environment, see [tangerine-frontend](https://github.com/coderbydesign/tangerine-frontend)

## Populate Data from S3
You can populate the vector database and list of agents by processing documents pulled from an S3 bucket. To do so you'll need to do the following:

1. Export environment variables that contain your S3 bucket auth info:
```sh
export AWS_ACCESS_KEY_ID="MYKEYID"
export AWS_DEFAULT_REGION="us-east-1"
export AWS_ENDPOINT_URL_S3="https://s3.us-east-1.amazonaws.com"
export AWS_SECRET_ACCESS_KEY="MYACCESSKEY"
export BUCKET="mybucket"
```

2. Create an `s3.yaml` file that describes your agents and the documents they should ingest. See `s3-example.yaml` for an example.

3. Run the ingest script

```sh
flask s3sync
```

4. Start the server and you should see your data available via the API.



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
| `/ping`                            | `GET`    | Health check               |
