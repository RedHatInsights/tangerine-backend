# tangerine-backend

🍊

A work in progress

## Local Environment Setup

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


To run the UI in a development environment, see [tangerine-frontend](https://github.com/coderbydesign/tangerine-frontend)


### Available API Paths
| Path                               | Method   | Description                |
| ---------------------------------- | -------- | -------------------------- |
| `/api/agents`                      | `GET`    | Get a list of all agents   |
| `/api/agents`                      | `POST`   | Create a new agent         |
| `/api/agents/<id>`                 | `GET`    | Get an agent               |
| `/api/agents/<id>`                 | `PUT`    | Update an agent            |
| `/api/agents/<id>`                 | `DELETE` | Delete an agent            |
| `/api/agents/<id>/document_upload` | `POST`   | Agent document uploads     |
| `/api/agents/<id>/chat`            | `POST`   | Chat with an agent         |
| `/api/agentDefaults`               | `GET`    | Get agent default settings |
| `/ping`                            | `GET`    | Health check               |
