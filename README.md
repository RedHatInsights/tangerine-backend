# tangerine-backend

🍊

A work in progress

## Local Environment Setup

The local dev environment uses ollama to serve the LLM.

You may require further tweaks to properly make use of your GPU. Refer to the [ollama docker image documentation](https://hub.docker.com/r/ollama/ollama).

1. Create the directory which will house the local environment data:
    ```
    mkdir data
    ```
2. Create a directory to house the embedding model and download the `snowflake-arctic-embed-m-long` model:
    ```
    mkdir data/embeddings
    git clone https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long \
      data/embeddings/snowflake-arctic-embed-m-long
    ```
3. Invoke docker compose (postgres data will persist in `data/postgres`):
    ```
    docker compose up --build
    ```
4. Pull the mistral LLM (data will persist in `data/ollama`):
    ```
    docker exec tangerine-ollama ollama pull mistral
    ```
5. The API can now be accessed on `http://localhost:5000`
6. To use UI, install and start [tangerine-frontend](https://github.com/tahmidefaz/tangerine-frontend)


### Available API Paths
| Path                           | Method   | Description              |
| ------------------------------ | -------- | ------------------------ |
| `/agents`                      | `GET`    | Get a list of all agents |
| `/agents`                      | `POST`   | Create a new agent       |
| `/agents/<id>`                 | `GET`    | Get an agent             |
| `/agents/<id>`                 | `PUT`    | Update an agent          |
| `/agents/<id>`                 | `DELETE` | Delete an agent          |
| `/agents/<id>/document_upload` | `POST`   | Agent document uploads   |
| `/agents/<id>/chat`            | `POST`   | Chat with an agent       |
| `/ping`                        | `GET`    | Health check             |
