# tangerine-backend

🍊

A work in progress

## Local Environment Setup

The local dev environment uses ollama to get you up and running quickly.

You may require further tweaks to properly make use of your GPU. Refer to the [ollama docker image documentation](https://hub.docker.com/r/ollama/ollama).

1. Invoke docker compose (postgres data will persist in `./postgres-data`):
    ```
    docker-compose up -d --build
    ```
2. Pull needed ollama models (data will persist in `./ollama`):
    ```
    docker exec tangerine-ollama ollama pull mistral
    docker exec tangerine-ollama ollama pull nomic-embed-text
    ```
3. The API can now be accessed on `http://localhost:5000`
4. To use UI, install and start [tangerine-frontend](https://github.com/tahmidefaz/tangerine-frontend)


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
