# tangerine-backend
🍊

## Setup
* Install [Ollama](https://ollama.com/) on your machine, and pull their version of the `mistral` model.
* `pipenv install`
* `pipenv shell`

## Spin up the DB
* `docker-compose up`

### Run the API Server
* `python main.py`
* access on `http://localhost:5000`


### Available API Paths
| Path                           | Method   | Description              |
| ------------------------------ | -------- | ------------------------ |
| `/agents`                      | `GET`    | Get a list of all agents |
| `/agents`                      | `POST`   | Create a new agent       |
| `/agents/<id>`                 | `GET`    | Get an agent             |
| `/agents/<id>`                 | `PUT`    | Update an agent          |
| `/agents/<id>`                 | `DELETE` | Delete an agent          |
| `/agents/<id>/document_upload` | `POST`   | Agent document uploads   |
| `/agents/<id>/chat`            | `POST`    | Chat with an agent       |
| `/ping`                        | `GET`    | Health check             |
