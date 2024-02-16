# tangerine-backend
🍊

## Setup
* `pipenv install`
* `pipenv run pip install psycopg2-binary`
* `pipenv shell`

### Run the API Server
* `python main.py`
* access on `http://localhost:5000`


### Available API Paths
| Path                | Method   | Description              |
| ------------------- | -------- | ------------------------ |
| `/agents`           | `GET`    | Get a list of all agents |
| `/agents`           | `POST`   | Create a new agent       |
| `/agents/<id>`      | `GET`    | Get an agent             |
| `/agents/<id>`      | `PUT`    | Update an agent          |
| `/agents/<id>`      | `DELETE` | Delete an agent          |
| `/agents/<id>/chat` | `GET`    | Chat with an agent       |
| `/ping`             | `GET`    | Health check             |
