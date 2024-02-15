from flask import Flask
from routes import agents

app = Flask(__name__)

app.register_blueprint(agents)


if __name__ == "__main__":
    app.run(debug=True)
