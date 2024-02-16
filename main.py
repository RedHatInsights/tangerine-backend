from flask import Flask
from flask_restful import Api
from sqlalchemy import text

from connectors.vector_store.db import db
from resources.routes import initialize_routes

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://citrus:citrus@localhost/citrus'

db.init_app(app)

api = Api(app)

initialize_routes(api)


if __name__ == "__main__":
    with app.app_context():
        db.session.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
        db.session.commit()
        db.create_all()

        print("db tables initiated.")

    app.run(debug=True)
