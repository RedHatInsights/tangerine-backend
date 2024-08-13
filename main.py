from flask import Flask
from flask_cors import CORS
from flask_restful import Api
import logging

import connectors.config as cfg
from connectors.vector_store.db import db, vector_interface
from resources.routes import initialize_routes

app = Flask("tangerine")
cors = CORS(app)

app.config["CORS_HEADERS"] = "Content-Type"
app.config["SQLALCHEMY_DATABASE_URI"] = cfg.DB_URI

db.init_app(app)

api = Api(app)

initialize_routes(api)


if __name__ == "__main__":
    app.logger.setLevel(logging.DEBUG)
    with app.app_context():
        db.session.commit()
        db.create_all()
        app.logger.info("db tables initiated.")

        vector_interface.init_vector_store()
        app.logger.info("vector store initiated.")

    app.run(host="0.0.0.0", debug=True)
