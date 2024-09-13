import logging

import click
from flask import Flask, current_app
from flask.cli import with_appcontext
from flask_cors import CORS
from flask_restful import Api

import connectors.config as cfg
import connectors.s3.sync
from connectors.db.agent import db
from connectors.db.vector import vector_db
from resources.routes import initialize_routes


def create_app():
    app = Flask("tangerine")

    app.config["CORS_HEADERS"] = "Content-Type"
    app.config["SQLALCHEMY_DATABASE_URI"] = cfg.DB_URI

    CORS(app)

    db.init_app(app)

    api = Api(app)
    initialize_routes(api)

    app.cli.add_command(s3sync)

    app.logger.setLevel(getattr(logging, cfg.LOG_LEVEL))

    with app.app_context():
        db.session.commit()
        db.create_all()
        app.logger.info("db tables initiated.")

        vector_db.init_vector_store()
        app.logger.info("vector store initiated.")

    return app


@click.command()
@with_appcontext
def s3sync():
    current_app.logger.info("running s3sync")
    connectors.s3.sync.run()
