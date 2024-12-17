import datetime as dt
import logging
import os
import sys

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
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL_GLOBAL))
    logging.getLogger("tangerine").setLevel(cfg.LOG_LEVEL_APP)

    app = Flask("tangerine")

    app.config["CORS_HEADERS"] = "Content-Type"
    app.config["SQLALCHEMY_DATABASE_URI"] = cfg.DB_URI

    CORS(app)

    db.init_app(app)

    api = Api(app)
    initialize_routes(api)

    app.cli.add_command(s3sync)

    with app.app_context():
        db.session.commit()
        db.create_all()
        app.logger.info("db tables initiated.")

        vector_db.init_vector_store()
        app.logger.info("vector store initiated.")

    return app


@click.command("s3sync")
@click.option("--force-resync", is_flag=True, help="Delete all files from agents and re-import")
@click.option(
    "--force-resync-until",
    help=(
        "Timestamp in ISO 8601 format. Run a force resync unless current time is later than "
        "specified time"
    ),
)
@with_appcontext
def s3sync(force_resync, force_resync_until):
    force_resync_env_var = os.getenv("FORCE_RESYNC", "").lower() in ("1", "true")
    force_resync_until_env_var = os.getenv("FORCE_RESYNC_UNTIL")

    force_resync = force_resync or force_resync_env_var
    force_resync_until = force_resync_until or force_resync_until_env_var

    if not force_resync and force_resync_until:
        expires_dt = dt.datetime.fromisoformat(force_resync_until_env_var)
        now_dt = dt.datetime.now(dt.timezone.utc).replace(microsecond=0)
        if now_dt < expires_dt:
            force_resync = True
        current_app.logger.info(
            "force resync until %s (utc: %s), current time: %s, force_resync=%s",
            expires_dt,
            expires_dt.astimezone(dt.timezone.utc),
            now_dt,
            force_resync,
        )

    current_app.logger.info("running s3sync, force_resync=%s", force_resync)
    exit_code = connectors.s3.sync.run(resync=force_resync)
    sys.exit(exit_code)
