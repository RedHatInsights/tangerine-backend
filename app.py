import datetime as dt
import logging
import os
import sys
import time

import click
import langchain
from flask import Flask, current_app
from flask.cli import with_appcontext
from flask_cors import CORS
from flask_migrate import Migrate
from flask_restful import Api

import connectors.config as cfg
import connectors.s3.sync

# Imported so SQLAlchemy can find the models
from connectors.db import interactions as _ # noqa: F401
from connectors.db.agent import db
from connectors.db.vector import vector_db
from resources.metrics import metrics
from resources.routes import initialize_routes


IGNORE_TABLES = ['langchain_pg_collection', 'langchain_pg_embedding']
def include_object(object, name, type_, reflected, compare_to):
    """
    Should you include this table or not?
    """
    if type_ == 'table' and (name in IGNORE_TABLES or object.info.get("skip_autogenerate", False)):
        return False

    elif type_ == "column" and object.info.get("skip_autogenerate", False):
        return False

    return True


def create_app():
    logging.basicConfig(level=getattr(logging, cfg.LOG_LEVEL_GLOBAL))
    logging.getLogger("tangerine").setLevel(cfg.LOG_LEVEL_APP)

    if cfg.DEBUG_VERBOSE:
        langchain.debug = True

    app = Flask("tangerine")

    app.config["CORS_HEADERS"] = "Content-Type"
    app.config["SQLALCHEMY_DATABASE_URI"] = cfg.DB_URI

    CORS(app)

    db.init_app(app)

    Migrate(app, db, include_object=include_object)
    
    api = Api(app)
    initialize_routes(api)

    metrics.init_app(app, api)
    app.cli.add_command(s3sync)

    with app.app_context():
        db.session.commit()
        vector_db.initialize()
        app.logger.info("vector store initiated.")
        db.create_all()
        app.logger.info("db tables initiated.")
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
    if cfg.S3_SYNC_EXPORT_METRICS:
        os.environ.pop("FLASK_RUN_FROM_CLI")
        metrics.start_http_server(port=8000)

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

    if cfg.S3_SYNC_EXPORT_METRICS:
        sleep_time = cfg.S3_SYNC_EXPORT_METRICS_SLEEP_SECS
        current_app.logger.info("sleeping %dsec for metrics collection", sleep_time)
        time.sleep(sleep_time)

    sys.exit(exit_code)
