import logging

from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy

from tangerine.config import SQLALCHEMY_MAX_OVERFLOW, SQLALCHEMY_POOL_SIZE

log = logging.getLogger("tangerine.db")

db = SQLAlchemy(
    engine_options={"pool_size": SQLALCHEMY_POOL_SIZE, "max_overflow": SQLALCHEMY_MAX_OVERFLOW}
)


def include_object(obj, name, db_type, _reflected, _compare_to):
    ignore_tables = ["langchain_pg_collection", "langchain_pg_embedding"]

    if db_type == "table" and (name in ignore_tables or obj.info.get("skip_autogenerate", False)):
        return False

    return True


migrate = Migrate(include_object=include_object)
