from flask import Flask
from flask_restful import Api

from connectors.vector_store.db import db, vector_interface
import connectors.config as cfg
from resources.routes import initialize_routes

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = cfg.DB_URI

db.init_app(app)

api = Api(app)

initialize_routes(api)


if __name__ == "__main__":
    with app.app_context():
        db.session.commit()
        db.create_all()
        print("db tables initiated.")

        vector_interface.init_vector_store()
        print("vector store initiated.")

    app.run(debug=True)
