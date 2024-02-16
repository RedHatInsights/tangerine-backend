from flask import Flask
from flask_restful import Api

from connectors.vector_store.db import db, db_connection_string, vector_interface
from resources.routes import initialize_routes

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = db_connection_string

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
