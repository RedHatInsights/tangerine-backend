from flask import Flask
from flask_restful import Api
from flask_cors import CORS, cross_origin

from connectors.vector_store.db import db, db_connection_string, vector_interface
from resources.routes import initialize_routes

app = Flask(__name__)
cors = CORS(app)

app.config['CORS_HEADERS'] = 'Content-Type'
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
