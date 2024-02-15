from flask import Response, request
from flask_restful import Resource

class PingApi(Resource):
    def get(self):
        return {"ping": "hello world."}
