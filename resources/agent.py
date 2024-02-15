from flask import Response, request
from flask_restful import Resource
import time

class AgentsApi(Resource):
    def get(self):
        return {"data": []}

    def post(self):
        return {"created": "agent_id"}, 201

class AgentApi(Resource):
    def get(self, id):
        return {"id": id}

    def put(self, id):
        return {"updated": id}

    def delete(self, id):
        return {"deleted": id}

class AgentChatApi(Resource):
    def get(self, id):
        def generate():
            for c in id:
                yield f"{c}"
                time.sleep(1)
        return Response(generate(), mimetype='text/plain')
