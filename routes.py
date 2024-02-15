from flask import Blueprint, Response, jsonify, request
import time


api_routes = Blueprint('api_routes', __name__)

data = {
    "ping": "hello world."
}


@api_routes.route('/hello', methods=['GET'])
def get_hello():
    return jsonify(data)


@api_routes.route('/create_agent', methods=['POST'])
def create_agent():
    return jsonify({"created": "agent_id"}), 201


@api_routes.route('/<agent_id>/update_agent', methods=['PATCH'])
def update_agent(agent_id):
    return jsonify({"updated": agent_id})


@api_routes.route('/<agent_id>/delete_agent', methods=['DELETE'])
def delete_agent(agent_id):
    return jsonify({"deleted": agent_id})


@api_routes.route('/<agent_id>/chat', methods=['GET'])
def agent_chat(agent_id):
    def generate():
        for c in agent_id:
            yield f"{c}"
            time.sleep(1)
    return Response(generate(), mimetype='text/plain')
