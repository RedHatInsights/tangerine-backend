from flask import Blueprint, Response, jsonify, request
import time


agents = Blueprint('agents', __name__)


@agents.route('/agents', methods=['GET'])
def get_agents():
    return jsonify({"data": []})

@agents.route('/agents', methods=['POST'])
def create_agent():
    return jsonify({"created": "agent_id"}), 201

@agents.route('/agents/<agent_id>', methods=['PUT'])
def update_agent(agent_id):
    return jsonify({"updated": agent_id})

@agents.route('/agents/<agent_id>', methods=['DELETE'])
def delete_agent(agent_id):
    return jsonify({"deleted": agent_id})

@agents.route('/agents/<agent_id>/chat', methods=['GET'])
def agent_chat(agent_id):
    def generate():
        for c in agent_id:
            yield f"{c}"
            time.sleep(1)
    return Response(generate(), mimetype='text/plain')

@agents.route('/hello', methods=['GET'])
def get_ping():
    return jsonify({"ping": "hello world."})
