from flask import Response, request
from flask_restful import Resource
import time

from connectors.vector_store.db import db, Agents


class AgentsApi(Resource):
    def get(self):
        try:
            all_agents = Agents.query.all()
        except Exception as e:
            print(f"Exception AgentsApi GET: {e}")
            return {'message': 'error fetching agents form DB'}, 500

        agents_list = []

        for agent in all_agents:
            agents_list.append({
                'id': agent.id,
                'agent_name': agent.agent_name,
                'description': agent.description,
                'system_prompt': agent.system_prompt,
            })
        return {'data': agents_list}, 200


    def post(self):
        agent = {
            "agent_name": request.form["name"],
            "description": request.form["description"]
        }

        try:
            new_data = Agents(**agent)
            db.session.add(new_data)
            db.session.commit()
        except Exception as e:
            print(f"Exception AgentsApi POST: {e}")
            return {'message': 'error inserting agent into DB'}, 500

        return agent, 201

class AgentApi(Resource):
    def get(self, id):
        agent = Agents.query.filter_by(id=id).first()

        if not agent:
            return {'message': 'Agent not found'}, 404

        return {
            'id': agent.id,
            'agent_name': agent.agent_name,
            'description': agent.description,
            'system_prompt': agent.system_prompt,
        }, 200


    def put(self, id):
        agent = Agents.query.get(id)
        if agent:
            data = request.get_json()

            # Don't let them update the agent id
            data.pop("id", None)

            for key, value in data.items():
                # Update each field if it exists in the request data
                setattr(agent, key, value)
            db.session.commit()
            return {'message': 'Agent updated successfully'}, 200
        else:
            return {'message': 'Agent not found'}, 404

    def delete(self, id):
        agent = Agents.query.get(id)
        if agent:
            db.session.delete(agent)
            db.session.commit()
            return {'message': 'Agent deleted successfully'}, 200
        else:
            return {'message': 'Agent not found'}, 404

class AgentChatApi(Resource):
    def get(self, id):
        def generate():
            for c in id:
                yield f"{c}"
                time.sleep(1)
        return Response(generate(), mimetype='text/plain')
