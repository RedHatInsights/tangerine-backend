import json
from flask import Response, request
from flask_restful import Resource
import time

from connectors.vector_store.db import db, Agents
from connectors.vector_store.db import vector_interface
from connectors.llm.interface import llm
from utils.processors import text_extractor


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

        # Don't let them create the agent id and filenames
        agent.pop("id", None)
        agent.pop("filenames", None)

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

            # Don't let them update the agent id and filenames
            data.pop("id", None)
            data.pop("filenames", None)

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

        # TODO: delete agent documents from vector store

class AgentDocUpload(Resource):
    def post(self, id):
        agent = Agents.query.get(id)
        if not agent:
            return {'message': 'Agent not found'}, 404

        # Check if the post request has the file part
        if 'file' not in request.files:
            return {'message': 'No file part'}, 400

        files = request.files.getlist('file')

        file_contents=[]
        for file in files:
            filename = file.filename
            if not any([filename.endswith(filetype) for filetype in [".txt", ".pdf", ".md"]]):
                return {'message': 'Unsupported file type uploaded'}, 400

            file_content = file.stream.read()

            file_contents.append([filename, file_content])

        # Add filenames to the DB
        new_filenames = agent.filenames.copy()
        for fileinfo in file_contents:
            new_filenames.append(fileinfo[0])
        agent.filenames = new_filenames
        db.session.commit()

        def generate_progress():
            for filename, file_content in file_contents:
                yield json.dumps({"file": filename, "step": "start"}) + "\n"
                extracted_text = text_extractor(filename, file_content)
                yield json.dumps({"file": filename, "step": "text_extracted"}) + "\n"

                # Only generate embeddings when there is actual texts
                if len(extracted_text) > 0:
                    vector_interface.add_document(extracted_text, id)
                    yield json.dumps({"file": filename, "step": "embedding_created"}) + "\n"

                yield json.dumps({"file": filename, "step": "end"}) + "\n"

        return Response(generate_progress(), mimetype='application/json')


class AgentChatApi(Resource):
    def get(self, id):
        def generate():
            for c in id:
                yield f"{c}"
                time.sleep(1)
        return Response(generate(), mimetype='text/plain')
    def post(self,id):
        agent = Agents.query.filter_by(id=id).first()
        if not agent:
            return {'message': 'Agent not found'}, 404
        query = request.json.get("query")
        return {'answer':llm.ask(query,id)} , 200

