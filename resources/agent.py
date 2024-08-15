import json
import logging

from flask import Response, request
from flask_restful import Resource
from sqlalchemy import text

from connectors.llm.interface import llm
from connectors.vector_store.db import Agents, db, vector_interface
from utils.processors import text_extractor

log = logging.getLogger("tangerine.agent")


class AgentsApi(Resource):
    def get(self):
        try:
            all_agents = Agents.query.all()
        except Exception:
            log.exception("exception in AgentsApi GET")
            return {"message": "error fetching agents from DB"}, 500

        agents_list = []

        for agent in all_agents:
            agents_list.append(
                {
                    "id": agent.id,
                    "agent_name": agent.agent_name,
                    "description": agent.description,
                    "system_prompt": agent.system_prompt,
                    "filenames": agent.filenames,
                }
            )
        return {"data": agents_list}, 200

    def post(self):
        agent = {
            "agent_name": request.form["name"],
            "description": request.form["description"],
            "system_prompt": request.form["system_prompt"],
        }

        if len(agent["agent_name"]) < 1:
            return {"message": "agent_name is required."}, 400

        # Don't let them create the agent id and filenames
        agent.pop("id", None)
        agent.pop("filenames", None)

        try:
            new_data = Agents(**agent)
            db.session.add(new_data)
            db.session.commit()
        except Exception:
            log.exception("exception in AgentsApi POST")
            return {"message": "error inserting agent into DB"}, 500

        return agent, 201


class AgentApi(Resource):
    def get(self, id):
        id = int(id)
        agent = Agents.query.filter_by(id=id).first()

        if not agent:
            return {"message": "Agent not found"}, 404

        return {
            "id": agent.id,
            "agent_name": agent.agent_name,
            "description": agent.description,
            "system_prompt": agent.system_prompt,
            "filenames": agent.filenames,
        }, 200

    def put(self, id):
        id = int(id)
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
            return {"message": "Agent updated successfully"}, 200
        else:
            return {"message": "Agent not found"}, 404

    def delete(self, id):
        id = int(id)
        agent = Agents.query.get(id)
        if agent:
            db.session.delete(agent)
            db.session.commit()
            return {"message": "Agent deleted successfully"}, 200
        else:
            return {"message": "Agent not found"}, 404

        # TODO: delete agent documents from vector store


class AgentDocUpload(Resource):
    def get_file_id(self, source, full_path):
        return f"{source}:{full_path}"

    def post(self, id):
        id = int(id)
        agent = Agents.query.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        # Check if the post request has the file part
        if "file" not in request.files:
            return {"message": "No file part"}, 400

        files = request.files.getlist("file")
        source = request.form.get("source")

        file_contents = []
        for file in files:
            full_path = file.filename
            file_id = self.get_file_id(source, full_path)
            if not any(
                [
                    file_id.endswith(filetype)
                    for filetype in [".txt", ".pdf", ".md", ".rst", ".html"]
                ]
            ):
                return {"message": "Unsupported file type uploaded"}, 400

            file_content = file.stream.read()

            file_contents.append([file_id, full_path, file_content])

        # Add filenames to the DB
        new_full_paths = agent.filenames.copy()
        for fileinfo in file_contents:
            new_full_paths.append(fileinfo[0])
        agent.filenames = new_full_paths
        db.session.commit()

        def generate_progress():
            for _, full_path, file_content in file_contents:
                yield json.dumps({"file": full_path, "step": "start"}) + "\n"
                extracted_text = text_extractor(full_path, file_content)
                yield json.dumps({"file": full_path, "step": "text_extracted"}) + "\n"

                # Only generate embeddings when there is actual texts
                if len(extracted_text) > 0:
                    vector_interface.add_document(extracted_text, id, source, full_path)
                    yield json.dumps({"file": full_path, "step": "embedding_created"}) + "\n"

                yield json.dumps({"file": full_path, "step": "end"}) + "\n"

        return Response(generate_progress(), mimetype="application/json")

    def delete(self, id):
        full_path = request.json.get("full_path", None)
        source = request.json.get("source")

        # delete single file
        if full_path:
            query = text(
                f"SELECT id FROM langchain_pg_embedding WHERE cmetadata->>'source'='{source}'"
                f" AND cmetadata->>'agent_id'='{id}'"
                f" AND cmetadata->>'full_path'='{full_path}';"
            )
            try:
                # delete documents from vector store
                documents = db.session.execute(query).all()
                if len(documents) == 0:
                    return {"message": f"File {full_path} not found."}, 400
                vector_interface.delete_documents([document[0] for document in documents])

            except Exception as e:
                return {"message": f"Error deleting {full_path} from vector store. {e}"}, 400

            # delete documents from agent
            try:
                id = int(id)
                agent = Agents.query.get(id)
                file_id = self.get_file_id(source, full_path)
                new_full_paths = [file for file in agent.filenames.copy() if file != file_id]
                agent.filenames = new_full_paths
                db.session.commit()
            except Exception as e:
                return {"message": f"Error deleting {full_path} from Agent {id}. {e}"}, 400

            return {"message": f"File {full_path} deleted successfully."}, 200

        # delete all files from a source
        else:
            query = text(
                "SELECT id, cmetadata FROM langchain_pg_embedding"
                f" WHERE cmetadata->>'source'='{source}'"
                f" AND cmetadata->>'agent_id'='{id}';"
            )
            try:
                # delete documents from vector store
                documents = db.session.execute(query).all()
                if len(documents) == 0:
                    return {"message": f"No files from the source {source} found."}, 400
                vector_interface.delete_documents([document[0] for document in documents])

            except Exception as e:
                return {
                    "message": f"Error deleting files from source {source} from vector store. {e}"
                }, 400

            # delete documents from agent
            try:
                id = int(id)
                agent = Agents.query.get(id)
                paths_to_remove = {
                    self.get_file_id(source, document[1]["full_path"]) for document in documents
                }
                new_full_paths = [
                    file for file in agent.filenames.copy() if file not in paths_to_remove
                ]
                agent.filenames = new_full_paths
                db.session.commit()
            except Exception as e:
                return {
                    "message": f"Error deleting files from source {source} from Agent {id}. {e}"
                }, 400

            return {"message": f"Files from source {source} deleted successfully."}, 200


class AgentChatApi(Resource):
    def post(self, id):
        id = int(id)
        agent = Agents.query.filter_by(id=id).first()
        if not agent:
            return {"message": "Agent not found"}, 404

        query = request.json.get("query")
        stream = request.json.get("stream") == "true"
        previous_messages = request.json.get("prevMsgs")

        llm_response = llm.ask(agent.system_prompt, previous_messages, query, agent.id, stream)

        if stream:
            return Response(llm_response(), mimetype="application/json")

        return llm_response, 200
