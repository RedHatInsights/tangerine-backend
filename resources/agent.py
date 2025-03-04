import json
import logging

from flask import Response, request, stream_with_context
from flask_restful import Resource

from connectors.config import DEFAULT_SYSTEM_PROMPT
from connectors.db.agent import Agent
from connectors.db.common import File, add_filenames_to_agent, embed_files, remove_files
from connectors.db.vector import vector_db
from connectors.llm.interface import llm

log = logging.getLogger("tangerine.agent")


class AgentDefaultsApi(Resource):
    def get(self):
        return {"system_prompt": DEFAULT_SYSTEM_PROMPT}, 200


class AgentsApi(Resource):
    def get(self):
        try:
            all_agents = Agent.list()
        except Exception:
            log.exception("error getting agents")
            return {"message": "error getting agents"}, 500

        return {"data": [agent.to_dict() for agent in all_agents]}, 200

    def post(self):
        name = request.json.get("name")
        description = request.json.get("description")
        if not name:
            return {"message": "agent 'name' required"}, 400
        if not description:
            return {"message": "agent 'description' required"}, 400

        try:
            agent = Agent.create(name, description, request.json.get("system_prompt"))
        except Exception:
            log.exception("error creating agent")
            return {"message": "error creating agent"}, 500

        return agent.to_dict(), 201


class AgentApi(Resource):
    def get(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404

        return agent.to_dict(), 200

    def put(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404

        data = request.get_json()
        # ignore 'id' or 'filenames' if provided in JSON payload
        data.pop("filenames", None)
        data.pop("id", None)
        agent.update(**data)

        return {"message": "agent updated successfully"}, 200

    def delete(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404

        agent.delete()
        vector_db.delete_document_chunks({"agent_id": agent.id})
        return {"message": "agent deleted successfully"}, 200


class AgentDocuments(Resource):
    def post(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404

        # Check if the post request has the file part
        if "file" not in request.files:
            return {"message": "No file part"}, 400

        request_source = request.form.get("source", "default")

        files = []
        for file in request.files.getlist("file"):
            content = file.stream.read()
            new_file = File(
                source=request_source, full_path=file.filename, content=content.decode("utf-8")
            )
            try:
                new_file.validate()
            except ValueError as err:
                return {"message": f"validation failed for {file.filename}: {str(err)}"}, 400
            files.append(new_file)

        def generate_progress():
            for file in files:
                yield json.dumps({"file": file.display_name, "step": "start"}) + "\n"
                embed_files([file], agent)
                add_filenames_to_agent([file], agent)
                yield json.dumps({"file": file.display_name, "step": "end"}) + "\n"

        return Response(stream_with_context(generate_progress()), mimetype="application/json")

    def delete(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404

        source = request.json.get("source")
        full_path = request.json.get("full_path")
        delete_all = bool(request.json.get("all", False))

        if not source and not full_path and not delete_all:
            return {"message": "'source' or 'full_path' required when not using 'all'"}, 400

        metadata = {}
        if source:
            metadata["source"] = source
        if full_path:
            metadata["full_path"] = full_path

        try:
            deleted = remove_files(agent, metadata)
        except ValueError as err:
            return {"message": str(err)}, 400
        except Exception:
            err = "unexpected error deleting document(s) from DB"
            log.exception(err)
            return {"message": err}, 500

        count = len(deleted)
        return {"message": f"{count} document(s) deleted", "count": count, "deleted": deleted}, 200


class AgentChatApi(Resource):
    def post(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "agent not found"}, 404
        agent_name = agent.agent_name

        query = request.json.get("query")
        stream = request.json.get("stream") == "true"
        previous_messages = request.json.get("prevMsgs")

        llm_response = llm.ask(
            agent.system_prompt, previous_messages, query, agent.id, agent_name, stream
        )

        if stream:
            return Response(llm_response(), mimetype="application/json")

        return llm_response, 200
