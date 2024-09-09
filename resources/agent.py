import json
import logging
import re
from typing import List

from flask import Response, request
from flask_restful import Resource

from connectors.config import DEFAULT_SYSTEM_PROMPT
from connectors.db.agent import Agent
from connectors.db.vector import vector_db
from connectors.llm.interface import llm
from utils.processors import text_extractor

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
        try:
            agent = Agent.create(
                request.form["name"], request.form["description"], request.form.get("system_prompt")
            )
        except Exception:
            log.exception("error creating agent")
            return {"message": "error creating agent"}, 500

        return agent.to_dict(), 201


class AgentApi(Resource):
    def get(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        return agent.to_dict(), 200

    def put(self, id):
        data = request.get_json()
        id = data.pop("id", None)
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        # do not allow filenames to be updated via PUT
        data.pop("filenames", None)
        agent.update(**data)

        return {"message": "Agent updated successfully"}, 200

    def delete(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        agent.delete()
        vector_db.delete_documents_by_metadata({"agent_id": str(agent.id)})
        return {"message": "Agent deleted successfully"}, 200


def _validate_source(source: str) -> None:
    source_regex = r"^[\w-]+$"
    if not source or not source.strip() or not re.match(source_regex, source):
        raise ValueError(f"source must match regex: {source_regex}")


def _validate_file_path(full_path: str) -> None:
    # intentionally more restrictive, matches a "typical" unix path and filename with extension
    file_regex = r"^[\w\-.\/ ]+\/?\.[\w\-. ]+[^.]$"
    if not full_path or not full_path.strip() or not re.match(file_regex, full_path):
        raise ValueError(f"file path must match regex: {file_regex}")


def _create_file_display_name(source: str, full_path: str) -> str:
    _validate_source(source)
    _validate_file_path(full_path)
    return f"{source}:{full_path}"


class AgentDocuments(Resource):
    def post(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        # Check if the post request has the file part
        if "file" not in request.files:
            return {"message": "No file part"}, 400

        files = request.files.getlist("file")
        source = request.form.get("source", "default")

        file_data = []
        for file in files:
            full_path = file.filename
            try:
                file_display_name = _create_file_display_name(source, full_path)
            except ValueError as err:
                return {"message": str(err)}, 400

            if not any(
                [
                    file_display_name.endswith(filetype)
                    for filetype in [".txt", ".pdf", ".md", ".rst", ".html"]
                ]
            ):
                return {"message": "Unsupported file type uploaded"}, 400

            file_content = file.stream.read()

            file_data.append([file_display_name, full_path, file_content])

        def generate_progress():
            for file_display_name, full_path, file_content in file_data:
                yield json.dumps({"file": full_path, "step": "start"}) + "\n"
                extracted_text = text_extractor(full_path, file_content)
                yield json.dumps({"file": full_path, "step": "text_extracted"}) + "\n"

                # Only generate embeddings when there is actual texts
                if len(extracted_text) > 0:
                    vector_db.add_document(extracted_text, agent.id, source, full_path)
                    agent.add_files([file_display_name])
                    yield json.dumps({"file": full_path, "step": "embedding_created"}) + "\n"

                yield json.dumps({"file": full_path, "step": "end"}) + "\n"

        return Response(generate_progress(), mimetype="application/json")

    def _delete_from_agent(self, agent: Agent, metadatas: List[dict]) -> List[str]:
        files_to_delete = {
            _create_file_display_name(metadata["source"], metadata["full_path"])
            for metadata in metadatas
        }

        agent.delete_files(files_to_delete)

        return list(files_to_delete)

    def delete(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        source = request.json.get("source")
        full_path = request.json.get("full_path")
        delete_all = bool(request.json.get("all", False))

        metadata = {}

        try:
            if full_path:
                _validate_file_path(full_path)
                metadata["full_path"] = full_path
            if source:
                _validate_source(source)
                metadata["source"] = source
        except ValueError as err:
            return {"message": err}, 400

        if not metadata and not delete_all:
            return {"message": "'source' or 'full_path' required when not using 'all'"}, 400

        metadata["agent_id"] = str(agent.id)

        # delete from vector store
        try:
            metadatas = vector_db.delete_documents_by_metadata(metadata)
        except Exception:
            err = "Error deleting document(s) from vector store"
            log.exception(err)
            return {"message": err}, 500

        # delete from agent DB
        try:
            deleted = self._delete_from_agent(agent, metadatas)
        except Exception:
            err = "Error deleting document(s) from agent DB"
            log.exception(err)
            return {"message": err}, 500

        count = len(deleted)
        return {"message": f"{count} document(s) deleted", "count": count, "deleted": deleted}, 200


class AgentChatApi(Resource):
    def post(self, id):
        agent = Agent.get(id)
        if not agent:
            return {"message": "Agent not found"}, 404

        query = request.json.get("query")
        stream = request.json.get("stream") == "true"
        previous_messages = request.json.get("prevMsgs")

        llm_response = llm.ask(agent.system_prompt, previous_messages, query, agent.id, stream)

        if stream:
            return Response(llm_response(), mimetype="application/json")

        return llm_response, 200
