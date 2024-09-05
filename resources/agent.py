import json
import logging
import re
from typing import List

from flask import Response, request
from flask_restful import Resource

from connectors.llm.interface import DEFAULT_SYSTEM_PROMPT, llm
from connectors.vector_store.db import Agents, db, vector_interface
from utils.processors import text_extractor

log = logging.getLogger("tangerine.agent")


class AgentDefaultsApi(Resource):
    def get(self):
        return {"system_prompt": DEFAULT_SYSTEM_PROMPT}, 200


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
            "system_prompt": request.form.get("system_prompt") or DEFAULT_SYSTEM_PROMPT,
        }

        # Don't let them create the agent id and filenames
        agent.pop("id", None)
        agent.pop("filenames", None)

        try:
            new_agent = Agents(**agent)
            db.session.add(new_agent)
            db.session.commit()
            db.session.refresh(new_agent)
        except Exception:
            log.exception("exception in AgentsApi POST")
            return {"message": "error inserting agent into DB"}, 500

        agent["id"] = new_agent.id

        return agent, 201


class AgentApi(Resource):
    def get(self, id):
        agent_id = int(id)
        agent = Agents.query.filter_by(id=agent_id).first()

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
        agent_id = int(id)
        agent = Agents.query.get(agent_id)
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
        agent_id = int(id)
        agent = Agents.query.get(agent_id)
        if agent:
            db.session.delete(agent)
            db.session.commit()
            vector_interface.delete_documents_by_metadata({"agent_id": str(agent_id)})
            return {"message": "Agent deleted successfully"}, 200
        else:
            return {"message": "Agent not found"}, 404


def _validate_source(source: str) -> None:
    source_regex = r"^[\w-]+$"
    if not source or not source.strip() or not re.match(source_regex, source):
        raise ValueError(f"source must match regex: {source_regex}")


def _validate_file_path(full_path: str) -> None:
    # intentionally more restrictive, matches a "typical" unix path and filename with extension
    file_regex = r"^[\w\-.\/ ]+\/?\.[\w\-. ]+[^.]$"
    if not full_path or not full_path.strip() or not re.match(file_regex, full_path):
        raise ValueError(f"file path must match regex: {file_regex}")


def _create_file_id(source: str, full_path: str) -> str:
    _validate_source(source)
    _validate_file_path(full_path)
    return f"{source}:{full_path}"


class AgentDocuments(Resource):
    def post(self, id):
        agent_id = int(id)
        agent = Agents.query.get(agent_id)
        if not agent:
            return {"message": "Agent not found"}, 404

        # Check if the post request has the file part
        if "file" not in request.files:
            return {"message": "No file part"}, 400

        files = request.files.getlist("file")
        source = request.form.get("source", "default")

        file_contents = []
        for file in files:
            full_path = file.filename
            try:
                file_id = _create_file_id(source, full_path)
            except ValueError as err:
                return {"message": str(err)}, 400

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
                    vector_interface.add_document(extracted_text, agent_id, source, full_path)
                    yield json.dumps({"file": full_path, "step": "embedding_created"}) + "\n"

                yield json.dumps({"file": full_path, "step": "end"}) + "\n"

        return Response(generate_progress(), mimetype="application/json")

    def _delete_from_agent(self, id: int, metadatas: List[dict]) -> List[str]:
        deleted_files = {
            _create_file_id(metadata["source"], metadata["full_path"]) for metadata in metadatas
        }

        agent = Agents.query.get(id)
        new_full_paths = [file for file in agent.filenames.copy() if file not in deleted_files]
        agent.filenames = new_full_paths
        db.session.commit()
        return list(deleted_files)

    def delete(self, id):
        agent_id = int(id)
        agent = Agents.query.get(agent_id)
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

        metadata["agent_id"] = str(agent_id)

        # delete from vector store
        try:
            metadatas = vector_interface.delete_documents_by_metadata(metadata)
        except Exception:
            err = "Error deleting document(s) from vector store"
            log.exception(err)
            return {"message": err}, 500

        # delete from agent DB
        try:
            deleted = self._delete_from_agent(agent_id, metadatas)
        except Exception:
            err = "Error deleting document(s) from agent DB"
            log.exception(err)
            return {"message": err}, 500

        count = len(deleted)
        return {"message": f"{count} document(s) deleted", "count": count, "deleted": deleted}, 200


class AgentChatApi(Resource):
    def post(self, id):
        agent_id = int(id)
        agent = Agents.query.filter_by(id=agent_id).first()
        if not agent:
            return {"message": "Agent not found"}, 404

        query = request.json.get("query")
        stream = request.json.get("stream") == "true"
        previous_messages = request.json.get("prevMsgs")

        llm_response = llm.ask(agent.system_prompt, previous_messages, query, agent.id, stream)

        if stream:
            return Response(llm_response(), mimetype="application/json")

        return llm_response, 200
