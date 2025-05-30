import json
import logging
import uuid

from flask import Response, request, stream_with_context
from flask_restful import Resource
from langchain_core.documents import Document

import tangerine.llm as llm
from tangerine import config
from tangerine.config import DEFAULT_SYSTEM_PROMPT
from tangerine.embeddings import embed_query
from tangerine.models.assistant import Assistant
from tangerine.models.interactions import store_interaction
from tangerine.search import search_engine
from tangerine.utils import File, add_filenames_to_assistant, embed_files, remove_files
from tangerine.vector import vector_db

log = logging.getLogger("tangerine.resources")

MODELS = {
    "default": {
        "base_url": config.LLM_BASE_URL,
        "name": config.LLM_MODEL_NAME,
        "api_key": config.LLM_API_KEY,
        "temperature": config.LLM_TEMPERATURE,
    }
}

DEFAULT_MODEL = MODELS["default"]


class AssistantDefaultsApi(Resource):
    def get(self):
        return {"system_prompt": DEFAULT_SYSTEM_PROMPT}, 200


class AssistantsApi(Resource):
    def get(self):
        try:
            all_assistants = Assistant.list()
        except Exception:
            log.exception("error getting assistants")
            return {"message": "error getting assistants"}, 500

        return {"data": [assistant.to_dict() for assistant in all_assistants]}, 200

    def post(self):
        name = request.json.get("name")
        description = request.json.get("description")
        if not name:
            return {"message": "assistant 'name' required"}, 400
        if not description:
            return {"message": "assistant 'description' required"}, 400

        try:
            assistant = Assistant.create(name, description, request.json.get("system_prompt"))
        except Exception:
            log.exception("error creating assistant")
            return {"message": "error creating assistant"}, 500

        return assistant.to_dict(), 201


class AssistantApi(Resource):
    def get(self, id):
        assistant = Assistant.get(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        return assistant.to_dict(), 200

    def put(self, id):
        assistant = Assistant.get(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        data = request.get_json()
        # ignore 'id' or 'filenames' if provided in JSON payload
        data.pop("filenames", None)
        data.pop("id", None)
        assistant.update(**data)

        return {"message": "assistant updated successfully"}, 200

    def delete(self, id):
        assistant = Assistant.get(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        assistant.delete()
        vector_db.delete_document_chunks({"assistant_id": assistant.id})
        return {"message": "assistant deleted successfully"}, 200


class AssistantDocuments(Resource):
    def post(self, id):
        assistant = Assistant.get(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

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
                embed_files([file], assistant)
                add_filenames_to_assistant([file], assistant)
                yield json.dumps({"file": file.display_name, "step": "end"}) + "\n"

        return Response(stream_with_context(generate_progress()), mimetype="application/json")

    def delete(self, id):
        assistant = Assistant.get(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

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
            deleted = remove_files(assistant, metadata)
        except ValueError as err:
            return {"message": str(err)}, 400
        except Exception:
            err = "unexpected error deleting document(s) from DB"
            log.exception(err)
            return {"message": err}, 500

        count = len(deleted)
        return {"message": f"{count} document(s) deleted", "count": count, "deleted": deleted}, 200


        
        
class AssistantChatApi(Resource):
    @staticmethod
    def _is_streaming_response(stream):
        return bool(stream)

    def post(self, id):
        assistant = self._get_assistant(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        log.debug("querying vector DB")
        question, session_uuid, stream, previous_messages, interaction_id, client = (
            self._extract_request_data()
        )
        embedding = self._embed_question(question)
        search_results = self._get_search_results([assistant.id], question, embedding)
        llm_response, search_metadata = self._call_llm(
            assistant, previous_messages, question, search_results, interaction_id
        )

        if self._is_streaming_response(stream):
            return self._handle_streaming_response(
                llm_response,
                search_metadata,
                question,
                embedding,
                search_results,
                session_uuid,
                interaction_id,
                client,
            )

        return self._handle_standard_response(
            llm_response,
            search_metadata,
            question,
            embedding,
            search_results,
            session_uuid,
            interaction_id,
            client,
        )

    def _get_assistant(self, assistant_id):
        return Assistant.get(assistant_id)

    def _extract_request_data(self):
        question = request.json.get("query")
        session_uuid = request.json.get("sessionId", str(uuid.uuid4()))
        stream = request.json.get("stream", "true") == "true"
        previous_messages = request.json.get("prevMsgs")
        interaction_id = request.json.get("interactionId", None)
        client = request.json.get("client", "unknown")
        return question, session_uuid, stream, previous_messages, interaction_id, client

    def _embed_question(self, question):
        return embed_query(question)

    def _call_llm(self, assistant, previous_messages, question, search_results, interaction_id):
        return llm.ask(
            assistant,
            previous_messages,
            question,
            search_results,
            interaction_id=interaction_id,
        )

    @staticmethod
    def _parse_search_results(search_results: list[Document]) -> list[dict]:
        return [
            {
                "text": doc.document.page_content,
                "source": doc.document.metadata.get("source"),
                "score": doc.document.metadata.get("relevance_score"),
                "retrieval_method": doc.document.metadata.get("retrieval_method"),
            }
            for doc in search_results
        ]

    @staticmethod
    def _get_search_results(assistant_id, query, embedding):
        return search_engine.search(assistant_id, query, embedding)

    def _handle_streaming_response(
        self,
        llm_response,
        search_metadata,
        question,
        embedding,
        search_results,
        session_uuid,
        interaction_id,
        client,
    ):
        source_doc_info = self._parse_search_results(search_results)

        # TODO: change the way we stream to something more standardized...
        def __api_response_generator():
            accumulated_text = ""

            for text in llm_response:
                accumulated_text += text
                chunk = {"text_content": text}
                yield f"data: {json.dumps(chunk)}\r\n"

            # final piece of content returned is the search metadata
            yield f"data: {json.dumps({'search_metadata': search_metadata})}\r\n"

            # log user interaction at the end
            self._log_interaction(
                question,
                accumulated_text,
                source_doc_info,
                embedding,
                session_uuid,
                interaction_id,
                client,
            )

        return Response(stream_with_context(__api_response_generator()))

    def _handle_standard_response(
        self,
        llm_response,
        search_metadata,
        question,
        embedding,
        search_results,
        session_uuid,
        interaction_id,
        client,
    ):
        source_doc_info = self._parse_search_results(search_results)

        response = {"text_content": "".join(llm_response), "search_metadata": search_metadata}

        self._log_interaction(
            question,
            response["text_content"],
            source_doc_info,
            embedding,
            session_uuid,
            interaction_id,
            client,
        )
        return response, 200

    # Looks like a silly function but it makes it easier to mock in tests
    def _interaction_storage_enabled(self) -> bool:
        return config.STORE_INTERACTIONS is True

    def _log_interaction(
        self,
        question,
        response_text,
        source_doc_info,
        embedding,
        session_uuid,
        interaction_id,
        client,
    ):
        if self._interaction_storage_enabled() is False:
            return
        try:
            store_interaction(
                question=question,
                llm_response=response_text,
                source_doc_chunks=source_doc_info,
                question_embedding=embedding,
                session_uuid=session_uuid,
                interaction_id=interaction_id,
                client=client,
            )
        except Exception:
            log.exception("Failed to log interaction")


class AssistantAdvancedChatApi(AssistantChatApi):

    def post(self, _id=None):
        assistant_names = request.json.get("assistants")
        assistants = []
        
        if not assistant_names:
            return {"message": "assistant name(s) required"}, 400
        try:
            assistants = self._get_assistants(assistant_names)
        except ValueError as err:
            return {"message": str(err)}, 400
        
        assistant_ids = [assistant.id for assistant in assistants]
        question = request.json.get("query")
        if not question:
            return {"message": "query is required"}, 400
        system_prompt = request.json.get("system_prompt", DEFAULT_SYSTEM_PROMPT)
        session_uuid = request.json.get("sessionId", str(uuid.uuid4()))
        stream = request.json.get("stream", "true") == "true"
        previous_messages = request.json.get("prevMsgs")
        interaction_id = request.json.get("interactionId", None)
        client = request.json.get("client", "unknown")
        model = MODELS.get(request.json.get("model", "default"), DEFAULT_MODEL)
        embedding = embed_query(question)
        search_results = search_engine.search(assistant_ids, question, embedding)
        llm_response, search_metadata = llm.ask_advanced(
            assistants,
            previous_messages,
            question,
            search_results,
            interaction_id=interaction_id,
            prompt=system_prompt,
            model=model,
        )
        if self._is_streaming_response(stream):
            return self._handle_streaming_response(
                llm_response,
                search_metadata,
                question,
                embedding,
                search_results,
                session_uuid,
                interaction_id,
                client,
            )

        return self._handle_standard_response(
            llm_response,
            search_metadata,
            question,
            embedding,
            search_results,
            session_uuid,
            interaction_id,
            client,
        )
    def _get_assistants(self, assistant_names):
        assistants = []
        for name in assistant_names:
            assistant = Assistant.get_by_name(name)
            if not assistant:
                raise ValueError(f"Assistant '{name}' not found")
            assistants.append(assistant)
        return assistants

    def _get_assistant_ids(self, assistants):
        return [assistant.id for assistant in assistants]


class AssistantSearchApi(Resource):
    def post(self, id):
        query = request.json.get("query")
        assistant = self._get_assistant(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        log.debug("querying vector DB")

        embedding = self._embed_question(query)
        search_results = self._get_search_results(assistant.id, query, embedding)

        return [result.to_json() for result in search_results], 200

    def _get_assistant(self, assistant_id):
        return Assistant.get(assistant_id)

    def _embed_question(self, question):
        return embed_query(question)

    @staticmethod
    def _get_search_results(assistant_id, query, embedding):
        return search_engine.search(assistant_id, query, embedding)
