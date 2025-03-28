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
    def post(self, id):
        assistant = self._get_assistant(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        log.debug("querying vector DB")
        question, session_uuid, stream, previous_messages, interaction_id, client = (
            self._extract_request_data()
        )
        embedding = self._embed_question(question)
        search_results = self._get_search_results(assistant.id, question, embedding)
        llm_response = self._call_llm(
            assistant, previous_messages, question, search_results, stream, interaction_id
        )

        if self._is_streaming_response(llm_response, stream):
            return self._handle_streaming_response(
                llm_response,
                question,
                embedding,
                search_results,
                session_uuid,
                interaction_id,
                client,
            )

        return self._handle_final_response(
            llm_response,
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

    def _call_llm(
        self, assistant, previous_messages, question, search_results, stream, interaction_id
    ):
        return llm.ask(
            assistant,
            previous_messages,
            question,
            search_results,
            stream=stream,
            interaction_id=interaction_id,
        )

    @staticmethod
    def _is_streaming_response(llm_response, stream):
        return stream and (callable(llm_response) or hasattr(llm_response, "__iter__"))

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
        question,
        embedding,
        search_results,
        session_uuid,
        interaction_id,
        client,
    ):
        source_doc_info = self._parse_search_results(search_results)

        def accumulate_and_stream():
            accumulated_response = ""
            for raw_chunk in llm_response():
                text_content = self._extract_text_from_chunk(raw_chunk)
                accumulated_response += text_content
                yield raw_chunk
            self._log_interaction(
                question,
                accumulated_response,
                source_doc_info,
                embedding,
                session_uuid,
                interaction_id,
                client,
            )

        return Response(stream_with_context(accumulate_and_stream()))

    def _handle_final_response(
        self,
        llm_response,
        question,
        search_results,
        embedding,
        session_uuid,
        interaction_id,
        client,
    ):
        source_doc_info = self._parse_search_results(search_results)

        self._log_interaction(
            question,
            llm_response,
            source_doc_info,
            embedding,
            session_uuid,
            interaction_id,
            client,
        )
        return {"response": llm_response}, 200

    def _extract_text_from_chunk(self, raw_chunk):
        try:
            # the raw_chunk is a string that looks like this:
            # data: {"text_content": "Hello, how can I help you today?"}\r\n
            # we need to extract the text_content from it for logging interactions
            _, data = raw_chunk.split("data:")
            return json.loads(data.strip()).get("text_content", "")
        except Exception:
            log.exception("error extracting text_content from chunk")
            return ""

    # Looks like a silly function but it makes it easier to mock in tests
    def _interaction_storage_enabled(self) -> bool:
        return config.STORE_INTERACTIONS is True

    def _log_interaction(
        self, question, response, source_doc_info, embedding, session_uuid, interaction_id, client
    ):
        if self._interaction_storage_enabled() is False:
            return
        try:
            store_interaction(
                question=question,
                llm_response=response,
                source_doc_chunks=source_doc_info,
                question_embedding=embedding,
                session_uuid=session_uuid,
                interaction_id=interaction_id,
                client=client,
            )
        except Exception:
            log.exception("Failed to log interaction")
