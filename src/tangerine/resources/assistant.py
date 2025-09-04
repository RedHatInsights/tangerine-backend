import hashlib
import json
import logging
import uuid

from flask import Response, request, stream_with_context
from flask_restful import Resource
from langchain_core.documents import Document
from sqlalchemy.exc import SQLAlchemyError

import tangerine.llm as llm
from tangerine import config
from tangerine.config import DEFAULT_SYSTEM_PROMPT
from tangerine.embeddings import embed_query
from tangerine.metrics import get_counter
from tangerine.models.assistant import Assistant
from tangerine.models.conversation import Conversation
from tangerine.models.interactions import store_interaction
from tangerine.models.knowledgebase import KnowledgeBase
from tangerine.search import SearchResult, search_engine

log = logging.getLogger("tangerine.resources")


def _get_search_results_for_assistant(assistant_id, query, embedding):
    """Helper function to get search results for an assistant by querying its knowledgebases."""
    assistant = Assistant.get(assistant_id)
    if not assistant:
        return []
    knowledgebase_ids = assistant.get_knowledgebase_ids()
    log.debug(
        "search request received for assistant '%s', found kb's: %s",
        assistant.name,
        knowledgebase_ids,
    )
    if not knowledgebase_ids:
        return []
    return search_engine.search(knowledgebase_ids, query, embedding)


# Prometheus metrics
user_interaction_counter = get_counter(
    "user_interaction_counter",
    "Total number of user interactions with assistants",
    ["user", "client", "assistant_id", "assistant_name"],
)


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
        if not request.json:
            return {"message": "No JSON data provided"}, 400

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
        return {"message": "assistant deleted successfully"}, 200


class AssistantChatApi(Resource):
    @staticmethod
    def _is_streaming_response(stream):
        return bool(stream)

    @staticmethod
    def _anonymize_user_id(user_id):
        """
        Anonymize user ID using SHA256 hash, unless the user ID is 'unknown'.

        Args:
            user_id (str): The user ID to anonymize

        Returns:
            str: The anonymized user ID or 'unknown' if the input was 'unknown'
        """
        if user_id == "unknown":
            return user_id

        # Create a hash of the user ID for anonymization
        return hashlib.sha256(user_id.encode("utf-8")).hexdigest()[
            :16
        ]  # Use first 16 chars of hash

    def post(self, id):
        assistant = self._get_assistant(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        log.info("AUDIT: querying vector DB")
        (
            question,
            session_uuid,
            stream,
            previous_messages,
            interaction_id,
            client,
            user,
            current_message,
        ) = self._extract_request_data()

        # Record user interaction metrics
        anonymized_user = self._anonymize_user_id(user)
        user_interaction_counter.labels(
            user=anonymized_user,
            client=client,
            assistant_id=assistant.id,
            assistant_name=assistant.name,
        ).inc()
        embedding = self._embed_question(question)
        search_results = self._get_search_results(assistant.id, question, embedding)
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
                user,
                previous_messages,
                assistant.name,
                current_message,
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
            user,
            previous_messages,
            assistant.name,
            current_message,
        )

    def _get_assistant(self, assistant_id):
        return Assistant.get(assistant_id)

    def _extract_request_data(self):
        if not request.json:
            raise ValueError("No JSON data provided")
        question = request.json.get("query")
        session_uuid = request.json.get("sessionId", str(uuid.uuid4()))
        stream = request.json.get("stream", "true") == "true"
        previous_messages = request.json.get("prevMsgs")
        interaction_id = request.json.get("interactionId", None)
        client = request.json.get("client", "unknown")
        user = request.json.get("user", "unknown")

        # Extract the current message data to preserve all fields
        current_message = request.json.get("currentMessage", {})
        # If no currentMessage is provided, create it from available fields
        if not current_message:
            current_message = {"sender": "human", "text": question}
            # Preserve any additional fields that might be in the root request
            for field in ["isIntroductionPrompt"]:
                if field in request.json:
                    current_message[field] = request.json[field]

        return (
            question,
            session_uuid,
            stream,
            previous_messages,
            interaction_id,
            client,
            user,
            current_message,
        )

    def _embed_question(self, question):
        return embed_query(question)

    def _call_llm(self, assistant, previous_messages, question, search_results, interaction_id):
        return llm.ask(
            [assistant],
            previous_messages,
            question,
            search_results,
            interaction_id=interaction_id,
            prompt=None,  # No override for basic API, use assistant config
        )

    @staticmethod
    def _parse_search_results(search_results: list[SearchResult]) -> list[dict]:
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
        return _get_search_results_for_assistant(assistant_id, query, embedding)

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
        user,
        previous_messages=None,
        assistant_name=None,
        current_message=None,
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
                user,
            )

            # Update conversation history with both user query and assistant response
            self._update_conversation_history(
                question,
                accumulated_text,
                session_uuid,
                previous_messages,
                user,
                assistant_name,
                current_message,
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
        user,
        previous_messages=None,
        assistant_name=None,
        current_message=None,
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
            user,
        )

        # Update conversation history with both user query and assistant response
        self._update_conversation_history(
            question,
            response["text_content"],
            session_uuid,
            previous_messages,
            user,
            assistant_name,
            current_message,
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
        user,
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
                user=user,
            )
        except Exception:
            log.exception("Failed to log interaction")

    def _update_conversation_history(
        self,
        question,
        response_text,
        session_uuid,
        previous_messages,
        user,
        assistant_name=None,
        current_message=None,
    ):
        """Update the conversation with the complete conversation history including the latest exchange."""
        try:
            # Validate required parameters
            if not question or not response_text or not session_uuid:
                log.warning("Missing required parameters for conversation history update")
                return

            # Build the updated conversation history
            updated_messages = previous_messages.copy() if previous_messages else []

            # Add the user's question - use the complete current_message to preserve all fields
            if current_message:
                updated_messages.append(current_message)
            else:
                # Fallback to minimal message if current_message not provided
                updated_messages.append({"sender": "human", "text": question})

            # Add the assistant's response
            updated_messages.append({"sender": "ai", "text": response_text})

            # Create the conversation payload
            conversation_payload = {
                "sessionId": session_uuid,
                "user": user,
                "query": question,
                "prevMsgs": updated_messages,
            }

            # Add assistant name if provided
            if assistant_name:
                conversation_payload["assistantName"] = assistant_name

            # Upsert the conversation
            Conversation.upsert(conversation_payload)
            log.info("AUDIT: Successfully updated conversation history for session %s", session_uuid)

        except Exception as e:
            log.exception(
                "Failed to update conversation history for session %s: %s", session_uuid, str(e)
            )


class AssistantAdvancedChatApi(AssistantChatApi):
    """
    API for advanced assistant chat supporting multiple assistants,
    external chunk injection, and model override.
    """

    def _convert_chunk_array_to_search_results(self, chunks):
        """
        Converts an array of chunks into a list of SearchResult objects.
        """
        return [
            SearchResult(document=Document(page_content=chunk, metadata={}), score=1)
            for chunk in chunks
        ]

    def post(self, _id=None):
        if not request.json:
            return {"message": "No JSON data provided"}, 400

        assistant_names = request.json.get("assistants")
        assistants = []

        if not assistant_names:
            return {"message": "assistant name(s) required"}, 400
        try:
            assistants = self._get_assistants(assistant_names)
        except ValueError as err:
            return {"message": str(err)}, 400

        question = request.json.get("query")
        if not question:
            return {"message": "query is required"}, 400
        # Support both 'system_prompt' and 'prompt' parameters for backward compatibility
        # Priority: API override -> Assistant config -> Default
        api_system_prompt = request.json.get("system_prompt") or request.json.get("prompt")
        system_prompt = api_system_prompt  # Will be None if no API override provided
        session_uuid = request.json.get("sessionId", str(uuid.uuid4()))
        stream = request.json.get("stream", "true") == "true"
        previous_messages = request.json.get("prevMsgs")
        interaction_id = request.json.get("interactionId", None)
        client = request.json.get("client", "unknown")
        model_name = request.json.get("model")
        user = request.json.get("user", "unknown")
        disable_agentic = request.json.get("disable_agentic", False)
        
        # AUDIT LOG: Request model parameter
        log.info("AUDIT: Advanced Chat API received model parameter: %s", model_name)
        user_prompt = request.json.get("userPrompt")

        # Extract the current message data to preserve all fields
        current_message = request.json.get("currentMessage", {})
        # If no currentMessage is provided, create it from available fields
        if not current_message:
            current_message = {"sender": "human", "text": question}
            # Preserve any additional fields that might be in the root request
            for field in ["isIntroductionPrompt"]:
                if field in request.json:
                    current_message[field] = request.json[field]

        # AUDIT LOG: Model validation
        log.info("AUDIT: Validating model_name=%s, available_models=%s", model_name, list(config.MODELS.keys()))
        if model_name and model_name not in config.MODELS:
            log.error("AUDIT: INVALID MODEL - model_name=%s not in available models %s", model_name, list(config.MODELS.keys()))
            return {"message": f"Invalid model name: {model_name}"}, 400
        log.info("AUDIT: Model validation passed for model_name=%s", model_name)

        # Record user interaction metrics for each assistant
        anonymized_user = self._anonymize_user_id(user)
        for assistant in assistants:
            user_interaction_counter.labels(
                user=anonymized_user,
                client=client,
                assistant_id=assistant.id,
                assistant_name=assistant.name,
            ).inc()

        embedding = embed_query(question)
        chunks = request.json.get("chunks", None)
        if chunks:
            chunks = self._convert_chunk_array_to_search_results(request.json.get("chunks"))

        # Get all knowledgebase IDs from all assistants
        all_knowledgebase_ids = set()
        for assistant in assistants:
            all_knowledgebase_ids.update(assistant.get_knowledgebase_ids())
        knowledgebase_ids = list(all_knowledgebase_ids)

        search_results = chunks or (
            search_engine.search(knowledgebase_ids, question, embedding)
            if knowledgebase_ids
            else []
        )

        # AUDIT LOG: Calling llm.ask with model parameter
        log.info("AUDIT: Calling llm.ask() with model=%s, disable_agentic=%s", model_name, disable_agentic)
        llm_response, search_metadata = llm.ask(
            assistants,
            previous_messages,
            question,
            search_results,
            interaction_id=interaction_id,
            prompt=system_prompt,
            model=model_name,
            disable_agentic=disable_agentic,
            user_prompt=user_prompt,
        )
        log.info("AUDIT: llm.ask() completed")
        # Create combined assistant name for multiple assistants
        combined_assistant_name = ", ".join([assistant.name for assistant in assistants])

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
                user,
                previous_messages,
                combined_assistant_name,
                current_message,
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
            user,
            previous_messages,
            combined_assistant_name,
            current_message,
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
        if not request.json:
            return {"message": "No JSON data provided"}, 400

        query = request.json.get("query")
        assistant = self._get_assistant(id)
        if not assistant:
            return {"message": "assistant not found"}, 404

        log.info("AUDIT: querying vector DB")

        embedding = self._embed_question(query)
        search_results = self._get_search_results(assistant.id, query, embedding)

        return [result.to_json() for result in search_results], 200

    def _get_assistant(self, assistant_id):
        return Assistant.get(assistant_id)

    def _embed_question(self, question):
        return embed_query(question)

    @staticmethod
    def _get_search_results(assistant_id, query, embedding):
        return _get_search_results_for_assistant(assistant_id, query, embedding)


class AssistantKnowledgeBasesApi(Resource):
    @staticmethod
    def _ensure_kb_ids_exist(knowledgebase_ids):
        """Validate that all KnowledgeBase IDs exist. Returns (knowledgebases, not_found_ids)."""
        knowledgebases = []
        not_found_ids = []

        for kb_id in knowledgebase_ids:
            kb = KnowledgeBase.get(kb_id)
            if not kb:
                not_found_ids.append(kb_id)
            else:
                knowledgebases.append(kb)

        return knowledgebases, not_found_ids

    def get(self, id):
        """Get knowledgebases associated with an assistant."""
        try:
            assistant_id = int(id)
        except ValueError:
            return {"error": "Invalid assistant ID"}, 400

        assistant = Assistant.get(assistant_id)
        if not assistant:
            return {"error": "Assistant not found"}, 404

        knowledgebases = assistant.get_knowledgebases()
        return {"data": [kb.to_dict() for kb in knowledgebases]}

    def post(self, id):
        """Associate knowledgebases with an assistant."""
        try:
            assistant_id = int(id)
        except ValueError:
            return {"error": "Invalid assistant ID"}, 400

        assistant = Assistant.get(assistant_id)
        if not assistant:
            return {"error": "Assistant not found"}, 404

        data = request.get_json()
        if not data or "knowledgebase_ids" not in data:
            return {"error": "knowledgebase_ids array is required in request body"}, 400

        knowledgebase_ids = data["knowledgebase_ids"]
        if not isinstance(knowledgebase_ids, list):
            return {"error": "knowledgebase_ids must be an array"}, 400

        # Step 1: Validate that all KnowledgeBase IDs exist
        knowledgebases, not_found_ids = self._ensure_kb_ids_exist(knowledgebase_ids)
        if not_found_ids:
            return {"error": f"KnowledgeBase IDs not found: {not_found_ids}"}, 404

        # Step 2: Associate all knowledgebases with the assistant
        associated = []
        try:
            for kb in knowledgebases:
                assistant.associate_knowledgebase(kb)
                associated.append(kb.to_dict())
                log.info("associated knowledgebase %d with assistant %d", kb.id, assistant_id)
        except SQLAlchemyError as e:
            log.exception(
                "database error associating knowledgebases with assistant %d", assistant_id
            )
            return {"error": f"Database error: {str(e)}"}, 500

        return {"associated_knowledgebases": associated}, 200

    def delete(self, id):
        """Disassociate knowledgebases from an assistant."""
        try:
            assistant_id = int(id)
        except ValueError:
            return {"error": "Invalid assistant ID"}, 400

        assistant = Assistant.get(assistant_id)
        if not assistant:
            return {"error": "Assistant not found"}, 404

        data = request.get_json()
        if not data or "knowledgebase_ids" not in data:
            return {"error": "knowledgebase_ids array is required in request body"}, 400

        knowledgebase_ids = data["knowledgebase_ids"]
        if not isinstance(knowledgebase_ids, list):
            return {"error": "knowledgebase_ids must be an array"}, 400

        # Step 1: Validate that all KnowledgeBase IDs exist
        knowledgebases, not_found_ids = self._ensure_kb_ids_exist(knowledgebase_ids)
        if not_found_ids:
            return {"error": f"KnowledgeBase IDs not found: {not_found_ids}"}, 404

        # Step 2: Disassociate all knowledgebases from the assistant
        disassociated = []
        try:
            for kb in knowledgebases:
                assistant.disassociate_knowledgebase(kb)
                disassociated.append(kb.to_dict())
                log.info("disassociated knowledgebase %d from assistant %d", kb.id, assistant_id)
        except SQLAlchemyError as e:
            log.exception(
                "database error disassociating knowledgebases from assistant %d", assistant_id
            )
            return {"error": f"Database error: {str(e)}"}, 500

        return {"disassociated_knowledgebases": disassociated}, 200
