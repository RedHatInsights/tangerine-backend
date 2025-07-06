from flask_restful import Resource
from flask import request

from tangerine.models.conversation import Conversation

class ConversationListApi(Resource):
    """
    Get a list of conversations for a specific user_id
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conversation = None

    def post(self):
        """
        Handle POST requests to retrieve a list of conversations.
        """
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        user_id = data.get("user_id")
        if not user_id:
            return {"error": "User ID is required"}, 400

        try:
            conversation_objects = Conversation.get_by_user(user_id)
            conversation_json = [conv.to_json() for conv in conversation_objects]
            return conversation_json, 200
        except Exception as e:
            return {"error": str(e)}, 500


class ConversationRetrievalApi(Resource):
    """
    Get a specific conversation by ID
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._conversation = None

    def post(self):
        """
        Handle POST requests to retrieve a specific conversation by ID.
        """
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        session_id = data.get("sessionId")
        if not session_id:
            return {"error": "Session ID is required"}, 400

        try:
            conversation = Conversation.get_by_session(session_id)
            if not conversation:
                return {"error": "Conversation not found"}, 404
            return conversation.to_json(), 200
        except Exception as e:
            return {"error": str(e)}, 500


class ConversationUpsertApi(Resource):
    """
    Upsert a conversation
    """

    def post(self):
        """
        Handle POST requests to upsert a conversation.
        """
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        try:
            conversation = Conversation.upsert(data)
            return conversation.to_json(), 200
        except Exception as e:
            return {"error": str(e)}, 500


class ConversationDeleteApi(Resource):
    """
    Delete a conversation
    """

    def post(self):
        """
        Handle POST requests to delete a conversation.
        """
        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        session_id = data.get("sessionId")
        user_id = data.get("user_id")

        if not session_id:
            return {"error": "Session ID is required"}, 400
        if not user_id:
            return {"error": "User ID is required"}, 400

        try:
            success, message = Conversation.delete_by_session(session_id, user_id)
            if success:
                return {"message": message}, 200
            else:
                return {"error": message}, 400 if "not found" in message.lower() else 403
        except Exception as e:
            return {"error": str(e)}, 500
