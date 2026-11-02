from flask import request
from flask_restful import Resource

import tangerine.config as cfg
from tangerine.models.conversation import Conversation


def _get_authenticated_user():
    """
    Return the acting principal as asserted by the fronting auth proxy via the
    USER_IDENTITY_HEADER (default: X-Forwarded-User), or None if absent.

    Identity is intentionally NOT taken from the JSON request body: body fields are
    fully attacker-controlled and using them for object-level authorization allows any
    caller to read/modify/delete another user's conversations (CWE-639).
    """
    user = request.headers.get(cfg.USER_IDENTITY_HEADER)
    if user:
        user = user.strip()
    return user or None


_UNAUTHENTICATED = (
    {"error": "Unauthenticated: missing identity header"},
    401,
)


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
        user_id = _get_authenticated_user()
        if not user_id:
            return _UNAUTHENTICATED

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
        user_id = _get_authenticated_user()
        if not user_id:
            return _UNAUTHENTICATED

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
            if not conversation.is_owned_by(user_id):
                return {"error": "Unauthorized: You can only access your own conversations"}, 403
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
        user_id = _get_authenticated_user()
        if not user_id:
            return _UNAUTHENTICATED

        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        # Bind the upsert to the authenticated principal regardless of what the
        # client placed in the body, so Conversation.upsert's ownership check keys
        # off a trusted identity rather than attacker-controlled input.
        data = dict(data)
        data["user"] = user_id

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
        user_id = _get_authenticated_user()
        if not user_id:
            return _UNAUTHENTICATED

        data = request.get_json()
        if not data:
            return {"error": "No data provided"}, 400

        session_id = data.get("sessionId")

        if not session_id:
            return {"error": "Session ID is required"}, 400

        try:
            success, message = Conversation.delete_by_session(session_id, user_id)
            if success:
                return {"message": message}, 200
            else:
                return {"error": message}, 400 if "not found" in message.lower() else 403
        except Exception as e:
            return {"error": str(e)}, 500
