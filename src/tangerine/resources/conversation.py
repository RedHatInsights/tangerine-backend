from flask import Response, request, stream_with_context
from flask_restful import Resource
from tangerine.models.interactions import Interaction


# I think sessions are going to need to be raised to the level of their own table and class
# Right now we can get distinct session UUIDs from the interactions table
# But we have no way to associate session specific data with a session such as its title, description, etc.
# whenever we store an interaction we will try to store a session in a new session table
# if it fails due to a duplicate session UUID, we will just ignore it
# when we create a new session we will take the user's query and as the LLM to generate a title and description for the session


class ConverationListApi(Resource):
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
            conversations = Interaction.get_user_sessions(user_id)
            return Response(conversations, mimetype='application/json')
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

        user_id = data.get("user_id")
        if not user_id:
            return {"error": "User ID is required"}, 400
    
        session_id = data.get("session_id")
        if not session_id:
            return {"error": "Session ID is required"}, 400

        try:
            interactions = Interaction.get_session_interactions(user_id, session_id)
            conversation = self.construct_conversation(interactions)
            return Response(conversation, mimetype='application/json')
        except Exception as e:
            return {"error": str(e)}, 500
        
    def construct_conversation(self, interactions):
        """
        Construct a conversation from the interactions.
        """
        conversation = []
        for interaction in interactions:
            human_entry = {
                "text": interaction.question,
                "sender": "human",
                "done": True,
                "interaction_id": interaction.id,
            }
            conversation.append(human_entry)
            ai_entry = {
                "text": interaction.llm_response,
                "sender": "ai",
                "done": True,
                "search_metadata": self.construct_search_metadata(interaction.source_doc_chunks),
            }
            conversation.append(ai_entry)
        return conversation
    
    def construct_search_metadata(self, source_doc_chunks):
        """
        Construct search metadata from the interactions.
        """
        search_metadata = []
        if source_doc_chunks:
            for chunk in source_doc_chunks:
                metadata = {
                    "page_content": chunk.text,
                    "metadata": {
                        "relevance_score": chunk.score,
                        
                    }
                }
                search_metadata.append(metadata)
        return search_metadata