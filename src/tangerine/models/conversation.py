import uuid

from sqlalchemy.dialects.postgresql import UUID

from tangerine.db import db


class Conversation(db.Model):
    __tablename__ = "conversations"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = db.Column(db.String(256), nullable=True)
    session_id = db.Column(UUID(as_uuid=True), nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.current_timestamp())
    updated_at = db.Column(
        db.DateTime, default=db.func.current_timestamp(), onupdate=db.func.current_timestamp()
    )
    payload = db.Column(db.JSON, nullable=False)
    title = db.Column(db.String(256), nullable=True)

    @classmethod
    def get_by_session(cls, session_id):
        """
        Retrieve a conversation by user ID and session ID.
        """
        return cls.query.filter_by(session_id=session_id).first()

    @classmethod
    def get_by_user(cls, user_id):
        """
        Retrieve all conversations for a specific user ID.
        """
        return cls.query.filter_by(user_id=user_id).all()

    @classmethod
    def upsert(cls, conversation_json):
        user_id = conversation_json.get("user")
        session_id = conversation_json.get("sessionId")

        # Convert session_id to UUID if it's a string
        if isinstance(session_id, str):
            try:
                session_id = uuid.UUID(session_id)
            except ValueError:
                # If it's not a valid UUID, generate a new one
                session_id = uuid.uuid4()

        conversation = cls.query.filter_by(session_id=session_id).first()

        if conversation and conversation.is_owned_by(user_id):
            # Update existing conversation
            conversation.updated_at = db.func.current_timestamp()
            conversation.payload = conversation_json
        elif conversation and not conversation.is_owned_by(user_id):
            # If the conversation exists but is owned by a different user, create a new one
            # that is owned by the user and has a new session ID
            conversation = cls()
            conversation.user_id = user_id
            conversation.session_id = uuid.uuid4()
            conversation.payload = conversation_json
            conversation.title = cls.generate_title(conversation_json)
            db.session.add(conversation)
        else:
            # Create a new conversation
            conversation = cls()
            conversation.user_id = user_id
            conversation.session_id = session_id
            conversation.payload = conversation_json
            conversation.title = cls.generate_title(conversation_json)
            db.session.add(conversation)

        db.session.commit()
        return conversation

    @classmethod
    def generate_title(cls, conversation_json):
        """
        Generate a title for the conversation based on the user's query.
        This is a placeholder for actual LLM integration.
        """
        user_query = conversation_json.get("query", "")
        if user_query:
            # Simple title generation logic, can be replaced with LLM call
            return f"{user_query[:50]}..."
        return "Untitled Conversation"

    @classmethod
    def from_json(cls, conversation_json):
        """
        Create a Conversation object from a JSON serializable dictionary.
        """
        session_id = conversation_json.get("sessionId")
        if isinstance(session_id, str):
            session_id = uuid.UUID(session_id)

        conversation = cls()
        conversation.user_id = conversation_json.get("user")
        conversation.session_id = session_id
        conversation.payload = conversation_json
        conversation.title = cls.generate_title(conversation_json)
        return conversation

    def copy(self):
        """
        Create a copy of the conversation object.
        """
        new_conversation = Conversation()
        new_conversation.id = self.id
        new_conversation.user_id = self.user_id
        new_conversation.session_id = self.session_id
        new_conversation.created_at = self.created_at
        new_conversation.updated_at = self.updated_at
        new_conversation.payload = self.payload.copy() if self.payload else None
        new_conversation.title = self.title
        return new_conversation

    def is_owned_by(self, user_id):
        """
        Check if the conversation is owned by the given user ID.
        """
        return self.user_id == user_id

    def delete(self):
        """
        Delete the conversation from the database.
        """
        db.session.delete(self)
        db.session.commit()

    @classmethod
    def delete_by_session(cls, session_id, user_id):
        """
        Delete a conversation by session ID, with ownership validation.
        Returns True if deleted, False if not found or not owned by user.
        """
        conversation = cls.query.filter_by(session_id=session_id).first()

        if not conversation:
            return False, "Conversation not found"

        if not conversation.is_owned_by(user_id):
            return False, "Unauthorized: You can only delete your own conversations"

        conversation.delete()
        return True, "Conversation deleted successfully"

    def to_json(self):
        """
        Convert the conversation object to a JSON serializable dictionary.
        """
        return {
            "id": str(self.id),
            "user_id": self.user_id,
            "session_id": str(self.session_id),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "payload": self.payload,
            "title": self.title,
        }
