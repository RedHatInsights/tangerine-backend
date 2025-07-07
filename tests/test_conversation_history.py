"""Simple tests for conversation history functionality.

Focus on testing business logic only, not infrastructure like Flask or database operations.
"""

import uuid
from unittest.mock import Mock, patch

import pytest

from tangerine.resources.assistant import AssistantChatApi


class TestConversationHistoryLogic:
    """Test the core conversation history logic."""

    def test_update_conversation_history_builds_correct_payload(self):
        """Test that conversation history payload is built correctly."""
        api = AssistantChatApi()

        # Mock the Conversation.upsert to capture what gets passed to it
        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            question = "What is AI?"
            response_text = "AI is artificial intelligence."
            session_uuid = "test-session-123"
            previous_messages = [
                {"sender": "human", "text": "Hello"},
                {"sender": "ai", "text": "Hi there!"},
            ]
            user = "test_user"

            api._update_conversation_history(
                question, response_text, session_uuid, previous_messages, user
            )

            # Verify the payload passed to upsert
            call_args = mock_conv.upsert.call_args[0][0]

            assert call_args["sessionId"] == session_uuid
            assert call_args["user"] == user
            assert call_args["query"] == question
            assert len(call_args["prevMsgs"]) == 4
            assert call_args["prevMsgs"][0] == {"sender": "human", "text": "Hello"}
            assert call_args["prevMsgs"][1] == {"sender": "ai", "text": "Hi there!"}
            assert call_args["prevMsgs"][2] == {"sender": "human", "text": question}
            assert call_args["prevMsgs"][3] == {"sender": "ai", "text": response_text}

    def test_update_conversation_history_with_empty_previous_messages(self):
        """Test conversation history with no previous messages."""
        api = AssistantChatApi()

        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            question = "Hello"
            response_text = "Hi there!"
            session_uuid = "test-session-456"
            previous_messages = []
            user = "test_user"

            api._update_conversation_history(
                question, response_text, session_uuid, previous_messages, user
            )

            call_args = mock_conv.upsert.call_args[0][0]

            assert len(call_args["prevMsgs"]) == 2
            assert call_args["prevMsgs"][0] == {"sender": "human", "text": question}
            assert call_args["prevMsgs"][1] == {"sender": "ai", "text": response_text}

    def test_update_conversation_history_with_none_previous_messages(self):
        """Test conversation history with None previous messages."""
        api = AssistantChatApi()

        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            api._update_conversation_history("Hello", "Hi!", "session-123", None, "user")

            call_args = mock_conv.upsert.call_args[0][0]
            assert len(call_args["prevMsgs"]) == 2

    def test_update_conversation_history_skips_invalid_inputs(self):
        """Test that invalid inputs are handled gracefully."""
        api = AssistantChatApi()

        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            # Test with empty question
            api._update_conversation_history("", "response", "session", [], "user")
            mock_conv.upsert.assert_not_called()

            # Test with empty response
            api._update_conversation_history("question", "", "session", [], "user")
            mock_conv.upsert.assert_not_called()

            # Test with empty session
            api._update_conversation_history("question", "response", "", [], "user")
            mock_conv.upsert.assert_not_called()

    def test_update_conversation_history_preserves_original_messages(self):
        """Test that original previous_messages list is not modified."""
        api = AssistantChatApi()

        with patch("tangerine.resources.assistant.Conversation"):
            original_messages = [
                {"sender": "human", "text": "Hello"},
                {"sender": "ai", "text": "Hi!"},
            ]
            original_copy = original_messages.copy()

            api._update_conversation_history(
                "New question", "New response", "session", original_messages, "user"
            )

            # Original list should be unchanged
            assert original_messages == original_copy
            assert len(original_messages) == 2


class TestConversationModelLogic:
    """Test core conversation model logic without database dependencies."""

    def test_generate_title_from_query(self):
        """Test title generation from query."""
        from tangerine.models.conversation import Conversation

        conversation_json = {"query": "What is machine learning?"}
        title = Conversation.generate_title(conversation_json)

        assert title == "What is machine learning?..."

    def test_generate_title_long_query(self):
        """Test title generation with long query gets truncated."""
        from tangerine.models.conversation import Conversation

        long_query = "This is a very long query that should be truncated because it exceeds the expected length"
        conversation_json = {"query": long_query}
        title = Conversation.generate_title(conversation_json)

        assert title == long_query[:50] + "..."
        assert len(title) <= 53  # 50 chars + "..."

    def test_generate_title_no_query(self):
        """Test title generation without query."""
        from tangerine.models.conversation import Conversation

        conversation_json = {"user": "test_user"}
        title = Conversation.generate_title(conversation_json)

        assert title == "Untitled Conversation"

    def test_generate_title_empty_query(self):
        """Test title generation with empty query."""
        from tangerine.models.conversation import Conversation

        conversation_json = {"query": ""}
        title = Conversation.generate_title(conversation_json)

        assert title == "Untitled Conversation"

    def test_is_owned_by_user(self):
        """Test ownership validation."""
        from tangerine.models.conversation import Conversation

        conversation = Conversation()
        conversation.user_id = "test_user"

        assert conversation.is_owned_by("test_user") is True
        assert conversation.is_owned_by("other_user") is False

    def test_from_json_creates_conversation(self):
        """Test creating conversation from JSON."""
        from tangerine.models.conversation import Conversation

        conversation_json = {
            "sessionId": "12345678-1234-5678-9012-123456789012",
            "user": "test_user",
            "query": "Test query",
            "prevMsgs": [],
        }

        conversation = Conversation.from_json(conversation_json)

        assert conversation.user_id == "test_user"
        assert str(conversation.session_id) == "12345678-1234-5678-9012-123456789012"
        assert conversation.payload == conversation_json
        assert conversation.title == "Test query..."

    def test_from_json_with_uuid_session_id(self):
        """Test creating conversation with UUID session ID."""
        from tangerine.models.conversation import Conversation

        session_uuid = uuid.UUID("12345678-1234-5678-9012-123456789012")
        conversation_json = {"sessionId": session_uuid, "user": "test_user", "query": "Test query"}

        conversation = Conversation.from_json(conversation_json)

        assert conversation.session_id == session_uuid
