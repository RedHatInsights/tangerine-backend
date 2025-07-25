"""Simple tests for conversation history functionality.

Focus on testing business logic only, not infrastructure like Flask or database operations.
"""

import uuid
from unittest.mock import patch

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

    def test_update_conversation_history_with_assistant_name(self):
        """Test that assistant name is properly included in conversation history."""
        api = AssistantChatApi()

        # Mock the Conversation.upsert to capture what gets passed to it
        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            question = "What is AI?"
            response_text = "AI is artificial intelligence."
            session_uuid = "test-session-123"
            previous_messages = []
            user = "test_user"
            assistant_name = "Support Bot"

            api._update_conversation_history(
                question, response_text, session_uuid, previous_messages, user, assistant_name
            )

            # Verify the payload passed to upsert includes assistant name
            call_args = mock_conv.upsert.call_args[0][0]

            assert call_args["sessionId"] == session_uuid
            assert call_args["user"] == user
            assert call_args["query"] == question
            assert call_args["assistantName"] == assistant_name
            assert len(call_args["prevMsgs"]) == 2
            assert call_args["prevMsgs"][0] == {"sender": "human", "text": question}
            assert call_args["prevMsgs"][1] == {"sender": "ai", "text": response_text}

    def test_update_conversation_history_without_assistant_name(self):
        """Test that conversation history works without assistant name."""
        api = AssistantChatApi()

        # Mock the Conversation.upsert to capture what gets passed to it
        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            question = "What is AI?"
            response_text = "AI is artificial intelligence."
            session_uuid = "test-session-123"
            previous_messages = []
            user = "test_user"

            api._update_conversation_history(
                question, response_text, session_uuid, previous_messages, user
            )

            # Verify the payload passed to upsert does not include assistant name
            call_args = mock_conv.upsert.call_args[0][0]

            assert call_args["sessionId"] == session_uuid
            assert call_args["user"] == user
            assert call_args["query"] == question
            assert "assistantName" not in call_args

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

        # Test with 1 user message - should return "New chat"
        conversation_json = {
            "query": "What is machine learning?",
            "prevMsgs": [
                {"sender": "human", "text": "What is machine learning?"}
            ]
        }
        title = Conversation.generate_title(conversation_json)

        assert title == "New chat"

    def test_generate_title_long_query(self):
        """Test title generation with long query gets truncated."""
        from tangerine.models.conversation import Conversation

        # Test with 1 user message - should return "New chat" regardless of length
        long_query = "This is a very long query that should be truncated because it exceeds the expected length"
        conversation_json = {
            "query": long_query,
            "prevMsgs": [
                {"sender": "human", "text": long_query}
            ]
        }
        title = Conversation.generate_title(conversation_json)

        assert title == "New chat"

    def test_generate_title_no_query(self):
        """Test title generation without query."""
        from tangerine.models.conversation import Conversation

        # Test with no previous messages - should return None
        conversation_json = {"user": "test_user", "prevMsgs": []}
        title = Conversation.generate_title(conversation_json)

        assert title is None

    def test_generate_title_empty_query(self):
        """Test title generation with empty query."""
        from tangerine.models.conversation import Conversation

        # Test with no previous messages - should return None
        conversation_json = {"query": "", "prevMsgs": []}
        title = Conversation.generate_title(conversation_json)

        assert title is None

    def test_generate_title_two_user_queries(self):
        """Test title generation with 2 user queries - should use LLM."""
        from tangerine.models.conversation import Conversation
        from unittest.mock import patch

        conversation_json = {
            "query": "What is Python?",
            "prevMsgs": [
                {"sender": "human", "text": "What is machine learning?"},
                {"sender": "ai", "text": "Machine learning is..."},
                {"sender": "human", "text": "What is Python?"}
            ]
        }
        
        # Mock the LLM call to return a predictable title
        with patch('tangerine.llm.generate_conversation_title') as mock_generate:
            mock_generate.return_value = "Python Programming Questions"
            title = Conversation.generate_title(conversation_json)
            assert title == "Python Programming Questions"
            mock_generate.assert_called_once_with(["What is Python?"])  # Only the second query

    def test_generate_title_two_user_queries_llm_fallback(self):
        """Test title generation with 2 user queries when LLM fails - should use fallback."""
        from tangerine.models.conversation import Conversation
        from unittest.mock import patch

        conversation_json = {
            "query": "What is Python?",
            "prevMsgs": [
                {"sender": "human", "text": "What is machine learning?"},
                {"sender": "ai", "text": "Machine learning is..."},
                {"sender": "human", "text": "What is Python?"}
            ]
        }
        
        # Mock the LLM call to raise an exception
        with patch('tangerine.llm.generate_conversation_title') as mock_generate:
            mock_generate.side_effect = Exception("LLM error")
            title = Conversation.generate_title(conversation_json)
            assert title == "What is Python?..."  # Uses second query as fallback

    def test_generate_title_more_than_two_user_queries(self):
        """Test title generation with >2 user queries - should return None."""
        from tangerine.models.conversation import Conversation

        conversation_json = {
            "query": "What is Java?",
            "prevMsgs": [
                {"sender": "human", "text": "What is machine learning?"},
                {"sender": "ai", "text": "Machine learning is..."},
                {"sender": "human", "text": "What is Python?"},
                {"sender": "ai", "text": "Python is..."},
                {"sender": "human", "text": "What is Java?"}
            ]
        }
        
        title = Conversation.generate_title(conversation_json)
        assert title is None

    def test_generate_title_mixed_messages(self):
        """Test title generation with mixed AI and human messages."""
        from tangerine.models.conversation import Conversation

        # Test with only AI messages - should return None
        conversation_json = {
            "prevMsgs": [
                {"sender": "ai", "text": "Hello!"},
                {"sender": "ai", "text": "How can I help?"}
            ]
        }
        
        title = Conversation.generate_title(conversation_json)
        assert title is None

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
            "prevMsgs": [
                {"sender": "human", "text": "Test query"}
            ],
        }

        conversation = Conversation.from_json(conversation_json)

        assert conversation.user_id == "test_user"
        assert str(conversation.session_id) == "12345678-1234-5678-9012-123456789012"
        assert conversation.payload == conversation_json
        assert conversation.title == "New chat"  # Updated for new logic

    def test_from_json_with_uuid_session_id(self):
        """Test creating conversation with UUID session ID."""
        from tangerine.models.conversation import Conversation

        session_uuid = uuid.UUID("12345678-1234-5678-9012-123456789012")
        conversation_json = {
            "sessionId": session_uuid, 
            "user": "test_user", 
            "query": "Test query",
            "prevMsgs": [
                {"sender": "human", "text": "Test query"}
            ]
        }

        conversation = Conversation.from_json(conversation_json)

        assert conversation.session_id == session_uuid
        assert conversation.title == "New chat"  # Updated for new logic

    def test_from_json_with_assistant_name(self):
        """Test creating conversation from JSON with assistant name."""
        from tangerine.models.conversation import Conversation

        conversation_json = {
            "sessionId": "12345678-1234-5678-9012-123456789012",
            "user": "test_user",
            "query": "Test query",
            "assistantName": "Support Bot",
            "prevMsgs": [
                {"sender": "human", "text": "Test query"}
            ],
        }

        conversation = Conversation.from_json(conversation_json)

        assert conversation.user_id == "test_user"
        assert conversation.assistant_name == "Support Bot"
        assert conversation.payload == conversation_json
        assert conversation.title == "New chat"  # Updated for new logic

    def test_to_json_includes_assistant_name(self):
        """Test that to_json includes assistant name in the output."""
        from tangerine.models.conversation import Conversation

        conversation = Conversation()
        conversation.id = uuid.uuid4()
        conversation.user_id = "test_user"
        conversation.session_id = uuid.uuid4()
        conversation.assistant_name = "Support Bot"
        conversation.payload = {"query": "Test query"}
        conversation.title = "Test query..."

        # Mock the timestamp fields
        from datetime import datetime

        conversation.created_at = datetime.now()
        conversation.updated_at = datetime.now()

        json_output = conversation.to_json()

        assert json_output["assistant_name"] == "Support Bot"
        assert json_output["user_id"] == "test_user"
        assert json_output["title"] == "Test query..."
