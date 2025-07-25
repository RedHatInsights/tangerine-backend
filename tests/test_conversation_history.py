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

    def test_update_conversation_history_preserves_current_message_fields(self):
        """Test that current message fields like isIntroductionPrompt are preserved."""
        api = AssistantChatApi()

        with patch("tangerine.resources.assistant.Conversation") as mock_conv:
            question = "Hello, I need help"
            response_text = "Hi there!"
            session_uuid = "test-session-123"
            previous_messages = []
            user = "test_user"

            # Current message with isIntroductionPrompt field
            current_message = {"sender": "human", "text": question, "isIntroductionPrompt": True}

            api._update_conversation_history(
                question,
                response_text,
                session_uuid,
                previous_messages,
                user,
                None,
                current_message,
            )

            # Verify the payload passed to upsert preserves the isIntroductionPrompt field
            call_args = mock_conv.upsert.call_args[0][0]

            assert call_args["sessionId"] == session_uuid
            assert call_args["user"] == user
            assert call_args["query"] == question
            assert len(call_args["prevMsgs"]) == 2

            # Check that the current message preserves all fields
            user_message = call_args["prevMsgs"][0]
            assert user_message["sender"] == "human"
            assert user_message["text"] == question
            assert user_message["isIntroductionPrompt"] is True  # This should be preserved!

            # Check AI response
            ai_message = call_args["prevMsgs"][1]
            assert ai_message["sender"] == "ai"
            assert ai_message["text"] == response_text


class TestConversationModelLogic:
    """Test core conversation model logic without database dependencies."""

    def test_generate_title_from_query(self):
        """Test title generation from first non-introduction query."""
        from tangerine.models.conversation import Conversation

        # Test with user message that has no isIntroductionPrompt field (should generate title)
        conversation_json = {
            "query": "What is machine learning?",
            "prevMsgs": [{"sender": "human", "text": "What is machine learning?"}],
        }

        # Mock the LLM call
        from unittest.mock import patch

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Machine Learning Questions"
            title = Conversation.generate_title(conversation_json)
            assert title == "Machine Learning Questions"
            mock_generate.assert_called_once_with(["What is machine learning?"])

    def test_generate_title_long_query(self):
        """Test title generation with long query gets truncated on fallback."""
        from tangerine.models.conversation import Conversation

        long_query = "This is a very long query that should be truncated because it exceeds the expected length"
        conversation_json = {
            "query": long_query,
            "prevMsgs": [{"sender": "human", "text": long_query}],
        }

        # Mock LLM to fail, testing fallback
        from unittest.mock import patch

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.side_effect = Exception("LLM error")
            title = Conversation.generate_title(conversation_json)
            assert title == long_query[:self.TRUNCATION_LENGTH] + "..."

    def test_generate_title_no_query(self):
        """Test title generation without user queries."""
        from tangerine.models.conversation import Conversation

        # Test with no previous messages - should return "New chat"
        conversation_json = {"user": "test_user", "prevMsgs": []}
        title = Conversation.generate_title(conversation_json)

        assert title == "New chat"

    def test_generate_title_empty_query(self):
        """Test title generation with only AI messages."""
        from tangerine.models.conversation import Conversation

        # Test with only AI messages - should return "New chat"
        conversation_json = {"prevMsgs": [{"sender": "ai", "text": "Hello! How can I help you?"}]}
        title = Conversation.generate_title(conversation_json)

        assert title == "New chat"

    def test_generate_title_skip_introduction_prompt(self):
        """Test title generation skips introduction prompts."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        conversation_json = {
            "prevMsgs": [
                {"sender": "human", "text": "Hello, I need help", "isIntroductionPrompt": True},
                {"sender": "ai", "text": "Hi! How can I assist you?"},
                {"sender": "human", "text": "What is machine learning?"},
            ]
        }

        # Should use the second user query (first non-introduction)
        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Machine Learning Questions"
            title = Conversation.generate_title(conversation_json)
            assert title == "Machine Learning Questions"
            mock_generate.assert_called_once_with(["What is machine learning?"])

    def test_generate_title_introduction_prompt_false(self):
        """Test title generation with explicit isIntroductionPrompt=false."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        conversation_json = {
            "prevMsgs": [
                {"sender": "human", "text": "What is Python?", "isIntroductionPrompt": False}
            ]
        }

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Python Programming"
            title = Conversation.generate_title(conversation_json)
            assert title == "Python Programming"
            mock_generate.assert_called_once_with(["What is Python?"])

    def test_generate_title_all_introduction_prompts(self):
        """Test title generation when all user queries are introduction prompts."""
        from tangerine.models.conversation import Conversation

        conversation_json = {
            "prevMsgs": [
                {"sender": "human", "text": "Hello", "isIntroductionPrompt": True},
                {"sender": "ai", "text": "Hi there!"},
                {"sender": "human", "text": "I need help", "isIntroductionPrompt": True},
            ]
        }

        title = Conversation.generate_title(conversation_json)
        assert title == "New chat"

    def test_generate_title_mixed_messages(self):
        """Test title generation with mixed AI and human messages."""
        from tangerine.models.conversation import Conversation

        # Test with only AI messages - should return "New chat"
        conversation_json = {
            "prevMsgs": [
                {"sender": "ai", "text": "Hello!"},
                {"sender": "ai", "text": "How can I help?"},
            ]
        }

        title = Conversation.generate_title(conversation_json)
        assert title == "New chat"

    def test_title_updates_on_first_real_message(self):
        """Test that title updates when first non-introduction message arrives."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        # Mock the LLM title generation
        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Machine Learning Basics"

            # Test the _update_title_if_needed method directly
            # Start with intro-only conversation
            intro_conversation = {
                "prevMsgs": [
                    {"sender": "human", "text": "Hello", "isIntroductionPrompt": True},
                    {"sender": "ai", "text": "Hi! How can I help?"},
                ]
            }

            # This should generate "New chat"
            intro_title = Conversation.generate_title(intro_conversation)
            assert intro_title == "New chat"

            # Now add a real user message
            full_conversation = {
                "prevMsgs": [
                    {"sender": "human", "text": "Hello", "isIntroductionPrompt": True},
                    {"sender": "ai", "text": "Hi! How can I help?"},
                    {"sender": "human", "text": "What is machine learning?"},
                ]
            }

            # Mock conversation object
            mock_conversation = type("MockConversation", (), {})()
            mock_conversation.title = "New chat"  # Current title from intro-only

            # This should update the title
            Conversation._update_title_if_needed(mock_conversation, full_conversation)

            # Title should now be the LLM-generated one
            assert mock_conversation.title == "Machine Learning Basics"
            mock_generate.assert_called_once()

    def test_title_not_updated_if_already_real(self):
        """Test that title doesn't get updated if it's already a real (non-default) title."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Advanced AI Topics"

            # Conversation with existing real title
            conversation_data = {
                "prevMsgs": [
                    {"sender": "human", "text": "What is AI?"},
                    {"sender": "ai", "text": "AI is..."},
                    {"sender": "human", "text": "Tell me about neural networks"},
                ]
            }

            # Mock conversation with existing real title
            mock_conversation = type("MockConversation", (), {})()
            mock_conversation.title = "Artificial Intelligence Basics"  # Already has real title

            # This should NOT update the title
            Conversation._update_title_if_needed(mock_conversation, conversation_data)

            # Title should remain unchanged
            assert mock_conversation.title == "Artificial Intelligence Basics"
            # LLM should not be called since we don't need to update
            mock_generate.assert_not_called()

    def test_is_owned_by_user(self):
        """Test ownership validation."""
        from tangerine.models.conversation import Conversation

        conversation = Conversation()
        conversation.user_id = "test_user"

        assert conversation.is_owned_by("test_user") is True
        assert conversation.is_owned_by("other_user") is False

    def test_from_json_creates_conversation(self):
        """Test creating conversation from JSON."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        conversation_json = {
            "sessionId": "12345678-1234-5678-9012-123456789012",
            "user": "test_user",
            "query": "Test query",
            "prevMsgs": [{"sender": "human", "text": "Test query"}],
        }

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Test Query Title"
            conversation = Conversation.from_json(conversation_json)

            assert conversation.user_id == "test_user"
            assert str(conversation.session_id) == "12345678-1234-5678-9012-123456789012"
            assert conversation.payload == conversation_json
            assert conversation.title == "Test Query Title"

    def test_from_json_with_uuid_session_id(self):
        """Test creating conversation with UUID session ID."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        session_uuid = uuid.UUID("12345678-1234-5678-9012-123456789012")
        conversation_json = {
            "sessionId": session_uuid,
            "user": "test_user",
            "query": "Test query",
            "prevMsgs": [{"sender": "human", "text": "Test query"}],
        }

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Test Query Title"
            conversation = Conversation.from_json(conversation_json)

            assert conversation.session_id == session_uuid
            assert conversation.title == "Test Query Title"

    def test_from_json_with_assistant_name(self):
        """Test creating conversation from JSON with assistant name."""
        from unittest.mock import patch

        from tangerine.models.conversation import Conversation

        conversation_json = {
            "sessionId": "12345678-1234-5678-9012-123456789012",
            "user": "test_user",
            "query": "Test query",
            "assistantName": "Support Bot",
            "prevMsgs": [{"sender": "human", "text": "Test query"}],
        }

        with patch("tangerine.llm.generate_conversation_title") as mock_generate:
            mock_generate.return_value = "Test Query Title"
            conversation = Conversation.from_json(conversation_json)

            assert conversation.user_id == "test_user"
            assert conversation.assistant_name == "Support Bot"
            assert conversation.payload == conversation_json
            assert conversation.title == "Test Query Title"

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
