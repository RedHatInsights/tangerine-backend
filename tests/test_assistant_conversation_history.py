import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from tangerine.resources.assistant import AssistantAdvancedChatApi, AssistantChatApi


class TestAssistantConversationHistory:
    """Tests for conversation history functionality in AssistantChatApi."""

    @pytest.fixture
    def assistant_chat_api(self):
        """Create AssistantChatApi instance with mocked dependencies."""
        return AssistantChatApi()

    @pytest.fixture
    def assistant_advanced_chat_api(self):
        """Create AssistantAdvancedChatApi instance with mocked dependencies."""
        return AssistantAdvancedChatApi()

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_new_conversation(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with no previous messages."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        expected_payload = {
            "sessionId": session_uuid,
            "user": user,
            "query": question,
            "prevMsgs": [
                {"sender": "human", "text": question},
                {"sender": "ai", "text": response_text},
            ],
        }

        mock_conversation.upsert.assert_called_once_with(expected_payload)

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_with_previous_messages(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with existing previous messages."""
        question = "What about machine learning?"
        response_text = "Machine learning is a subset of AI."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = [
            {"sender": "human", "text": "Hello"},
            {"sender": "ai", "text": "Hi there!"},
            {"sender": "human", "text": "What is AI?"},
            {"sender": "ai", "text": "AI is artificial intelligence."},
        ]
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        expected_payload = {
            "sessionId": session_uuid,
            "user": user,
            "query": question,
            "prevMsgs": [
                {"sender": "human", "text": "Hello"},
                {"sender": "ai", "text": "Hi there!"},
                {"sender": "human", "text": "What is AI?"},
                {"sender": "ai", "text": "AI is artificial intelligence."},
                {"sender": "human", "text": question},
                {"sender": "ai", "text": response_text},
            ],
        }

        mock_conversation.upsert.assert_called_once_with(expected_payload)

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_none_previous_messages(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with None previous messages."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = None
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        expected_payload = {
            "sessionId": session_uuid,
            "user": user,
            "query": question,
            "prevMsgs": [
                {"sender": "human", "text": question},
                {"sender": "ai", "text": response_text},
            ],
        }

        mock_conversation.upsert.assert_called_once_with(expected_payload)

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_missing_question(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with missing question."""
        question = ""
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        # Should return early without calling upsert
        mock_conversation.upsert.assert_not_called()

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_missing_response(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with missing response."""
        question = "What is AI?"
        response_text = ""
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        # Should return early without calling upsert
        mock_conversation.upsert.assert_not_called()

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_missing_session(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with missing session UUID."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = ""
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        # Should return early without calling upsert
        mock_conversation.upsert.assert_not_called()

    @patch("tangerine.resources.assistant.Conversation")
    def test_update_conversation_history_exception_handling(
        self, mock_conversation, assistant_chat_api
    ):
        """Test updating conversation history with database exception."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        mock_conversation.upsert.side_effect = Exception("Database error")

        # Should not raise exception, but log it
        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        mock_conversation.upsert.assert_called_once()

    def test_handle_standard_response_calls_update_conversation_history(self, assistant_chat_api):
        """Test that _handle_standard_response calls _update_conversation_history."""
        # Mock dependencies
        assistant_chat_api._parse_search_results = Mock(return_value=[])
        assistant_chat_api._log_interaction = Mock()
        assistant_chat_api._update_conversation_history = Mock()

        llm_response = ["AI is ", "artificial ", "intelligence."]
        search_metadata = [{"doc": "metadata"}]
        question = "What is AI?"
        embedding = [0.1, 0.2, 0.3]
        search_results = []
        session_uuid = "12345678-1234-5678-9012-123456789012"
        interaction_id = "interaction-123"
        client = "test-client"
        user = "test_user"
        previous_messages = [{"sender": "human", "text": "Hello"}]

        response, status_code = assistant_chat_api._handle_standard_response(
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
        )

        # Verify response
        assert status_code == 200
        assert response["text_content"] == "AI is artificial intelligence."

        # Verify conversation history was updated
        assistant_chat_api._update_conversation_history.assert_called_once_with(
            question, "AI is artificial intelligence.", session_uuid, previous_messages, user
        )

    def test_handle_streaming_response_calls_update_conversation_history(self, assistant_chat_api):
        """Test that _handle_streaming_response calls _update_conversation_history."""
        # Mock dependencies
        assistant_chat_api._parse_search_results = Mock(return_value=[])
        assistant_chat_api._log_interaction = Mock()
        assistant_chat_api._update_conversation_history = Mock()

        llm_response = ["AI is ", "artificial ", "intelligence."]
        search_metadata = [{"doc": "metadata"}]
        question = "What is AI?"
        embedding = [0.1, 0.2, 0.3]
        search_results = []
        session_uuid = "12345678-1234-5678-9012-123456789012"
        interaction_id = "interaction-123"
        client = "test-client"
        user = "test_user"
        previous_messages = [{"sender": "human", "text": "Hello"}]

        with patch("tangerine.resources.assistant.Response") as mock_response:
            with patch("tangerine.resources.assistant.stream_with_context") as mock_stream:
                assistant_chat_api._handle_streaming_response(
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
                )

                # The update should be called within the generator function
                # We need to execute the generator to verify this
                generator_func = mock_stream.call_args[0][0]
                list(generator_func())  # Execute the generator

                # Verify conversation history was updated
                assistant_chat_api._update_conversation_history.assert_called_once_with(
                    question,
                    "AI is artificial intelligence.",
                    session_uuid,
                    previous_messages,
                    user,
                )

    @patch("tangerine.resources.assistant.request")
    def test_assistant_chat_api_integration(self, mock_request, assistant_chat_api):
        """Test full integration of conversation history in AssistantChatApi.post()."""
        # Mock all dependencies
        assistant_chat_api._get_assistant = Mock(return_value=Mock())
        assistant_chat_api._extract_request_data = Mock(
            return_value=(
                "What is AI?",
                "session-123",
                False,
                [],
                "interaction-123",
                "client",
                "user",
            )
        )
        assistant_chat_api._embed_question = Mock(return_value=[0.1, 0.2])
        assistant_chat_api._get_search_results = Mock(return_value=[])
        assistant_chat_api._call_llm = Mock(return_value=(["AI response"], [{"metadata": "test"}]))
        assistant_chat_api._is_streaming_response = Mock(return_value=False)
        assistant_chat_api._handle_standard_response = Mock(
            return_value=({"response": "test"}, 200)
        )

        response, status_code = assistant_chat_api.post(1)

        # Verify that _handle_standard_response was called with previous_messages
        args = assistant_chat_api._handle_standard_response.call_args[0]
        kwargs = (
            assistant_chat_api._handle_standard_response.call_args[1]
            if assistant_chat_api._handle_standard_response.call_args[1]
            else {}
        )

        # Check that previous_messages was passed as the last positional argument or as kwarg
        if len(args) > 9:
            previous_messages = args[9]
        else:
            previous_messages = kwargs.get("previous_messages")

        assert previous_messages == []  # Should be the previous messages from extract_request_data

    @patch("tangerine.resources.assistant.request")
    def test_assistant_advanced_chat_api_integration(
        self, mock_request, assistant_advanced_chat_api
    ):
        """Test full integration of conversation history in AssistantAdvancedChatApi.post()."""
        # Mock request.json
        mock_request.json = {
            "assistants": ["test-assistant"],
            "query": "What is AI?",
            "sessionId": "session-123",
            "prevMsgs": [{"sender": "human", "text": "Hello"}],
        }
        mock_request.json.get = lambda key, default=None: mock_request.json.get(key, default)

        # Mock all dependencies
        assistant_advanced_chat_api._get_assistants = Mock(return_value=[Mock(id=1)])
        assistant_advanced_chat_api._is_streaming_response = Mock(return_value=False)
        assistant_advanced_chat_api._handle_standard_response = Mock(
            return_value=({"response": "test"}, 200)
        )

        with patch("tangerine.resources.assistant.embed_query") as mock_embed:
            with patch("tangerine.resources.assistant.search_engine") as mock_search:
                with patch("tangerine.resources.assistant.llm") as mock_llm:
                    mock_embed.return_value = [0.1, 0.2]
                    mock_search.search.return_value = []
                    mock_llm.ask.return_value = (["AI response"], [{"metadata": "test"}])

                    response, status_code = assistant_advanced_chat_api.post()

                    # Verify that _handle_standard_response was called with previous_messages
                    args = assistant_advanced_chat_api._handle_standard_response.call_args[0]
                    kwargs = (
                        assistant_advanced_chat_api._handle_standard_response.call_args[1]
                        if assistant_advanced_chat_api._handle_standard_response.call_args[1]
                        else {}
                    )

                    # Check that previous_messages was passed
                    if len(args) > 9:
                        previous_messages = args[9]
                    else:
                        previous_messages = kwargs.get("previous_messages")

                    assert previous_messages == [{"sender": "human", "text": "Hello"}]

    def test_previous_messages_copy_integrity(self, assistant_chat_api):
        """Test that previous_messages are properly copied and not modified."""
        original_messages = [
            {"sender": "human", "text": "Hello"},
            {"sender": "ai", "text": "Hi there!"},
        ]
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        user = "test_user"

        with patch("tangerine.resources.assistant.Conversation") as mock_conversation:
            assistant_chat_api._update_conversation_history(
                question, response_text, session_uuid, original_messages, user
            )

            # Verify original messages were not modified
            assert original_messages == [
                {"sender": "human", "text": "Hello"},
                {"sender": "ai", "text": "Hi there!"},
            ]

            # Verify the correct payload was sent to upsert
            call_args = mock_conversation.upsert.call_args[0][0]
            assert len(call_args["prevMsgs"]) == 4  # original 2 + new 2
            assert call_args["prevMsgs"][0] == {"sender": "human", "text": "Hello"}
            assert call_args["prevMsgs"][1] == {"sender": "ai", "text": "Hi there!"}
            assert call_args["prevMsgs"][2] == {"sender": "human", "text": question}
            assert call_args["prevMsgs"][3] == {"sender": "ai", "text": response_text}


class TestConversationHistoryMessageFormat:
    """Tests for proper message format in conversation history."""

    @pytest.fixture
    def assistant_chat_api(self):
        """Create AssistantChatApi instance."""
        return AssistantChatApi()

    @patch("tangerine.resources.assistant.Conversation")
    def test_message_format_consistency(self, mock_conversation, assistant_chat_api):
        """Test that conversation history messages follow the expected format."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        call_args = mock_conversation.upsert.call_args[0][0]
        messages = call_args["prevMsgs"]

        # Verify message structure
        assert len(messages) == 2

        # Human message
        human_msg = messages[0]
        assert human_msg["sender"] == "human"
        assert human_msg["text"] == question
        assert len(human_msg) == 2  # Only sender and text fields

        # AI message
        ai_msg = messages[1]
        assert ai_msg["sender"] == "ai"
        assert ai_msg["text"] == response_text
        assert len(ai_msg) == 2  # Only sender and text fields

    @patch("tangerine.resources.assistant.Conversation")
    def test_conversation_payload_structure(self, mock_conversation, assistant_chat_api):
        """Test that the conversation payload has the correct structure."""
        question = "What is AI?"
        response_text = "AI is artificial intelligence."
        session_uuid = "12345678-1234-5678-9012-123456789012"
        previous_messages = []
        user = "test_user"

        assistant_chat_api._update_conversation_history(
            question, response_text, session_uuid, previous_messages, user
        )

        call_args = mock_conversation.upsert.call_args[0][0]

        # Verify payload structure
        expected_keys = {"sessionId", "user", "query", "prevMsgs"}
        assert set(call_args.keys()) == expected_keys

        assert call_args["sessionId"] == session_uuid
        assert call_args["user"] == user
        assert call_args["query"] == question
        assert isinstance(call_args["prevMsgs"], list)
