from unittest.mock import MagicMock

import pytest
from langchain_core.documents import Document

from tangerine.resources.assistant import AssistantChatApi  # Import your API class


@pytest.fixture
def assistant_chat_api():
    """Fixture to create an instance of AssistantChatApi with dependencies mocked."""
    api_instance = AssistantChatApi()

    # Mock dependencies
    api_instance._get_assistant = MagicMock()
    api_instance._extract_request_data = MagicMock()
    api_instance._get_search_results = MagicMock()
    api_instance._parse_search_results = MagicMock()
    api_instance._embed_question = MagicMock()
    api_instance._call_llm = MagicMock()
    api_instance._is_streaming_response = MagicMock()
    api_instance._handle_streaming_response = MagicMock()
    api_instance._handle_final_response = MagicMock()
    api_instance._log_interaction = MagicMock()
    api_instance._interaction_storage_enabled = MagicMock()

    return api_instance


def test_post_chat_assistant_not_found(assistant_chat_api):
    """Test case when the assistant is not found."""

    assistant_chat_api._get_assistant.return_value = None  # Simulate no assistant found

    response, status_code = assistant_chat_api.post(1)

    assert status_code == 404
    assert response == {"message": "assistant not found"}


def test_post_chat_non_streaming(assistant_chat_api):
    """Test case when chat is non-streaming."""

    mock_assistant = MagicMock()
    mock_query = "What is AI?"
    mock_session_uuid = "1234-5678"
    mock_previous_messages = []
    mock_search_results = [
        Document(page_content="AI is artificial intelligence", metadata={"source": "wiki"})
    ]
    mock_embedding = [0.1, 0.2, 0.3]
    mock_llm_response = "AI is the simulation of human intelligence."
    mock_interaction_id = "interaction-1234"
    mock_client = "client"

    # Configure mocks
    assistant_chat_api._get_assistant.return_value = mock_assistant
    assistant_chat_api._extract_request_data.return_value = (
        mock_query,
        mock_session_uuid,
        False,
        mock_previous_messages,
        mock_interaction_id,
        mock_client,
    )
    assistant_chat_api._embed_question.return_value = mock_embedding
    assistant_chat_api._get_search_results.return_value = mock_search_results
    assistant_chat_api._call_llm.return_value = mock_llm_response
    assistant_chat_api._is_streaming_response.return_value = False
    assistant_chat_api._handle_final_response.return_value = {"response": mock_llm_response}, 200
    assistant_chat_api._interaction_storage_enabled.return_value = True

    response, status_code = assistant_chat_api.post(1)

    # Assertions
    assert status_code == 200
    assert response == {"response": mock_llm_response}

    # Ensure all expected functions were called
    assistant_chat_api._get_assistant.assert_called_once_with(1)
    assistant_chat_api._extract_request_data.assert_called_once()
    assistant_chat_api._embed_question.assert_called_once_with(mock_query)
    assistant_chat_api._call_llm.assert_called_once_with(
        mock_assistant,
        mock_previous_messages,
        mock_query,
        mock_search_results,
        False,
        mock_interaction_id,
    )
    assistant_chat_api._handle_final_response.assert_called_once_with(
        mock_llm_response,
        mock_query,
        mock_embedding,
        mock_search_results,
        mock_session_uuid,
        mock_interaction_id,
        mock_client,
    )


def test_post_chat_streaming(assistant_chat_api):
    """Test case when chat is streaming."""

    mock_assistant = MagicMock()
    mock_query = "What is AI?"
    mock_session_uuid = "1234-5678"
    mock_previous_messages = []
    mock_search_results = [
        Document(page_content="AI is artificial intelligence", metadata={"source": "wiki"})
    ]
    mock_embedding = [0.1, 0.2, 0.3]
    mock_llm_response = MagicMock()  # Mock a streaming generator
    mock_llm_response.__iter__.return_value = iter(["data: AI is", "data: an intelligence"])
    mock_interaction_id = "interaction-1234"
    mock_client = "client"

    # Configure mocks
    assistant_chat_api._get_assistant.return_value = mock_assistant
    assistant_chat_api._extract_request_data.return_value = (
        mock_query,
        mock_session_uuid,
        True,
        mock_previous_messages,
        mock_interaction_id,
        mock_client,
    )
    assistant_chat_api._embed_question.return_value = mock_embedding
    assistant_chat_api._get_search_results.return_value = mock_search_results
    assistant_chat_api._call_llm.return_value = mock_llm_response
    assistant_chat_api._is_streaming_response.return_value = True
    assistant_chat_api._handle_streaming_response.return_value = "mock_stream_response"
    assistant_chat_api._interaction_storage_enabled.return_value = True

    response = assistant_chat_api.post(1)

    # Assertions
    assert response == "mock_stream_response"

    # Ensure all expected functions were called
    assistant_chat_api._get_assistant.assert_called_once_with(1)
    assistant_chat_api._extract_request_data.assert_called_once()
    assistant_chat_api._embed_question.assert_called_once_with(mock_query)
    assistant_chat_api._call_llm.assert_called_once_with(
        mock_assistant,
        mock_previous_messages,
        mock_query,
        mock_search_results,
        True,
        mock_interaction_id,
    )
    assistant_chat_api._handle_streaming_response.assert_called_once_with(
        mock_llm_response,
        mock_query,
        mock_embedding,
        mock_search_results,
        mock_session_uuid,
        mock_interaction_id,
        mock_client,
    )
