import pytest
from unittest.mock import MagicMock, patch
from resources.agent import AgentChatApi  # Import your API class


@pytest.fixture
def agent_chat_api():
    """Fixture to create an instance of AgentChatApi with dependencies mocked."""
    api_instance = AgentChatApi()

    # Mock dependencies
    api_instance._get_agent = MagicMock()
    api_instance._extract_request_data = MagicMock()
    api_instance._retrieve_relevant_documents = MagicMock()
    api_instance._embed_query = MagicMock()
    api_instance._call_llm = MagicMock()
    api_instance._is_streaming_response = MagicMock()
    api_instance._handle_streaming_response = MagicMock()
    api_instance._handle_final_response = MagicMock()
    api_instance._log_interaction = MagicMock()

    return api_instance


def test_post_chat_agent_not_found(agent_chat_api):
    """Test case when the agent is not found."""
    agent_chat_api._get_agent.return_value = None  # Simulate no agent found

    response, status_code = agent_chat_api.post(1)

    assert status_code == 404
    assert response == {"message": "agent not found"}


def test_post_chat_non_streaming(agent_chat_api):
    """Test case when chat is non-streaming."""
    mock_agent = MagicMock()
    mock_query = "What is AI?"
    mock_session_uuid = "1234-5678"
    mock_previous_messages = []
    mock_source_doc_chunks = [{"text": "AI is artificial intelligence", "source": "wiki"}]
    mock_embedding = [0.1, 0.2, 0.3]
    mock_llm_response = "AI is the simulation of human intelligence."

    # Configure mocks
    agent_chat_api._get_agent.return_value = mock_agent
    agent_chat_api._extract_request_data.return_value = (mock_query, mock_session_uuid, False, mock_previous_messages)
    agent_chat_api._retrieve_relevant_documents.return_value = mock_source_doc_chunks
    agent_chat_api._embed_query.return_value = mock_embedding
    agent_chat_api._call_llm.return_value = mock_llm_response
    agent_chat_api._is_streaming_response.return_value = False
    agent_chat_api._handle_final_response.return_value = {"response": mock_llm_response}, 200

    response, status_code = agent_chat_api.post(1)

    # Assertions
    agent_chat_api._get_agent.assert_called_once_with(1)
    agent_chat_api._extract_request_data.assert_called_once()
    agent_chat_api._retrieve_relevant_documents.assert_called_once_with(mock_agent, mock_query)
    agent_chat_api._embed_query.assert_called_once_with(mock_query)
    agent_chat_api._call_llm.assert_called_once_with(mock_agent, mock_query, mock_previous_messages, False)
    agent_chat_api._handle_final_response.assert_called_once_with(mock_llm_response, mock_query, mock_source_doc_chunks, mock_embedding, mock_session_uuid)

    assert status_code == 200
    assert response == {"response": mock_llm_response}


def test_post_chat_streaming(agent_chat_api):
    """Test case when chat is streaming."""
    mock_agent = MagicMock()
    mock_query = "What is AI?"
    mock_session_uuid = "1234-5678"
    mock_previous_messages = []
    mock_source_doc_chunks = [{"text": "AI is artificial intelligence", "source": "wiki"}]
    mock_embedding = [0.1, 0.2, 0.3]
    mock_llm_response = MagicMock()  # Mock a streaming generator
    mock_llm_response.__iter__.return_value = iter(["data: AI is", "data: an intelligence"])

    # Configure mocks
    agent_chat_api._get_agent.return_value = mock_agent
    agent_chat_api._extract_request_data.return_value = (mock_query, mock_session_uuid, True, mock_previous_messages)
    agent_chat_api._retrieve_relevant_documents.return_value = mock_source_doc_chunks
    agent_chat_api._embed_query.return_value = mock_embedding
    agent_chat_api._call_llm.return_value = mock_llm_response
    agent_chat_api._is_streaming_response.return_value = True
    agent_chat_api._handle_streaming_response.return_value = "mock_stream_response"

    response = agent_chat_api.post(1)

    # Assertions
    agent_chat_api._get_agent.assert_called_once_with(1)
    agent_chat_api._extract_request_data.assert_called_once()
    agent_chat_api._retrieve_relevant_documents.assert_called_once_with(mock_agent, mock_query)
    agent_chat_api._embed_query.assert_called_once_with(mock_query)
    agent_chat_api._call_llm.assert_called_once_with(mock_agent, mock_query, mock_previous_messages, True)
    agent_chat_api._handle_streaming_response.assert_called_once_with(mock_llm_response, mock_query, mock_source_doc_chunks, mock_embedding, mock_session_uuid)

    assert response == "mock_stream_response"
