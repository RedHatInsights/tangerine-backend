import uuid
from unittest.mock import Mock, patch

import pytest

from tangerine.resources.conversation import (
    ConversationDeleteApi,
    ConversationListApi,
    ConversationRetrievalApi,
    ConversationUpsertApi,
)


@pytest.fixture
def sample_conversation_object():
    """Sample Conversation object for testing."""
    conversation = Mock()
    conversation.id = uuid.uuid4()
    conversation.user_id = "test_user"
    conversation.session_id = uuid.UUID("12345678-1234-5678-9012-123456789012")
    conversation.payload = {
        "sessionId": "12345678-1234-5678-9012-123456789012",
        "user": "test_user",
        "query": "What is AI?",
        "prevMsgs": [{"sender": "human", "text": "Hello"}, {"sender": "ai", "text": "Hi there!"}],
    }
    conversation.title = "What is AI?..."
    conversation.to_json.return_value = {
        "id": str(conversation.id),
        "user_id": "test_user",
        "session_id": "12345678-1234-5678-9012-123456789012",
        "created_at": "2023-01-01T00:00:00",
        "updated_at": "2023-01-01T00:00:00",
        "payload": conversation.payload,
        "title": "What is AI?...",
    }
    return conversation


class TestConversationListApi:
    """Tests for ConversationListApi."""

    @pytest.fixture
    def conversation_list_api(self):
        """Create ConversationListApi instance."""
        return ConversationListApi()

    def test_post_success(self, conversation_list_api, sample_conversation_object):
        """Test successful conversation list retrieval."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                # Mock request data
                mock_request.get_json.return_value = {"user_id": "test_user"}

                # Mock Conversation.get_by_user
                mock_conversation.get_by_user.return_value = [sample_conversation_object]

                response, status_code = conversation_list_api.post()

                assert status_code == 200
                assert response == [sample_conversation_object.to_json()]
                mock_conversation.get_by_user.assert_called_once_with("test_user")

    def test_post_no_data(self, conversation_list_api):
        """Test conversation list with no data provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = None

            response, status_code = conversation_list_api.post()

            assert status_code == 400
            assert response == {"error": "No data provided"}

    def test_post_no_user_id(self, conversation_list_api):
        """Test conversation list with no user_id provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = {"other_field": "value"}

            response, status_code = conversation_list_api.post()

            assert status_code == 400
            assert response == {"error": "User ID is required"}

    def test_post_exception(self, conversation_list_api):
        """Test conversation list with database exception."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"user_id": "test_user"}
                mock_conversation.get_by_user.side_effect = Exception("Database error")

                response, status_code = conversation_list_api.post()

                assert status_code == 500
                assert "error" in response
                assert "Database error" in response["error"]


class TestConversationRetrievalApi:
    """Tests for ConversationRetrievalApi."""

    @pytest.fixture
    def conversation_retrieval_api(self):
        """Create ConversationRetrievalApi instance."""
        return ConversationRetrievalApi()

    def test_post_success(self, conversation_retrieval_api, sample_conversation_object):
        """Test successful conversation retrieval."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "12345678-1234-5678-9012-123456789012"}
                mock_conversation.get_by_session.return_value = sample_conversation_object

                response, status_code = conversation_retrieval_api.post()

                assert status_code == 200
                assert response == sample_conversation_object.to_json()
                mock_conversation.get_by_session.assert_called_once_with(
                    "12345678-1234-5678-9012-123456789012"
                )

    def test_post_no_data(self, conversation_retrieval_api):
        """Test conversation retrieval with no data provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = None

            response, status_code = conversation_retrieval_api.post()

            assert status_code == 400
            assert response == {"error": "No data provided"}

    def test_post_no_session_id(self, conversation_retrieval_api):
        """Test conversation retrieval with no session ID provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = {"other_field": "value"}

            response, status_code = conversation_retrieval_api.post()

            assert status_code == 400
            assert response == {"error": "Session ID is required"}

    def test_post_not_found(self, conversation_retrieval_api):
        """Test conversation retrieval when conversation not found."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "nonexistent-session"}
                mock_conversation.get_by_session.return_value = None

                response, status_code = conversation_retrieval_api.post()

                assert status_code == 404
                assert response == {"error": "Conversation not found"}

    def test_post_exception(self, conversation_retrieval_api):
        """Test conversation retrieval with database exception."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "test-session"}
                mock_conversation.get_by_session.side_effect = Exception("Database error")

                response, status_code = conversation_retrieval_api.post()

                assert status_code == 500
                assert "error" in response
                assert "Database error" in response["error"]


class TestConversationUpsertApi:
    """Tests for ConversationUpsertApi."""

    @pytest.fixture
    def conversation_upsert_api(self):
        """Create ConversationUpsertApi instance."""
        return ConversationUpsertApi()

    def test_post_success(self, conversation_upsert_api, sample_conversation_object):
        """Test successful conversation upsert."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                conversation_data = {
                    "sessionId": "12345678-1234-5678-9012-123456789012",
                    "user": "test_user",
                    "query": "What is AI?",
                    "prevMsgs": [],
                }
                mock_request.get_json.return_value = conversation_data
                mock_conversation.upsert.return_value = sample_conversation_object

                response, status_code = conversation_upsert_api.post()

                assert status_code == 200
                assert response == sample_conversation_object.to_json()
                mock_conversation.upsert.assert_called_once_with(conversation_data)

    def test_post_no_data(self, conversation_upsert_api):
        """Test conversation upsert with no data provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = None

            response, status_code = conversation_upsert_api.post()

            assert status_code == 400
            assert response == {"error": "No data provided"}

    def test_post_exception(self, conversation_upsert_api):
        """Test conversation upsert with database exception."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                conversation_data = {"sessionId": "test-session", "user": "test_user"}
                mock_request.get_json.return_value = conversation_data
                mock_conversation.upsert.side_effect = Exception("Database error")

                response, status_code = conversation_upsert_api.post()

                assert status_code == 500
                assert "error" in response
                assert "Database error" in response["error"]


class TestConversationDeleteApi:
    """Tests for ConversationDeleteApi."""

    @pytest.fixture
    def conversation_delete_api(self):
        """Create ConversationDeleteApi instance."""
        return ConversationDeleteApi()

    def test_post_success(self, conversation_delete_api):
        """Test successful conversation deletion."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {
                    "sessionId": "12345678-1234-5678-9012-123456789012",
                    "user_id": "test_user",
                }
                mock_conversation.delete_by_session.return_value = (
                    True,
                    "Conversation deleted successfully",
                )

                response, status_code = conversation_delete_api.post()

                assert status_code == 200
                assert response == {"message": "Conversation deleted successfully"}
                mock_conversation.delete_by_session.assert_called_once_with(
                    "12345678-1234-5678-9012-123456789012", "test_user"
                )

    def test_post_no_data(self, conversation_delete_api):
        """Test conversation deletion with no data provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = None

            response, status_code = conversation_delete_api.post()

            assert status_code == 400
            assert response == {"error": "No data provided"}

    def test_post_no_session_id(self, conversation_delete_api):
        """Test conversation deletion with no session ID provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = {"user_id": "test_user"}

            response, status_code = conversation_delete_api.post()

            assert status_code == 400
            assert response == {"error": "Session ID is required"}

    def test_post_no_user_id(self, conversation_delete_api):
        """Test conversation deletion with no user ID provided."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            mock_request.get_json.return_value = {"sessionId": "test-session"}

            response, status_code = conversation_delete_api.post()

            assert status_code == 400
            assert response == {"error": "User ID is required"}

    def test_post_not_found(self, conversation_delete_api):
        """Test conversation deletion when conversation not found."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {
                    "sessionId": "nonexistent-session",
                    "user_id": "test_user",
                }
                mock_conversation.delete_by_session.return_value = (False, "Conversation not found")

                response, status_code = conversation_delete_api.post()

                assert status_code == 400
                assert response == {"error": "Conversation not found"}

    def test_post_unauthorized(self, conversation_delete_api):
        """Test conversation deletion when user not authorized."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "test-session", "user_id": "wrong_user"}
                mock_conversation.delete_by_session.return_value = (
                    False,
                    "Unauthorized: You can only delete your own conversations",
                )

                response, status_code = conversation_delete_api.post()

                assert status_code == 403
                assert response == {"error": "Unauthorized: You can only delete your own conversations"}

    def test_post_exception(self, conversation_delete_api):
        """Test conversation deletion with database exception."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "test-session", "user_id": "test_user"}
                mock_conversation.delete_by_session.side_effect = Exception("Database error")

                response, status_code = conversation_delete_api.post()

                assert status_code == 500
                assert "error" in response
                assert "Database error" in response["error"]

    def test_post_status_code_mapping(self, conversation_delete_api):
        """Test that error messages are mapped to correct status codes."""
        with patch("tangerine.resources.conversation.request") as mock_request:
            with patch("tangerine.resources.conversation.Conversation") as mock_conversation:
                mock_request.get_json.return_value = {"sessionId": "test-session", "user_id": "test_user"}

                # Test 400 for "not found"
                mock_conversation.delete_by_session.return_value = (False, "conversation not found")
                response, status_code = conversation_delete_api.post()
                assert status_code == 400

                # Test 403 for other errors (unauthorized)
                mock_conversation.delete_by_session.return_value = (False, "Other error message")
                response, status_code = conversation_delete_api.post()
                assert status_code == 403
