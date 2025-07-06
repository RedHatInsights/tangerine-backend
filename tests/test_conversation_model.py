import uuid
from unittest.mock import MagicMock, Mock, patch

import pytest

from tangerine.models.conversation import Conversation


@pytest.fixture
def sample_conversation_json():
    """Sample conversation JSON data for testing."""
    return {
        "sessionId": "12345678-1234-5678-9012-123456789012",
        "user": "test_user",
        "query": "What is AI?",
        "prevMsgs": [{"sender": "human", "text": "Hello"}, {"sender": "ai", "text": "Hi there!"}],
    }


@pytest.fixture
def sample_conversation_object():
    """Sample Conversation object for testing."""
    conversation = Conversation()
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
    return conversation


class TestConversationModel:
    """Tests for the Conversation model."""

    @patch("tangerine.models.conversation.db.session")
    def test_get_by_session_found(self, mock_db_session, sample_conversation_object):
        """Test getting a conversation by session ID when it exists."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = sample_conversation_object

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.get_by_session("12345678-1234-5678-9012-123456789012")

            assert result == sample_conversation_object
            mock_query.filter_by.assert_called_once_with(
                session_id="12345678-1234-5678-9012-123456789012"
            )

    @patch("tangerine.models.conversation.db.session")
    def test_get_by_session_not_found(self, mock_db_session):
        """Test getting a conversation by session ID when it doesn't exist."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = None

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.get_by_session("nonexistent-session")

            assert result is None
            mock_query.filter_by.assert_called_once_with(session_id="nonexistent-session")

    @patch("tangerine.models.conversation.db.session")
    def test_get_by_user_found(self, mock_db_session, sample_conversation_object):
        """Test getting conversations by user ID when they exist."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.all.return_value = [sample_conversation_object]

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.get_by_user("test_user")

            assert result == [sample_conversation_object]
            mock_query.filter_by.assert_called_once_with(user_id="test_user")

    @patch("tangerine.models.conversation.db.session")
    def test_get_by_user_empty(self, mock_db_session):
        """Test getting conversations by user ID when none exist."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.all.return_value = []

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.get_by_user("nonexistent_user")

            assert result == []
            mock_query.filter_by.assert_called_once_with(user_id="nonexistent_user")

    @patch("tangerine.models.conversation.db.session")
    def test_upsert_new_conversation(self, mock_db_session, sample_conversation_json):
        """Test upserting a new conversation that doesn't exist."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = None

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.upsert(sample_conversation_json)

            # Verify the conversation was created
            assert result.user_id == "test_user"
            assert result.session_id == uuid.UUID("12345678-1234-5678-9012-123456789012")
            assert result.payload == sample_conversation_json
            assert result.title == "What is AI?..."

            # Verify database operations
            mock_db_session.add.assert_called_once_with(result)
            mock_db_session.commit.assert_called_once()

    @patch("tangerine.models.conversation.db.session")
    def test_upsert_existing_conversation_same_user(
        self, mock_db_session, sample_conversation_json, sample_conversation_object
    ):
        """Test upserting an existing conversation owned by the same user."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = sample_conversation_object

        with patch.object(Conversation, "query", mock_query):
            with patch.object(sample_conversation_object, "is_owned_by", return_value=True):
                result = Conversation.upsert(sample_conversation_json)

                # Verify the conversation was updated
                assert result == sample_conversation_object
                assert result.payload == sample_conversation_json

                # Verify database operations
                mock_db_session.add.assert_not_called()
                mock_db_session.commit.assert_called_once()

    @patch("tangerine.models.conversation.db.session")
    def test_upsert_existing_conversation_different_user(
        self, mock_db_session, sample_conversation_json, sample_conversation_object
    ):
        """Test upserting an existing conversation owned by a different user."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = sample_conversation_object

        with patch.object(Conversation, "query", mock_query):
            with patch.object(sample_conversation_object, "is_owned_by", return_value=False):
                result = Conversation.upsert(sample_conversation_json)

                # Verify a new conversation was created
                assert result != sample_conversation_object
                assert result.user_id == "test_user"
                assert result.payload == sample_conversation_json

                # Verify database operations
                mock_db_session.add.assert_called_once_with(result)
                mock_db_session.commit.assert_called_once()

    @patch("tangerine.models.conversation.db.session")
    def test_upsert_with_string_session_id(self, mock_db_session):
        """Test upserting with a string session ID that gets converted to UUID."""
        conversation_json = {
            "sessionId": "12345678-1234-5678-9012-123456789012",
            "user": "test_user",
            "query": "Test query",
        }

        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = None

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.upsert(conversation_json)

            # Verify the session_id was converted to UUID
            assert isinstance(result.session_id, uuid.UUID)
            assert str(result.session_id) == "12345678-1234-5678-9012-123456789012"

    @patch("tangerine.models.conversation.db.session")
    def test_upsert_with_invalid_session_id(self, mock_db_session):
        """Test upserting with an invalid session ID that gets replaced."""
        conversation_json = {
            "sessionId": "invalid-uuid",
            "user": "test_user",
            "query": "Test query",
        }

        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = None

        with patch.object(Conversation, "query", mock_query):
            result = Conversation.upsert(conversation_json)

            # Verify a new UUID was generated
            assert isinstance(result.session_id, uuid.UUID)
            assert str(result.session_id) != "invalid-uuid"

    def test_generate_title_with_query(self):
        """Test title generation with a query."""
        conversation_json = {"query": "This is a very long query that should be truncated"}

        result = Conversation.generate_title(conversation_json)

        assert result == "This is a very long query that should be trunca..."

    def test_generate_title_without_query(self):
        """Test title generation without a query."""
        conversation_json = {"user": "test_user"}

        result = Conversation.generate_title(conversation_json)

        assert result == "Untitled Conversation"

    def test_generate_title_with_empty_query(self):
        """Test title generation with an empty query."""
        conversation_json = {"query": ""}

        result = Conversation.generate_title(conversation_json)

        assert result == "Untitled Conversation"

    def test_from_json_with_string_session_id(self, sample_conversation_json):
        """Test creating a conversation from JSON with string session ID."""
        result = Conversation.from_json(sample_conversation_json)

        assert result.user_id == "test_user"
        assert isinstance(result.session_id, uuid.UUID)
        assert str(result.session_id) == "12345678-1234-5678-9012-123456789012"
        assert result.payload == sample_conversation_json
        assert result.title == "What is AI?..."

    def test_from_json_with_uuid_session_id(self, sample_conversation_json):
        """Test creating a conversation from JSON with UUID session ID."""
        sample_conversation_json["sessionId"] = uuid.UUID("12345678-1234-5678-9012-123456789012")

        result = Conversation.from_json(sample_conversation_json)

        assert result.user_id == "test_user"
        assert isinstance(result.session_id, uuid.UUID)
        assert result.session_id == uuid.UUID("12345678-1234-5678-9012-123456789012")

    def test_copy_with_payload(self, sample_conversation_object):
        """Test copying a conversation with payload."""
        result = sample_conversation_object.copy()

        assert result.id == sample_conversation_object.id
        assert result.user_id == sample_conversation_object.user_id
        assert result.session_id == sample_conversation_object.session_id
        assert result.payload == sample_conversation_object.payload
        assert result.payload is not sample_conversation_object.payload  # Should be a copy

    def test_copy_without_payload(self, sample_conversation_object):
        """Test copying a conversation without payload."""
        sample_conversation_object.payload = None

        result = sample_conversation_object.copy()

        assert result.payload is None

    def test_is_owned_by_true(self, sample_conversation_object):
        """Test ownership check when user owns the conversation."""
        result = sample_conversation_object.is_owned_by("test_user")

        assert result is True

    def test_is_owned_by_false(self, sample_conversation_object):
        """Test ownership check when user doesn't own the conversation."""
        result = sample_conversation_object.is_owned_by("different_user")

        assert result is False

    @patch("tangerine.models.conversation.db.session")
    def test_delete(self, mock_db_session, sample_conversation_object):
        """Test deleting a conversation."""
        sample_conversation_object.delete()

        mock_db_session.delete.assert_called_once_with(sample_conversation_object)
        mock_db_session.commit.assert_called_once()

    @patch("tangerine.models.conversation.db.session")
    def test_delete_by_session_success(self, mock_db_session, sample_conversation_object):
        """Test deleting a conversation by session ID successfully."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = sample_conversation_object

        with patch.object(Conversation, "query", mock_query):
            with patch.object(sample_conversation_object, "is_owned_by", return_value=True):
                with patch.object(sample_conversation_object, "delete") as mock_delete:
                    success, message = Conversation.delete_by_session("session_id", "test_user")

                    assert success is True
                    assert message == "Conversation deleted successfully"
                    mock_delete.assert_called_once()

    @patch("tangerine.models.conversation.db.session")
    def test_delete_by_session_not_found(self, mock_db_session):
        """Test deleting a conversation by session ID when it doesn't exist."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = None

        with patch.object(Conversation, "query", mock_query):
            success, message = Conversation.delete_by_session("nonexistent", "test_user")

            assert success is False
            assert message == "Conversation not found"

    @patch("tangerine.models.conversation.db.session")
    def test_delete_by_session_unauthorized(self, mock_db_session, sample_conversation_object):
        """Test deleting a conversation by session ID when user doesn't own it."""
        mock_query = MagicMock()
        mock_query.filter_by.return_value.first.return_value = sample_conversation_object

        with patch.object(Conversation, "query", mock_query):
            with patch.object(sample_conversation_object, "is_owned_by", return_value=False):
                success, message = Conversation.delete_by_session("session_id", "different_user")

                assert success is False
                assert message == "Unauthorized: You can only delete your own conversations"

    def test_to_json(self, sample_conversation_object):
        """Test converting a conversation to JSON."""
        # Mock the datetime objects
        sample_conversation_object.created_at = Mock()
        sample_conversation_object.created_at.isoformat.return_value = "2023-01-01T00:00:00"
        sample_conversation_object.updated_at = Mock()
        sample_conversation_object.updated_at.isoformat.return_value = "2023-01-01T00:00:00"

        result = sample_conversation_object.to_json()

        expected = {
            "id": str(sample_conversation_object.id),
            "user_id": "test_user",
            "session_id": "12345678-1234-5678-9012-123456789012",
            "created_at": "2023-01-01T00:00:00",
            "updated_at": "2023-01-01T00:00:00",
            "payload": sample_conversation_object.payload,
            "title": "What is AI?...",
        }

        assert result == expected
