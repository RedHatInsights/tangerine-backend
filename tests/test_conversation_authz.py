"""Regression tests for object-level authorization on /api/conversations/* endpoints.

Identity must be derived from the gateway-injected USER_IDENTITY_HEADER, never from
the JSON request body. These tests stub Flask's request and the model layer so they
exercise the authz logic in isolation.
"""

from types import SimpleNamespace
from unittest.mock import patch

import tangerine.config as cfg
from tangerine.resources import conversation as conv


def _fake_request(headers=None, json_body=None):
    return SimpleNamespace(
        headers=headers or {},
        get_json=lambda: json_body,
    )


def test_list_ignores_body_user_id_and_uses_header():
    """ConversationListApi must scope to the header principal, not body user_id."""
    seen = {}

    def fake_get_by_user(uid):
        seen["uid"] = uid
        return []

    req = _fake_request(
        headers={cfg.USER_IDENTITY_HEADER: "alice"},
        json_body={"user_id": "victim-bob"},
    )
    with patch.object(conv, "request", req), patch.object(
        conv.Conversation, "get_by_user", staticmethod(fake_get_by_user)
    ):
        body, status = conv.ConversationListApi().post()
    assert status == 200
    assert seen["uid"] == "alice"  # NOT "victim-bob"


def test_list_rejects_missing_identity_header():
    req = _fake_request(headers={}, json_body={"user_id": "victim-bob"})
    with patch.object(conv, "request", req):
        body, status = conv.ConversationListApi().post()
    assert status == 401


def test_retrieval_enforces_ownership():
    """ConversationRetrievalApi must 403 when the session belongs to another user."""
    other = SimpleNamespace(
        is_owned_by=lambda uid: uid == "bob",
        to_json=lambda: {"session_id": "s"},
    )
    req = _fake_request(
        headers={cfg.USER_IDENTITY_HEADER: "alice"},
        json_body={"sessionId": "11111111-1111-1111-1111-111111111111"},
    )
    with patch.object(conv, "request", req), patch.object(
        conv.Conversation, "get_by_session", staticmethod(lambda sid: other)
    ):
        body, status = conv.ConversationRetrievalApi().post()
    assert status == 403


def test_upsert_overrides_body_user_with_header_principal():
    captured = {}

    def fake_upsert(data):
        captured.update(data)
        return SimpleNamespace(to_json=lambda: {})

    req = _fake_request(
        headers={cfg.USER_IDENTITY_HEADER: "alice"},
        json_body={"user": "anonymous", "sessionId": "s", "prevMsgs": []},
    )
    with patch.object(conv, "request", req), patch.object(
        conv.Conversation, "upsert", staticmethod(fake_upsert)
    ):
        body, status = conv.ConversationUpsertApi().post()
    assert status == 200
    assert captured["user"] == "alice"  # 'anonymous' bypass closed


def test_delete_uses_header_principal_not_body():
    seen = {}

    def fake_delete(session_id, user_id):
        seen["uid"] = user_id
        return True, "Conversation deleted successfully"

    req = _fake_request(
        headers={cfg.USER_IDENTITY_HEADER: "alice"},
        json_body={"sessionId": "s", "user_id": "victim-bob"},
    )
    with patch.object(conv, "request", req), patch.object(
        conv.Conversation, "delete_by_session", staticmethod(fake_delete)
    ):
        body, status = conv.ConversationDeleteApi().post()
    assert status == 200
    assert seen["uid"] == "alice"
