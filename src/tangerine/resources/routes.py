from .assistant import (
    AssistantAdvancedChatApi,
    AssistantApi,
    AssistantChatApi,
    AssistantDefaultsApi,
    AssistantDocuments,
    AssistantsApi,
    AssistantSearchApi,
)
from .conversation import (
    ConversationDeleteApi,
    ConversationListApi,
    ConversationRetrievalApi,
    ConversationUpsertApi,
)
from .feedback import FeedbackApi
from .ping import PingApi


def initialize_routes(api):
    api.add_resource(AssistantDefaultsApi, "/api/assistantDefaults")
    api.add_resource(AssistantsApi, "/api/assistants")
    api.add_resource(AssistantApi, "/api/assistants/<id>")
    api.add_resource(AssistantDocuments, "/api/assistants/<id>/documents")
    api.add_resource(AssistantChatApi, "/api/assistants/<id>/chat", methods=["POST"])
    api.add_resource(AssistantAdvancedChatApi, "/api/assistants/chat", methods=["POST"])
    api.add_resource(PingApi, "/ping", methods=["GET"])
    api.add_resource(FeedbackApi, "/api/feedback", methods=["POST"])
    api.add_resource(AssistantSearchApi, "/api/assistants/<id>/search", methods=["POST"])
    api.add_resource(ConversationListApi, "/api/conversations/list", methods=["POST"])
    api.add_resource(ConversationRetrievalApi, "/api/conversations/load", methods=["POST"])
    api.add_resource(ConversationUpsertApi, "/api/conversations/upsert", methods=["POST"])
    api.add_resource(ConversationDeleteApi, "/api/conversations/delete", methods=["POST"])
