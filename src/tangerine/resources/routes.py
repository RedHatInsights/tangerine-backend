from .assistant import (
    AssistantAdvancedChatApi,
    AssistantApi,
    AssistantChatApi,
    AssistantDefaultsApi,
    AssistantKnowledgeBasesApi,
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
from .knowledgebase import KnowledgeBaseApi, KnowledgeBaseDocuments, KnowledgeBasesApi
from .ping import PingApi


def initialize_routes(api):
    api.add_resource(AssistantDefaultsApi, "/api/assistantDefaults")
    api.add_resource(AssistantsApi, "/api/assistants")
    api.add_resource(AssistantApi, "/api/assistants/<id>")
    api.add_resource(AssistantKnowledgeBasesApi, "/api/assistants/<id>/knowledgebases")
    api.add_resource(AssistantChatApi, "/api/assistants/<id>/chat", methods=["POST"])
    api.add_resource(AssistantAdvancedChatApi, "/api/assistants/chat", methods=["POST"])
    api.add_resource(AssistantSearchApi, "/api/assistants/<id>/search", methods=["POST"])

    # KnowledgeBase routes
    api.add_resource(KnowledgeBasesApi, "/api/knowledgebases")
    api.add_resource(KnowledgeBaseApi, "/api/knowledgebases/<id>")
    api.add_resource(KnowledgeBaseDocuments, "/api/knowledgebases/<id>/documents")

    # Other routes
    api.add_resource(PingApi, "/ping", methods=["GET"])
    api.add_resource(FeedbackApi, "/api/feedback", methods=["POST"])
    api.add_resource(ConversationListApi, "/api/conversations/list", methods=["POST"])
    api.add_resource(ConversationRetrievalApi, "/api/conversations/load", methods=["POST"])
    api.add_resource(ConversationUpsertApi, "/api/conversations/upsert", methods=["POST"])
    api.add_resource(ConversationDeleteApi, "/api/conversations/delete", methods=["POST"])
