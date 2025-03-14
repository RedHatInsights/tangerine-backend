from .assistant import assistantApi, assistantChatApi, assistantDefaultsApi, assistantDocuments, assistantsApi
from .feedback import FeedbackApi
from .utils import PingApi


def initialize_routes(api):
    api.add_resource(assistantDefaultsApi, "/api/assistantDefaults")
    api.add_resource(assistantsApi, "/api/assistants")
    api.add_resource(assistantApi, "/api/assistants/<id>")
    api.add_resource(assistantDocuments, "/api/assistants/<id>/documents")
    api.add_resource(assistantChatApi, "/api/assistants/<id>/chat", methods=["GET", "POST"])
    api.add_resource(PingApi, "/ping", methods=["GET"])
    api.add_resource(FeedbackApi, "/api/feedback", methods=["POST"])
