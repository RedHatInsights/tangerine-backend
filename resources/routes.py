from .agent import (AgentApi, AgentChatApi, AgentDefaultsApi, AgentDocuments,
                    AgentsApi)
from .utils import PingApi


def initialize_routes(api):
    api.add_resource(AgentDefaultsApi, "/api/agentDefaults")
    api.add_resource(AgentsApi, "/api/agents")
    api.add_resource(AgentApi, "/api/agents/<id>")
    api.add_resource(AgentDocuments, "/api/agents/<id>/documents")
    api.add_resource(AgentChatApi, "/api/agents/<id>/chat", methods=["GET", "POST"])
    api.add_resource(PingApi, "/ping", methods=["GET"])
