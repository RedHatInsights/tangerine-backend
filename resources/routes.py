from .agent import AgentsApi, AgentApi, AgentChatApi, AgentDocUpload
from .utils import PingApi

def initialize_routes(api):
    api.add_resource(AgentsApi, '/agents')
    api.add_resource(AgentApi, '/agents/<id>')
    api.add_resource(AgentDocUpload, '/agents/<id>/document_upload')
    api.add_resource(AgentChatApi, '/agents/<id>/chat', methods=['GET'])
    api.add_resource(PingApi, '/ping', methods=['GET'])
