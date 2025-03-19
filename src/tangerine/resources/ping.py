from flask_restful import Resource

from ..metrics import metrics


class PingApi(Resource):
    @metrics.do_not_track()
    def get(self):
        return {"ping": "pong."}
