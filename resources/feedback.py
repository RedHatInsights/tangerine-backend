import logging

from flask import request
from flask_restful import Resource

from connectors.db.interactions import store_user_feedback
from connectors import config

log = logging.getLogger("tangerine")


class FeedbackApi(Resource):
    def post(self):
        if config.STORE_INTERACTIONS is False:
            return {"message": "feedback is disabled"}, 400
        interaction_id = request.json.get("interactionId")
        like = request.json.get("like")
        dislike = request.json.get("dislike")
        feedback = request.json.get("feedback")

        if not interaction_id:
            return {"message": "interaction_id is required"}, 400
        if like is None and dislike is None:
            return {"message": "either 'like' or 'dislike' is required"}, 400
        if like is True and dislike is True:
            return {"message": "cannot like and dislike at the same time"}, 400

        try:
            # Store the feedback in the database
            store_user_feedback(interaction_id, like, dislike, feedback)
        except Exception:
            log.exception("Error storing feedback")
            return {"message": "error storing feedback"}, 500

        return {"message": "feedback stored successfully"}, 201
