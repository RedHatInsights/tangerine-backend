import logging
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

from .agent import db

log = logging.getLogger("tangerine.db.interactions")


class RelevanceScore(db.Model):
    __tablename__ = "relevance_scores"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    interaction_id = db.Column(UUID(as_uuid=True), db.ForeignKey("interactions.id"))
    retrieval_method = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, interaction_id, retrieval_method, score):
        self.interaction_id = interaction_id
        self.retrieval_method = retrieval_method
        self.score = score
        self.timestamp = db.func.current_timestamp()


class QuestionEmbedding(db.Model):
    __tablename__ = "question_embeddings"

    interaction_id = db.Column(
        UUID(as_uuid=True), db.ForeignKey("interactions.id"), primary_key=True
    )
    question_embedding = db.Column(Vector(768), nullable=False)

class UserFeedback(db.Model):
    __tablename__ = "user_feedback"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    interaction_id = db.Column(UUID(as_uuid=True), db.ForeignKey("interactions.id"))
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    like = db.Column(db.Boolean, nullable=False)
    dislike = db.Column(db.Boolean, nullable=False)
    feedback = db.Column(db.Text, nullable=False)
    def __init__(self, interaction_id, like, dislike, feedback):
        self.interaction_id = interaction_id
        self.like = like
        self.dislike = dislike
        self.feedback = feedback
        self.timestamp = db.func.current_timestamp()

def store_user_feedback(
    interaction_id,
    like,
    dislike,
    feedback,
):
    """
    Logs user feedback into the database.

    Args:
        interaction_id (str): The ID of the interaction being rated.
        like (bool): Whether the user liked the response.
        dislike (bool): Whether the user disliked the response.
        feedback (str): Additional feedback from the user.
    """
    # Create feedback record
    feedback_record = UserFeedback(
        interaction_id=interaction_id,
        like=like,
        dislike=dislike,
        feedback=feedback,
    )
    insert(feedback_record, "Feedback")
    return feedback_record.id

class Interaction(db.Model):
    __tablename__ = "interactions"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_uuid = db.Column(db.String(36))
    question = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text)
    source_doc_chunks = db.Column(db.JSON)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    client = db.Column(db.String(50), nullable=True)


def insert(model, name="DB Model"):
    try:
        db.session.add(model)
        db.session.commit()
        db.session.refresh(model)
    except Exception:
        db.session.rollback()
        log.exception("Error logging %s", name)
        raise
    return model


def store_interaction(
    question,
    llm_response,
    source_doc_chunks,
    question_embedding,
    session_uuid=None,
    interaction_id=None,
    client=None,
):
    """
    Logs a RAG interaction and its question embedding into the database.

    Args:
        question (str): The user's natural language question.
        llm_response (str): The LLM-generated response.
        source_doc_chunks (list[dict]): Retrieved document chunks (list of dicts).
        relevance_scores (list[float]): Relevance scores corresponding to the chunks.
        question_embedding (list[float]): The vector embedding of the question.
        session_uuid (str, optional): Session UUID if available. Auto-generated if not provided.
    """
    session_uuid = session_uuid or str(uuid.uuid4())
    interaction_id = interaction_id or str(uuid.uuid4())

    # Create interaction record
    interaction = Interaction(
        id=interaction_id,
        session_uuid=session_uuid,
        question=question,
        llm_response=llm_response,
        source_doc_chunks=source_doc_chunks,
        client=client,
    )
    interaction = insert(interaction, "Interaction")

    # Create embedding record
    embedding_record = QuestionEmbedding(
        interaction_id=interaction.id,
        question_embedding=question_embedding,
    )
    insert(embedding_record, "Question Embedding")

    for chunk in source_doc_chunks:
        retrieval_method = chunk.get("retrieval_method", "unknown")
        score = chunk.get("score", 0.0)
        relevance_score = RelevanceScore(
            interaction_id=interaction.id,
            retrieval_method=retrieval_method,
            score=score,
        )
        insert(relevance_score, "Relevance Score")

    return interaction.id
