import logging
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID

from .agent import db

log = logging.getLogger("tangerine.db.interactions")


class RelevanceScore(db.Model):
    __tablename__ = "relevance_scores"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    question_uuid = db.Column(UUID(as_uuid=True), db.ForeignKey("interactions.question_uuid"))
    retrieval_method = db.Column(db.String(50), nullable=False)
    score = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    def __init__(self, question_uuid, retrieval_method, score):
        self.question_uuid = question_uuid
        self.retrieval_method = retrieval_method
        self.score = score
        self.timestamp = db.func.current_timestamp()


class QuestionEmbedding(db.Model):
    __tablename__ = "question_embeddings"

    question_uuid = db.Column(
        UUID(as_uuid=True), db.ForeignKey("interactions.question_uuid"), primary_key=True
    )
    question_embedding = db.Column(Vector(768), nullable=False)


class Interaction(db.Model):
    __tablename__ = "interactions"

    id = db.Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_uuid = db.Column(db.String(36))
    user_query = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text)
    source_doc_chunks = db.Column(db.JSON)
    user_feedback = db.Column(db.String(20))
    feedback_comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())


def store_interaction(
    user_query,
    llm_response,
    source_doc_chunks,
    question_embedding,
    session_uuid=None,
    user_feedback=None,
    feedback_comment=None,
):
    """
    Logs a RAG interaction and its question embedding into the database.

    Args:
        user_query (str): The user's natural language question.
        llm_response (str): The LLM-generated response.
        source_doc_chunks (list[dict]): Retrieved document chunks (list of dicts).
        relevance_scores (list[float]): Relevance scores corresponding to the chunks.
        question_embedding (list[float]): The vector embedding of the question.
        session_uuid (str, optional): Session UUID if available. Auto-generated if not provided.
        user_feedback (str, optional): 'thumbs_up', 'thumbs_down', 'neutral', or None.
        feedback_comment (str, optional): Optional free-text feedback from the user.
    """
    session_uuid = session_uuid or str(uuid.uuid4())

    # Create interaction record
    interaction = Interaction(
        session_uuid=session_uuid,
        user_query=user_query,
        llm_response=llm_response,
        source_doc_chunks=source_doc_chunks,
        user_feedback=user_feedback,
        feedback_comment=feedback_comment,
    )

    # Create embedding record
    embedding_record = QuestionEmbedding(
        question_uuid=interaction.id,
        question_embedding=question_embedding,
    )

    # Create relevance scores
    relevance_scores = []
    for chunk in source_doc_chunks:
        retrieval_method = chunk.get("retrieval_method", "unknown")
        score = chunk.get("score", 0.0)
        relevance_score = RelevanceScore(
            question_uuid=interaction.id,
            retrieval_method=retrieval_method,
            score=score,
        )
        relevance_scores.append(relevance_score)
        db.session.add(relevance_score)

    # Commit both within a transaction
    try:
        db.session.add(interaction)
        db.session.add(embedding_record)
        db.session.commit()
        log.info("Interaction and embedding logged successfully.")
        return interaction.id
    except Exception as e:
        db.session.rollback()
        log.error("Error logging interaction or embedding", exc_info=True)
        raise e
