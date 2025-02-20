from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import UUID
import logging
from flask_sqlalchemy import SQLAlchemy
from connectors.config import SQLALCHEMY_MAX_OVERFLOW, SQLALCHEMY_POOL_SIZE
import uuid

db = SQLAlchemy(
    engine_options={"pool_size": SQLALCHEMY_POOL_SIZE, "max_overflow": SQLALCHEMY_MAX_OVERFLOW}
)

log = logging.getLogger("tangerine.db.agent")

class QuestionEmbedding(db.Model):
    __tablename__ = 'question_embeddings'

    question_uuid = db.Column(UUID(as_uuid=True), db.ForeignKey('rag_interactions.question_uuid'), primary_key=True)
    question_embedding = db.Column(Vector(1536), nullable=False)  
    
class Interaction(db.Model):
    __tablename__ = 'interactions'

    question_uuid = db.Column(UUID(as_uuid=True), primary_key=True)
    session_uuid = db.Column(db.String(36))
    user_query = db.Column(db.Text, nullable=False)
    llm_response = db.Column(db.Text)
    source_doc_chunks = db.Column(db.JSON)
    relevance_scores = db.Column(db.JSON)
    user_feedback = db.Column(db.String(20))
    feedback_comment = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())
    
class InteractionLogger:
    @staticmethod
    def log_interaction(
        user_query,
        llm_response,
        source_doc_chunks,
        relevance_scores,
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
        question_uuid = uuid.uuid4()
        session_uuid = session_uuid or str(uuid.uuid4())

        # Create interaction record
        interaction = Interaction(
            question_uuid=question_uuid,
            session_uuid=session_uuid,
            user_query=user_query,
            llm_response=llm_response,
            source_doc_chunks=source_doc_chunks,
            relevance_scores=relevance_scores,
            user_feedback=user_feedback,
            feedback_comment=feedback_comment,
        )

        # Create embedding record
        embedding_record = QuestionEmbedding(
            question_uuid=question_uuid,
            question_embedding=question_embedding,
        )

        # Commit both within a transaction
        try:
            db.session.add(interaction)
            db.session.add(embedding_record)
            db.session.commit()
            log.info("Interaction and embedding logged successfully.")
            return question_uuid  # Return the UUID in case you need it later
        except Exception as e:
            db.session.rollback()
            log.error("Error logging interaction or embedding", exc_info=True)
            raise e
