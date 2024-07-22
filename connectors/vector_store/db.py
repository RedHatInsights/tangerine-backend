from flask_sqlalchemy import SQLAlchemy
from langchain_community.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.pgvector import PGVector
from langchain.text_splitter import RecursiveCharacterTextSplitter
import connectors.config as cfg

db = SQLAlchemy()

class Agents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    filenames = db.Column(db.ARRAY(db.String), default=[], nullable=True)

    def __repr__(self):
        return f'<Agents {self.id}>'


class VectorStoreInterface():
    def __init__(self):
        self.store = None
        self.vector_chunk_size = 2000
        self.vector_chunk_overlap = 500

        if cfg.EMBEDDING_MODEL_SOURCE == "openai":
            self.embeddings = OpenAIEmbeddings(
                model=cfg.EMBEDDING_MODEL_NAME,
                openai_api_base=cfg.OPENAI_BASE_URL,
                openai_api_key=cfg.OPENAI_API_KEY
            )
        elif cfg.EMBEDDING_MODEL_SOURCE == "local":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=cfg.EMBEDDING_MODEL_NAME,
                model_kwargs={'trust_remote_code': cfg.TRUST_REMOTE_CODE}
            )

    def init_vector_store(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection_string=cfg.DB_URI,
                embedding_function=self.embeddings,
            )
        except Exception as e:
            print(f"Error init_vector_store: {e}")
        return

    def add_document(self, text, agent_id, filename):
        documents = [Document(page_content=text, metadata={"agent_id": agent_id, "filename": filename})]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        docs = text_splitter.split_documents(documents)

        try:
            self.store.add_documents(docs)
        except Exception as e:
            print(f"Error adding_documents: {e}")

        return

    def search(self, query, agent_id):
        docs_with_score = self.store.max_marginal_relevance_search_with_score(query=query, filter={"agent_id": agent_id}, k=2)
        return docs_with_score      # list(int, Document(page_content, metadata))


vector_interface = VectorStoreInterface()
