import uuid
from flask_sqlalchemy import SQLAlchemy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

        self.embeddings = OpenAIEmbeddings(
            model=cfg.EMBED_MODEL_NAME,
            openai_api_base=cfg.EMBED_BASE_URL,
            openai_api_key=cfg.EMBED_API_KEY,
            check_embedding_ctx_length=False
        )

    def init_vector_store(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection=cfg.DB_URI,
                embeddings=self.embeddings,
            )
        except Exception as e:
            print(f"Error init_vector_store: {e}")
        return

    def add_document(self, text, agent_id, repo, path, filename):
        documents = [Document(page_content=text, metadata={"agent_id": str(agent_id), "repo": repo, "path": path, "filename": filename})]
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            length_function=len,
            is_separator_regex=False
        )
        docs = text_splitter.split_documents(documents)

        try:
            #texts = [doc.page_content for doc in docs]
            #embeddings = self.embeddings.embed_documents(list(texts))
            #self.store.add_embeddings(texts=texts, embeddings=embeddings, metadatas=None, ids=None)
            self.store.add_documents(docs)
        except Exception as e:
            print(f"Error adding_documents: {e}")

        return

    def search(self, query, agent_id):
        docs_with_score = self.store.max_marginal_relevance_search_with_score(query=query, filter={"agent_id": agent_id}, k=4)
        return docs_with_score      # list(int, Document(page_content, metadata))
    
    def delete_documents(self, ids):
        self.store.delete(ids)


vector_interface = VectorStoreInterface()
