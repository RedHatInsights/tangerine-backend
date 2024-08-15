import logging
import re
import string

import html2text
from bs4 import BeautifulSoup
from flask_sqlalchemy import SQLAlchemy
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_text_splitters import (HTMLSectionSplitter,
                                      MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)

import connectors.config as cfg

log = logging.getLogger("tangerine.db")

db = SQLAlchemy()


class Agents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    filenames = db.Column(db.ARRAY(db.String), default=[], nullable=True)

    def __repr__(self):
        return f"<Agents {self.id}>"


class VectorStoreInterface:
    def __init__(self):
        self.store = None
        self.vector_chunk_size = 1024
        self.vector_chunk_overlap = int(self.vector_chunk_size * 0.3)

        self.embeddings = OpenAIEmbeddings(
            model=cfg.EMBED_MODEL_NAME,
            openai_api_base=cfg.EMBED_BASE_URL,
            openai_api_key=cfg.EMBED_API_KEY,
            check_embedding_ctx_length=False,
        )

    def init_vector_store(self):
        try:
            self.store = PGVector(
                collection_name=cfg.VECTOR_COLLECTION_NAME,
                connection=cfg.DB_URI,
                embeddings=self.embeddings,
            )
        except Exception:
            log.exception("error initializing vector store")

    def split_docs_to_chunks(self, documents):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.vector_chunk_size,
            chunk_overlap=self.vector_chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        return text_splitter.split_documents(documents)

    def split_md(self, documents):
        headers_to_split_on = [
            ("#", "h1"),
            ("##", "h2"),
            ("###", "h3"),
        ]
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=headers_to_split_on, strip_headers=False
        )
        new_docs = []
        for doc in documents:
            md_docs = md_splitter.split_text(doc.page_content)
            for md_doc in md_docs:
                # make sure new set of documents contains old doc metadata
                md_doc.metadata.update(doc.metadata)
            new_docs.extend(md_docs)
        return new_docs

    def html_to_md(self, text):
        md = ""
        # parse a page built with mkdocs
        # TODO: possibly handle html pages not built with mkdocs
        soup = BeautifulSoup(text, "lxml")
        # extract 'md-content', this ignores nav/header/footer/etc.
        md_content = soup.find("div", class_="md-content")
        if md_content:
            # remove "Edit this page" button
            edit_button = md_content.find("a", title="Edit this page")
            if edit_button:
                edit_button.decompose()

            h = html2text.HTML2Text()
            h.ignore_images = True
            h.mark_code = True
            h.body_width = 0
            h.ignore_emphasis = True
            h.wrap_links = False
            h.ignore_tables = True

            md = h.handle(str(md_content))

            # remove strange chars (like paragraph markers)
            md = "".join(filter(lambda x: x in string.printable, md))
            # remove number lines created by code blocks
            md = re.sub(r"\[code\](\s+)?(\n\s+\d+)+\n\[\/code\](\s+)?", "", md)
            # remove excessive newlines
            md = re.sub(r"\n\n+", "\n\n", md)
            # replace html2text code block marker with standard md
            md = md.replace("[code]", "```")
            md = md.replace("[/code]", "```")
        else:
            log.error("no 'md-content' div found")

        return md

    def create_documents(self, text, agent_id, source, full_path):
        log.debug("processsing %s", full_path)
        if full_path.lower().endswith(".html"):
            text = self.html_to_md(text)

        if not text:
            raise ValueError("no document text provided")

        documents = [
            Document(
                page_content=text,
                metadata={"agent_id": str(agent_id), "source": source, "full_path": full_path},
            )
        ]

        chunked_docs = self.split_docs_to_chunks(documents)
        return chunked_docs

    def add_document(self, text, agent_id, source, full_path):
        try:
            self.store.add_documents(self.create_documents(text, agent_id, source, full_path))
        except Exception:
            log.exception("error adding documents")

    def search(self, query, agent_id):
        results = self.store.max_marginal_relevance_search(
            query=query, filter={"agent_id": str(agent_id)}, k=6
        )

        return results

    def delete_documents(self, ids):
        self.store.delete(ids)


vector_interface = VectorStoreInterface()
