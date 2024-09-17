import re
from io import StringIO
from typing import List, Optional

import PyPDF2

from .agent import Agent
from .vector import vector_db


def validate_file_path(full_path: str) -> None:
    # intentionally more restrictive, matches a "typical" unix path and filename with extension
    file_regex = r"^[\w\-.\/ ]+\/?\.[\w\-. ]+[^.]$"
    if not full_path or not full_path.strip() or not re.match(file_regex, full_path):
        raise ValueError(f"file path must match regex: {file_regex}")


def validate_source(source: str) -> None:
    source_regex = r"^[\w-]+$"
    if not source or not source.strip() or not re.match(source_regex, source):
        raise ValueError(f"source must match regex: {source_regex}")


def validate_file_type(full_path: str) -> None:
    if not any(
        [full_path.endswith(filetype) for filetype in [".txt", ".pdf", ".md", ".rst", ".html"]]
    ):
        raise ValueError("unsupported file type")


class File:
    def __init__(self, source: str, full_path: str, content: Optional[str] = ""):
        self.source = source
        self.full_path = full_path
        self.content = content

    def validate(self):
        validate_file_path(self.full_path)
        validate_file_type(self.full_path)
        validate_source(self.source)

    @property
    def display_name(self) -> str:
        return f"{self.source}:{self.full_path}"

    def extract_text(self):
        if self.full_path.endswith(".pdf"):
            pdf_reader = PyPDF2.PdfReader(StringIO(self.content))

            text_content = ""
            # iterate through each page in the PDF and extract text
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text_content += page.extract_text()

            return text_content

        if self.full_path.endswith((".md", ".txt", ".rst", ".html")):
            return self.content

        return ""


def embed_files(files: List[File], agent: Agent) -> None:
    for file in files:
        file.validate()
        extracted_text = file.extract_text()
        # Only generate embeddings when there is actual text
        if len(extracted_text) > 0:
            vector_db.add_document(extracted_text, agent.id, file.source, file.full_path)


def add_files_to_agent(files: List[File], agent: Agent) -> None:
    agent.add_files([file.display_name for file in files])


def remove_files(agent: Agent, metadata: dict) -> List[str]:
    metadata["agent_id"] = str(agent.id)
    if "full_path" in metadata:
        validate_file_path(metadata["full_path"])
    if "source" in metadata:
        validate_source(metadata["source"])

    # first delete docs from vector store, get metadata back for deleted files
    deleted_doc_metadatas = vector_db.delete_documents_by_metadata(metadata)

    # delete from agent DB
    file_display_names = set(
        [
            File(source=metadata["source"], full_path=metadata["full_path"]).display_name
            for metadata in deleted_doc_metadatas
        ]
    )
    agent.remove_files(file_display_names)

    return list(file_display_names)
