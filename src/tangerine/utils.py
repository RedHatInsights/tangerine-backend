from typing import List

from .file import File, validate_file_path, validate_source
from .models.agent import Agent
from .vector import vector_db


def embed_files(files: List[File], agent: Agent) -> None:
    for file in files:
        file.validate()
        vector_db.add_file(file, agent.id)


def add_filenames_to_agent(files: List[File], agent: Agent) -> None:
    agent.add_files([file.display_name for file in files])


def remove_files(agent: Agent, metadata: dict) -> List[str]:
    metadata["agent_id"] = agent.id
    if "full_path" in metadata:
        validate_file_path(metadata["full_path"])
    if "source" in metadata:
        validate_source(metadata["source"])

    # first delete docs from vector store, get metadata back for deleted files
    deleted_doc_metadatas = vector_db.delete_document_chunks(metadata)

    # delete from agent DB
    file_display_names = set(
        [
            File(source=metadata["source"], full_path=metadata["full_path"]).display_name
            for metadata in deleted_doc_metadatas
        ]
    )
    agent.remove_files(file_display_names)

    return list(file_display_names)
