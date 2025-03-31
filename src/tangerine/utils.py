from typing import List

from .file import File, validate_file_path, validate_source
from .models.assistant import Assistant
from .vector import vector_db


def embed_files(files: List[File], assistant: Assistant) -> None:
    for file in files:
        file.validate()
        vector_db.add_file(file, assistant.id)


def add_filenames_to_assistant(files: List[File], assistant: Assistant) -> None:
    assistant.add_files([file.display_name for file in files])


def remove_files(assistant: Assistant, metadata: dict) -> List[str]:
    metadata["assistant_id"] = assistant.id
    if "full_path" in metadata:
        validate_file_path(metadata["full_path"])
    if "source" in metadata:
        validate_source(metadata["source"])

    # first delete docs from vector store, get metadata back for deleted files
    deleted_doc_metadatas = vector_db.delete_document_chunks(metadata)

    # delete from assistant DB
    file_display_names = set(
        [
            File(source=metadata["source"], full_path=metadata["full_path"]).display_name
            for metadata in deleted_doc_metadatas
        ]
    )
    assistant.remove_files(file_display_names)

    return list(file_display_names)
