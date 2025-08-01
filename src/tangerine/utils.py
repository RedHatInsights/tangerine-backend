from typing import List

from .file import File, validate_file_path, validate_source
from .vector import vector_db


def embed_files_for_knowledgebase(files: List[File], knowledgebase_id: int) -> None:
    for file in files:
        file.validate()
        vector_db.add_file(file, knowledgebase_id)


def remove_files_from_knowledgebase(knowledgebase, metadata: dict) -> List[str]:
    metadata["knowledgebase_id"] = str(knowledgebase.id)
    if "full_path" in metadata:
        validate_file_path(metadata["full_path"])
    if "source" in metadata:
        validate_source(metadata["source"])

    # first delete docs from vector store, get metadata back for deleted files
    deleted_doc_metadatas = vector_db.delete_document_chunks(metadata)

    # delete from knowledgebase DB
    file_display_names = set(
        [
            File(source=doc_metadata["source"], full_path=doc_metadata["full_path"]).display_name
            for doc_metadata in deleted_doc_metadatas
        ]
    )
    knowledgebase.remove_files(file_display_names)

    return list(file_display_names)
