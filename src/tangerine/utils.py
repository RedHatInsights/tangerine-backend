from typing import List

from .file import File, validate_file_path, validate_source
from .vector import vector_db


def embed_files_for_knowledgebase(files: List[File], knowledgebase_id: int) -> None:
    for file in files:
        file.validate()
        vector_db.add_file(file, knowledgebase_id)


def get_files_for_knowledgebase(knowledgebase_id: int) -> List[str]:
    """
    Get unique list of all file display names for a knowledgebase by querying vector database.

    Returns display names in format 'source:full_path' as determined by File.display_name property.
    """
    search_filter = {"knowledgebase_id": str(knowledgebase_id)}

    # Get all unique metadata entries for this knowledgebase
    unique_metadatas = vector_db.get_distinct_cmetadata(search_filter)

    # Extract unique display names from metadata
    display_names = set()
    for metadata in unique_metadatas:
        if "source" in metadata and "full_path" in metadata:
            # Create display name in same format as File.display_name
            display_name = f"{metadata['source']}:{metadata['full_path']}"
            display_names.add(display_name)

    return sorted(list(display_names))


def remove_files_from_knowledgebase(knowledgebase, metadata: dict) -> List[str]:
    metadata["knowledgebase_id"] = str(knowledgebase.id)
    if "full_path" in metadata:
        validate_file_path(metadata["full_path"])
    if "source" in metadata:
        validate_source(metadata["source"])

    # Delete docs from vector store, get metadata back for deleted files
    deleted_doc_metadatas = vector_db.delete_document_chunks(metadata)

    file_display_names = [
        File(source=doc_metadata["source"], full_path=doc_metadata["full_path"]).display_name
        for doc_metadata in deleted_doc_metadatas
    ]

    return file_display_names
