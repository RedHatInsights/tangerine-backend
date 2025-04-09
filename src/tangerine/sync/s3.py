import logging
import tempfile
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator, List, Optional

import boto3
import jinja2
import yaml
from flask import current_app
from pydantic import BaseModel
from sqlalchemy import text

import tangerine.config as cfg
from tangerine.db import db
from tangerine.models import Assistant
from tangerine.utils import File, embed_files
from tangerine.vector import vector_db

s3 = boto3.client("s3")

log = logging.getLogger("tangerine.s3sync")


class PathConfig(BaseModel):
    prefix: str
    citation_url_template: Optional[str] = None
    extensions: Optional[List[str]] = None


class AssistantConfig(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    bucket: str
    paths: List[PathConfig]


class SyncConfigDefaults(BaseModel):
    extensions: List[str]
    citation_url_template: str


class SyncConfig(BaseModel):
    defaults: SyncConfigDefaults
    assistants: List[AssistantConfig]


def get_all_s3_objects(bucket: str, prefix: str) -> List:
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        if "Contents" in page:
            objects.extend(page["Contents"])

    return objects


def get_sync_config() -> SyncConfig:
    with open(cfg.S3_SYNC_CONFIG_FILE) as fp:
        data = yaml.safe_load(fp)
        sync_config = SyncConfig(**data)

    return sync_config


def download_obj(bucket: str, obj_key: str, dest_dir: str):
    """Downloads an object from S3 to dest dir."""
    dest_path = Path(dest_dir)
    download_path = dest_path / obj_key

    # create directory tree for this file
    download_path.parents[0].mkdir(parents=True, exist_ok=True)

    log.debug("downloading %s to %s", obj_key, download_path)
    s3.download_file(bucket, obj_key, str(download_path))


def download_objs_concurrent(bucket: str, files: List[File], dest_dir: str) -> Iterator[bool]:
    keys = [file.full_path for file in files]
    log.debug("downloading %d files from s3 bucket '%s' to %s", len(keys), bucket, dest_dir)
    with ThreadPoolExecutor() as executor:
        key_for_future = {executor.submit(download_obj, bucket, key, dest_dir): key for key in keys}

        for future in futures.as_completed(key_for_future):
            key = key_for_future[future]
            try:
                future.result()
                log.info("download for %s: success", key)
                yield True
            except Exception as err:
                log.error("download for %s hit error: %s", key, err)
                yield False


def embed_file(app_context, file: File, tmpdir: str, assistant_id: int) -> File:
    """Adds an s3 object stored locally to assistant"""
    app_context.push()

    log.debug("embedding file %s", file.full_path)

    with db.session():
        assistant = Assistant.get(assistant_id)
        path_on_disk = Path(tmpdir) / Path(file.full_path)

        with open(path_on_disk, "r") as fp:
            # add new files as active=False until all embedding was successful
            file.content = fp.read()
            file.active = False
            file.pending_removal = False

        embed_files([file], assistant)
        return file


def embed_files_concurrent(
    bucket: str, files: List[File], tmpdir: str, assistant_id: int
) -> Iterator[Optional[File]]:
    with ThreadPoolExecutor(max_workers=cfg.S3_SYNC_POOL_SIZE) as executor:
        key_for_future = {
            executor.submit(
                embed_file, current_app.app_context(), file, tmpdir, assistant_id
            ): file.full_path
            for file in files
        }

        for future in futures.as_completed(key_for_future):
            key = key_for_future[future]
            try:
                file = future.result()
                log.info("create embeddings for %s: success", key)
                yield file
            except Exception as err:
                log.error("hit error creating embeddings for %s: %s", key, err)
                yield None


def get_file_list(assistant_config: AssistantConfig, defaults: SyncConfigDefaults) -> List[File]:
    files = []

    bucket = assistant_config.bucket

    for path_config in assistant_config.paths:
        prefix = path_config.prefix
        log.debug("fetching objects from bucket %s at prefix %s", bucket, prefix)
        objects = get_all_s3_objects(bucket, path_config.prefix)
        log.debug("%d objects found in bucket %s at prefix %s", len(objects), bucket, prefix)
        for obj in objects:
            full_path = obj["Key"]

            # check if this file extension matches any of the desired extensions
            if not path_config.extensions:
                path_config.extensions = defaults.extensions
            if not any([full_path.endswith(f".{ext}") for ext in path_config.extensions]):
                continue

            # generate citation URL for this file
            if not path_config.citation_url_template:
                path_config.citation_url_template = defaults.citation_url_template
            template = jinja2.Template(path_config.citation_url_template)
            citation_url = template.render(full_path=full_path)

            file = File(
                source=f"s3-{bucket}",
                full_path=full_path,
                active=False,
                pending_removal=False,
                hash=obj["ETag"],
                citation_url=citation_url,
                content="",  # content will be populated later, after downloading the file
            )
            files.append(file)

    log.debug(
        "%d total s3 objects left after filtering for extensions %s",
        len(files),
        path_config.extensions,
    )

    return files


def _get_new_files_to_add(files_by_key, assistant_objects_by_path, resync):
    files_to_add = []

    for key, file in files_by_key.items():
        add = False

        # if 'resync' is true, we are adding all of them
        if resync:
            log.debug("resync: %s to be added", key)
            add = True

        # check if there's a new remote file to add
        elif key not in assistant_objects_by_path:
            log.debug("%s is new in s3, will add file", key)
            add = True

        if add:
            files_to_add.append(file)

    return files_to_add


def compare_files(
    assistant_config: AssistantConfig,
    assistant: Assistant,
    defaults: SyncConfigDefaults,
    resync: bool,
) -> tuple[List[dict], List[File], set[dict], int, int, int]:
    files = get_file_list(assistant_config, defaults)

    # collect all unique file objects currently stored for this assistant in the DB
    assistant_objects = vector_db.get_distinct_cmetadata(
        search_filter={"assistant_id": assistant.id}
    )

    # group by keys for easier comparisons
    files_by_key = {file.full_path: file for file in files}
    assistant_objects_by_path = {obj["full_path"]: obj for obj in assistant_objects}

    assistant_objects_to_delete = []
    files_to_insert = []
    metadata_update_args = []

    num_to_add = 0
    num_to_delete = 0
    num_to_update = 0

    for assistant_object in assistant_objects:
        full_path = assistant_object["full_path"]

        # if 'resync' is true, we are deleting all files
        if resync:
            log.debug("%s removing for resync", full_path)
            assistant_objects_to_delete.append(assistant_object)
            num_to_delete += 1
            continue

        # check if the entire prefix is no longer defined in the assistant config
        prefixes = [path_config.prefix for path_config in assistant_config.paths]
        if not any([full_path.startswith(prefix) for prefix in prefixes]):
            log.debug("%s uses prefix not found in assistant config, will remove file", full_path)
            assistant_objects_to_delete.append(assistant_object)
            num_to_delete += 1
            continue

        # check if remote file has been removed
        if full_path not in files_by_key:
            # stored file is not present in s3, mark for deletion
            log.debug("%s no longer present in s3, will remove file", full_path)
            assistant_objects_to_delete.append(assistant_object)
            num_to_delete += 1
            continue

        # check if remote file has been updated
        current_hash = assistant_object.get("hash")
        new_hash = files_by_key[full_path].hash
        if current_hash != new_hash:
            log.debug("%s hash changed, will update file", full_path)
            files_to_insert.append(files_by_key[full_path])
            assistant_objects_to_delete.append(assistant_object)
            num_to_update += 1
            continue

        # check if citation URL needs an update
        elif assistant_object.get("citation_url") != files_by_key[full_path].citation_url:
            log.debug("%s needs citation url update", full_path)
            metadata_update_args.append(
                dict(
                    metadata={"citation_url": files_by_key[full_path].citation_url},
                    filter={"full_path": full_path},
                )
            )

    # determine which new files to add
    files_to_add = _get_new_files_to_add(files_by_key, assistant_objects_by_path, resync)
    files_to_insert.extend(files_to_add)
    num_to_add += len(files_to_add)

    for obj in assistant_objects_to_delete:
        # remove active and pending_removal from the metadata so we don't use
        # these values as metadata filters
        if "active" in obj:
            del obj["active"]
        if "pending_removal" in obj:
            del obj["pending_removal"]

    return (
        assistant_objects_to_delete,
        files_to_insert,
        metadata_update_args,
        num_to_add,
        num_to_delete,
        num_to_update,
    )


def download_s3_files_and_embed(
    bucket, files: List[File], assistant_id: int
) -> tuple[List[File], int, int]:
    log.debug("%d s3 objects to download", len(files))

    completed_files = []

    download_errors = 0
    embed_errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for download_success in download_objs_concurrent(bucket, files, tmpdir):
            if not download_success:
                download_errors += 1

        for file in embed_files_concurrent(bucket, files, tmpdir, assistant_id):
            if file:
                completed_files.append(file)
            else:
                embed_errors += 1

    return completed_files, download_errors, embed_errors


def _purge_docs_with_agent_metadata():
    # remove any lingering documents that still use 'agent_id',
    # as we have now migrated to 'assistant_id'
    query = text("select distinct cmetadata->'agent_id' from langchain_pg_embedding")
    agent_ids = db.session.execute(query).all()
    for agent_id in agent_ids:
        agent_id = str(agent_id)
        if agent_id.isdigit():
            log.info("purging old documents with agent_id = %s", agent_id)
            vector_db.delete_document_chunks({"agent_id": agent_id})


def run(resync: bool = False) -> int:
    sync_config = get_sync_config()

    # remove any lingering inactive documents
    vector_db.delete_document_chunks({"active": False})

    if resync:
        _purge_docs_with_agent_metadata()

    download_errors_for_assistant = {}
    embed_errors_for_assistant = {}

    for assistant_config in sync_config.assistants:
        # check to see if assistant already exists... if so, update... if not, create
        assistant = Assistant.get_by_name(assistant_config.name)
        if assistant:
            if not assistant_config.system_prompt:
                log.debug("using default system prompt for assistant '%s'", assistant.name)
                assistant_config.system_prompt = cfg.DEFAULT_SYSTEM_PROMPT
            assistant.update(**dict(assistant_config))
        else:
            assistant = Assistant.create(**dict(assistant_config))

        # determine what changes need to be made
        (
            assistant_objects_to_delete,
            files_to_insert,
            metadata_update_args,
            num_adding,
            num_deleting,
            num_updating,
        ) = compare_files(assistant_config, assistant, sync_config.defaults, resync)

        log.info(
            "s3 sync: adding %d, deleting %d, updating %d, and %d metadata updates",
            num_adding,
            num_deleting,
            num_updating,
            len(metadata_update_args),
        )

        if assistant_objects_to_delete or files_to_insert or metadata_update_args:
            # set docs which will be removed to state pending_removal=True
            for metadata in assistant_objects_to_delete:
                vector_db.set_doc_states(active=True, pending_removal=True, search_filter=metadata)

            # download new docs for this assistant and embed in vector DB
            _, download_errors, embed_errors = download_s3_files_and_embed(
                assistant_config.bucket, files_to_insert, assistant.id
            )

            download_errors_for_assistant[assistant.id] = download_errors
            embed_errors_for_assistant[assistant.id] = embed_errors

            # set new doc chunks to active
            # all new docs will have state active=False, pending_removal=False
            metadata = {"assistant_id": assistant.id, "active": False, "pending_removal": False}
            vector_db.set_doc_states(active=True, pending_removal=False, search_filter=metadata)

            # set any old docs with state pending_removal=True to inactive
            metadata = {"assistant_id": assistant.id, "pending_removal": True}
            vector_db.set_doc_states(active=False, pending_removal=True, search_filter=metadata)

            # delete the now-inactive document chunks
            metadata = {"assistant_id": assistant.id, "active": False}
            vector_db.delete_document_chunks(metadata)

            for args in metadata_update_args:
                vector_db.update_cmetadata(**args, commit=False)
            vector_db.db.session.commit()

        # update list of filenames associated with the assistant
        assistant_objects = vector_db.get_distinct_cmetadata(
            search_filter={"assistant_id": assistant.id}
        )
        assistant_files = [File(**obj) for obj in assistant_objects]
        assistant.update(filenames=[file.display_name for file in assistant_files])

    exit_code = 0
    for assistant_id, error_count in download_errors_for_assistant.items():
        if error_count:
            log.error(
                "assistant %d hit %d download errors during sync, check logs",
                assistant_id,
                error_count,
            )
            exit_code = 1
    for assistant_id, error_count in embed_errors_for_assistant.items():
        if error_count:
            log.error(
                "assistant %d hit %d embedding errors during sync, check logs",
                assistant_id,
                error_count,
            )
            exit_code = 1

    return exit_code
