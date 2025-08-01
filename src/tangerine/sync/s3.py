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
from tangerine.models import Assistant, KnowledgeBase
from tangerine.utils import File, embed_files_for_knowledgebase
from tangerine.vector import vector_db

s3 = boto3.client("s3")

log = logging.getLogger("tangerine.s3sync")


class PathConfig(BaseModel):
    prefix: str
    citation_url_template: Optional[str] = None
    extensions: Optional[List[str]] = None


class KnowledgeBaseConfig(BaseModel):
    name: str
    description: str
    bucket: str
    paths: List[PathConfig]


class AssistantConfig(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    knowledgebases: List[str]  # List of knowledgebase names


class SyncConfigDefaults(BaseModel):
    extensions: List[str]
    citation_url_template: str


class SyncConfig(BaseModel):
    defaults: SyncConfigDefaults
    knowledgebases: List[KnowledgeBaseConfig]
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


def embed_file(app_context, file: File, tmpdir: str, knowledgebase_id: int) -> File:
    """Adds an s3 object stored locally to knowledgebase"""
    app_context.push()

    log.debug("embedding file %s", file.full_path)

    with db.session():
        knowledgebase = KnowledgeBase.get(knowledgebase_id)
        path_on_disk = Path(tmpdir) / Path(file.full_path)

        with open(path_on_disk, "r") as fp:
            # add new files as active=False until all embedding was successful
            file.content = fp.read()
            file.active = False
            file.pending_removal = False

        embed_files_for_knowledgebase([file], knowledgebase.id)
        return file


def embed_files_concurrent(
    bucket: str, files: List[File], tmpdir: str, knowledgebase_id: int
) -> Iterator[Optional[File]]:
    with ThreadPoolExecutor(max_workers=cfg.S3_SYNC_POOL_SIZE) as executor:
        key_for_future = {
            executor.submit(
                embed_file, current_app.app_context(), file, tmpdir, knowledgebase_id
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


def get_file_list(
    knowledgebase_config: KnowledgeBaseConfig, defaults: SyncConfigDefaults
) -> List[File]:
    files = []

    bucket = knowledgebase_config.bucket

    for path_config in knowledgebase_config.paths:
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
    knowledgebase_config: KnowledgeBaseConfig,
    knowledgebase: KnowledgeBase,
    defaults: SyncConfigDefaults,
    resync: bool,
) -> tuple[List[dict], List[File], set[dict], int, int, int]:
    files = get_file_list(knowledgebase_config, defaults)

    # collect all unique file objects currently stored for this knowledgebase in the DB
    knowledgebase_objects = vector_db.get_distinct_cmetadata(
        search_filter={"knowledgebase_id": str(knowledgebase.id)}
    )

    # group by keys for easier comparisons
    files_by_key = {file.full_path: file for file in files}
    knowledgebase_objects_by_path = {obj["full_path"]: obj for obj in knowledgebase_objects}

    knowledgebase_objects_to_delete = []
    files_to_insert = []
    metadata_update_args = []

    num_to_add = 0
    num_to_delete = 0
    num_to_update = 0

    for knowledgebase_object in knowledgebase_objects:
        full_path = knowledgebase_object["full_path"]

        # if 'resync' is true, we are deleting all files
        if resync:
            log.debug("%s removing for resync", full_path)
            knowledgebase_objects_to_delete.append(knowledgebase_object)
            num_to_delete += 1
            continue

        # check if the entire prefix is no longer defined in the knowledgebase config
        prefixes = [path_config.prefix for path_config in knowledgebase_config.paths]
        if not any([full_path.startswith(prefix) for prefix in prefixes]):
            log.debug(
                "%s uses prefix not found in knowledgebase config, will remove file", full_path
            )
            knowledgebase_objects_to_delete.append(knowledgebase_object)
            num_to_delete += 1
            continue

        # check if remote file has been removed
        if full_path not in files_by_key:
            # stored file is not present in s3, mark for deletion
            log.debug("%s no longer present in s3, will remove file", full_path)
            knowledgebase_objects_to_delete.append(knowledgebase_object)
            num_to_delete += 1
            continue

        # check if remote file has been updated
        current_hash = knowledgebase_object.get("hash")
        new_hash = files_by_key[full_path].hash
        if current_hash != new_hash:
            log.debug("%s hash changed, will update file", full_path)
            files_to_insert.append(files_by_key[full_path])
            knowledgebase_objects_to_delete.append(knowledgebase_object)
            num_to_update += 1
            continue

        # check if citation URL needs an update
        elif knowledgebase_object.get("citation_url") != files_by_key[full_path].citation_url:
            log.debug("%s needs citation url update", full_path)
            metadata_update_args.append(
                dict(
                    metadata={"citation_url": files_by_key[full_path].citation_url},
                    search_filter={
                        "full_path": full_path,
                        "knowledgebase_id": str(knowledgebase.id),
                    },
                )
            )

    # determine which new files to add
    files_to_add = _get_new_files_to_add(files_by_key, knowledgebase_objects_by_path, resync)
    files_to_insert.extend(files_to_add)
    num_to_add += len(files_to_add)

    for obj in knowledgebase_objects_to_delete:
        # remove active and pending_removal from the metadata so we don't use
        # these values as metadata filters
        if "active" in obj:
            del obj["active"]
        if "pending_removal" in obj:
            del obj["pending_removal"]

    return (
        knowledgebase_objects_to_delete,
        files_to_insert,
        metadata_update_args,
        num_to_add,
        num_to_delete,
        num_to_update,
    )


def download_s3_files_and_embed(
    bucket, files: List[File], knowledgebase_id: int
) -> tuple[List[File], int, int]:
    log.debug("%d s3 objects to download", len(files))

    completed_files = []

    download_errors = 0
    embed_errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for download_success in download_objs_concurrent(bucket, files, tmpdir):
            if not download_success:
                download_errors += 1

        for file in embed_files_concurrent(bucket, files, tmpdir, knowledgebase_id):
            if file:
                completed_files.append(file)
            else:
                embed_errors += 1

    return completed_files, download_errors, embed_errors


def _purge_docs_with_old_metadata():
    # remove any lingering documents that still use 'agent_id' or 'assistant_id',
    # as we have now migrated to 'knowledgebase_id'
    query = text(
        "SELECT DISTINCT cmetadata->'agent_id' AS id FROM langchain_pg_embedding WHERE cmetadata->'agent_id' IS NOT NULL"
    )
    results = db.session.execute(query).all()
    for row in results:
        agent_id = str(row.id)
        if agent_id.isdigit():
            log.info("purging old documents using obsolete field 'agent_id' = %s", agent_id)
            vector_db.delete_document_chunks({"agent_id": agent_id})

    query = text(
        "SELECT DISTINCT cmetadata->'assistant_id' AS id FROM langchain_pg_embedding WHERE cmetadata->'assistant_id' IS NOT NULL"
    )
    results = db.session.execute(query).all()
    for row in results:
        assistant_id = str(row.id)
        if assistant_id.isdigit():
            log.info("purging old documents using obsolete field 'assistant_id' = %s", assistant_id)
            vector_db.delete_document_chunks({"assistant_id": assistant_id})


def run(resync: bool = False) -> int:
    sync_config = get_sync_config()

    # remove any lingering inactive documents
    vector_db.delete_document_chunks({"active": False})

    if resync:
        _purge_docs_with_old_metadata()

    compare_errors_for_knowledgebase = {}
    download_errors_for_knowledgebase = {}
    embed_errors_for_knowledgebase = {}

    # First, process all knowledgebases
    for knowledgebase_config in sync_config.knowledgebases:
        # check to see if knowledgebase already exists... if so, update... if not, create
        knowledgebase = KnowledgeBase.get_by_name(knowledgebase_config.name)
        if knowledgebase:
            knowledgebase.update(**dict(knowledgebase_config))
        else:
            knowledgebase = KnowledgeBase.create(**dict(knowledgebase_config))

        # determine what changes need to be made
        try:
            (
                knowledgebase_objects_to_delete,
                files_to_insert,
                metadata_update_args,
                num_adding,
                num_deleting,
                num_updating,
            ) = compare_files(knowledgebase_config, knowledgebase, sync_config.defaults, resync)
        except Exception as err:
            log.exception(
                "s3 sync: unexpected error when comparing files for knowledgebase, moving on..."
            )
            compare_errors_for_knowledgebase[knowledgebase.id] = str(err)
            continue

        log.info(
            "s3 sync knowledgebase '%s': adding %d, deleting %d, updating %d, and %d metadata updates",
            knowledgebase.name,
            num_adding,
            num_deleting,
            num_updating,
            len(metadata_update_args),
        )

        if knowledgebase_objects_to_delete or files_to_insert or metadata_update_args:
            # set docs which will be removed to state pending_removal=True
            for metadata in knowledgebase_objects_to_delete:
                vector_db.set_doc_states(active=True, pending_removal=True, search_filter=metadata)

            # download new docs for this knowledgebase and embed in vector DB
            _, download_errors, embed_errors = download_s3_files_and_embed(
                knowledgebase_config.bucket, files_to_insert, knowledgebase.id
            )

            download_errors_for_knowledgebase[knowledgebase.id] = download_errors
            embed_errors_for_knowledgebase[knowledgebase.id] = embed_errors

            # set new doc chunks to active
            # all new docs will have state active=False, pending_removal=False
            metadata = {
                "knowledgebase_id": str(knowledgebase.id),
                "active": False,
                "pending_removal": False,
            }
            vector_db.set_doc_states(active=True, pending_removal=False, search_filter=metadata)

            # set any old docs with state pending_removal=True to inactive
            metadata = {"knowledgebase_id": str(knowledgebase.id), "pending_removal": True}
            vector_db.set_doc_states(active=False, pending_removal=True, search_filter=metadata)

            # delete the now-inactive document chunks
            metadata = {"knowledgebase_id": str(knowledgebase.id), "active": False}
            vector_db.delete_document_chunks(metadata)

            for args in metadata_update_args:
                vector_db.update_cmetadata(**args, commit=False)
            vector_db.db.session.commit()

        # update list of filenames associated with the knowledgebase
        knowledgebase_objects = vector_db.get_distinct_cmetadata(
            search_filter={"knowledgebase_id": str(knowledgebase.id)}
        )
        knowledgebase_files = [File(**obj) for obj in knowledgebase_objects]
        knowledgebase.update(filenames=[file.display_name for file in knowledgebase_files])

    # Then, process all assistants
    for assistant_config in sync_config.assistants:
        if not assistant_config.system_prompt:
            log.debug("using default system prompt for assistant '%s'", assistant_config.name)
            assistant_config.system_prompt = cfg.DEFAULT_SYSTEM_PROMPT

        # Remove knowledgebases field for now, we will look up the id later
        assistant_data = dict(assistant_config)
        knowledgebase_names = assistant_data.pop("knowledgebases", [])

        # check to see if assistant already exists... if not, create it, otherwise update it
        assistant = Assistant.get_by_name(assistant_config.name)
        if not assistant:
            assistant = Assistant.create(**assistant_data)
        else:
            assistant.update(**assistant_data)

        # Associate knowledgebases with the assistant
        kbs_not_found = set()
        for kb_name in knowledgebase_names:
            knowledgebase = KnowledgeBase.get_by_name(kb_name)
            if knowledgebase:
                assistant.associate_knowledgebase(knowledgebase.id)
                log.info(
                    "associated knowledgebase '%s' with assistant '%s'", kb_name, assistant.name
                )
            else:
                kbs_not_found.add(kb_name)
                log.error(
                    "knowledgebase '%s' not found, cannot associate to assistant '%s'",
                    kb_name,
                    assistant.name,
                )

    exit_code = 0
    for knowledgebase_id, error_count in download_errors_for_knowledgebase.items():
        if error_count:
            log.error(
                "knowledgebase %d hit %d download errors during sync, check logs",
                knowledgebase_id,
                error_count,
            )
            exit_code = 1
    for knowledgebase_id, error_count in embed_errors_for_knowledgebase.items():
        if error_count:
            log.error(
                "knowledgebase %d hit %d embedding errors during sync, check logs",
                knowledgebase_id,
                error_count,
            )
            exit_code = 1
    for knowledgebase_id, _ in compare_errors_for_knowledgebase.items():
        log.error(
            "knowledgebase %d hit errors during file comparison, check logs",
            knowledgebase_id,
        )
        exit_code = 1
    if kbs_not_found:
        kb_names_joined = ", ".join([name for name in kbs_not_found])
        log.error(
            f"coud not associate assistants with non-existant knowledgebases: {kb_names_joined}"
        )
        exit_code = 1

    return exit_code
