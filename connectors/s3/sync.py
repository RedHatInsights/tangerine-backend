import logging
import tempfile
from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

import boto3
import boto3.session
import yaml
from flask import current_app
from pydantic import BaseModel

import connectors.config as cfg
from connectors.db.agent import Agent, db
from connectors.db.common import File, embed_files
from connectors.db.vector import vector_db

s3 = boto3.client("s3")

log = logging.getLogger("tangerine.s3sync")


class AgentConfig(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    bucket: str
    prefixes: List[str]
    file_types: Optional[List[str]] = None


class SyncConfig(BaseModel):
    default_file_types: List[str]
    agents: List[AgentConfig]


def get_all_s3_objects(bucket: str, prefix: str) -> List:
    objects = []
    paginator = s3.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
    for page in pages:
        objects.extend(page["Contents"])

    return objects


def get_sync_config() -> SyncConfig:
    with open(cfg.S3_SYNC_CONFIG_FILE) as fp:
        data = yaml.safe_load(fp)
        sync_config = SyncConfig(**data)

    return sync_config


def download_obj(bucket, obj_key, dest_dir):
    """Downloads an object from S3 to dest dir."""
    dest_path = Path(dest_dir)
    download_path = dest_path / obj_key

    # create directory tree for this file
    download_path.parents[0].mkdir(parents=True, exist_ok=True)

    log.debug("downloading %s to %s", obj_key, download_path)
    s3.download_file(bucket, obj_key, str(download_path))


def download_objs_concurrent(bucket, s3_objects, dest_dir):
    keys = [obj["Key"] for obj in s3_objects]
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


def embed_file(app_context, bucket, s3_object, tmpdir, agent_id):
    """Adds an s3 object stored locally to agent"""
    app_context.push()

    file_path = s3_object["Key"]

    log.debug("embedding file %s", file_path)

    with db.session():
        agent = Agent.get(agent_id)
        path_on_disk = Path(tmpdir) / Path(s3_object["Key"])

        with open(path_on_disk, "r") as fp:
            # add new files as active=False until all embedding was successful
            file = File(
                source=f"s3-{bucket}",
                full_path=file_path,
                active=False,
                pending_removal=False,
                content=fp.read(),
                hash=s3_object["ETag"],
            )

        embed_files([file], agent)
        return file


def embed_files_concurrent(bucket, s3_objects, tmpdir, agent_id):
    max_workers = int(cfg.SQLALCHEMY_POOL_SIZE / 2)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        key_for_future = {
            executor.submit(
                embed_file, current_app.app_context(), bucket, s3_object, tmpdir, agent_id
            ): s3_object["Key"]
            for s3_object in s3_objects
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


def compare_files(agent_config: AgentConfig, agent: Agent):
    # fetch all objects from the s3 bucket at 'prefix'
    bucket = agent_config.bucket
    prefixes = agent_config.prefixes
    s3_objects = []
    for prefix in prefixes:
        new_objects = get_all_s3_objects(bucket, prefix)
        s3_objects.extend(new_objects)
        log.debug("%d objects found in bucket %s at prefix %s", len(new_objects), bucket, prefix)

    # filter by desired extensions
    extensions = agent_config.file_types

    def _filter(file_data):
        return any([file_data["Key"].endswith(f".{extension}") for extension in extensions])

    s3_objects = list(filter(_filter, s3_objects))
    log.debug("%d total s3 objects after filtering for extensions %s", len(s3_objects), extensions)

    # collect all unique file objects stored on the agent
    agent_objects = vector_db.get_distinct_cmetadata(filter={"agent_id": agent.id})

    # group by keys for easier comparisons
    s3_objects_by_key = {obj["Key"]: obj for obj in s3_objects}
    agent_objects_by_path = {obj["full_path"]: obj for obj in agent_objects}

    agent_objects_to_delete = []
    s3_objects_to_insert = []

    num_to_add = 0
    num_to_delete = 0
    num_to_update = 0

    for agent_object in agent_objects:
        full_path = agent_object["full_path"]

        # check if the entire prefix is no longer defined in the agent config
        if not any([full_path.startswith(prefix) for prefix in prefixes]):
            log.debug("%s uses prefix not found in agent config, will remove file", full_path)
            agent_objects_to_delete.append(agent_object)
            num_to_delete += 1

        # check if remote file has been removed
        elif full_path not in s3_objects_by_key:
            # stored file is not present in s3, mark for deletion
            log.debug("%s no longer present in s3, will remove file", full_path)
            agent_objects_to_delete.append(agent_object)
            num_to_delete += 1

        # check if remote file has been updated
        elif full_path in s3_objects_by_key:
            hash = agent_object.get("hash")
            etag = s3_objects_by_key[full_path]["ETag"]
            if hash != etag:
                log.debug("%s hash changed, will update file", full_path)
                s3_objects_to_insert.append(s3_objects_by_key[full_path])
                agent_objects_to_delete.append(agent_object)
                num_to_update += 1

    # check if there's a new remote file to add
    for key, s3_object in s3_objects_by_key.items():
        if key not in agent_objects_by_path:
            log.debug("%s is new in s3, will add file", key)
            s3_objects_to_insert.append(s3_object)
            num_to_add += 1

    for obj in agent_objects_to_delete:
        # remove active and pending_removal from the metadata so we don't use
        # these values as metadata filters
        if "active" in obj:
            del obj["active"]
        if "pending_removal" in obj:
            del obj["pending_removal"]

    return agent_objects_to_delete, s3_objects_to_insert, num_to_add, num_to_delete, num_to_update


def download_s3_files_and_embed(bucket, s3_objects, agent_id):
    log.debug("%d objects to download", len(s3_objects))

    completed_files = []

    download_errors = 0
    embed_errors = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        for download_success in download_objs_concurrent(bucket, s3_objects, tmpdir):
            if not download_success:
                download_errors += 1

        for file in embed_files_concurrent(bucket, s3_objects, tmpdir, agent_id):
            if file:
                completed_files.append(file)
            else:
                embed_errors += 1

    return completed_files, download_errors, embed_errors


def run() -> int:
    sync_config = get_sync_config()

    # remove any lingering inactive documents
    vector_db.delete_document_chunks({"active": False})

    download_errors_for_agent = {}
    embed_errors_for_agent = {}

    for agent_config in sync_config.agents:
        if not agent_config.file_types:
            agent_config.file_types = sync_config.default_file_types

        # check to see if agent already exists... if so, update... if not, create
        agent = Agent.get_by_name(agent_config.name)
        if agent:
            agent.update(**dict(agent_config))
        else:
            agent = Agent.create(**dict(agent_config))

        # determine what changes need to be made
        agent_objects_to_delete, s3_objects_to_insert, num_adding, num_deleting, num_updating = (
            compare_files(agent_config, agent)
        )

        log.info(
            "s3 sync: adding %d, deleting %d, updating %d", num_adding, num_deleting, num_updating
        )

        if agent_objects_to_delete or s3_objects_to_insert:
            # set docs which will be removed to state pending_removal=True
            for metadata in agent_objects_to_delete:
                vector_db.set_doc_states(active=True, pending_removal=True, filter=metadata)

            # download new docs for this agent and embed in vector DB
            _, download_errors, embed_errors = download_s3_files_and_embed(
                agent_config.bucket, s3_objects_to_insert, agent.id
            )

            download_errors_for_agent[agent.id] = download_errors
            embed_errors_for_agent[agent.id] = embed_errors

            # set new doc chunks to active
            # all new docs will have state active=False, pending_removal=False
            metadata = {"agent_id": agent.id, "active": False, "pending_removal": False}
            vector_db.set_doc_states(active=True, pending_removal=False, filter=metadata)

            # set any old docs with state pending_removal=True to inactive
            metadata = {"agent_id": agent.id, "pending_removal": True}
            vector_db.set_doc_states(active=False, pending_removal=True, filter=metadata)

            # delete the now-inactive document chunks
            metadata = {"agent_id": agent.id, "active": False}
            vector_db.delete_document_chunks(metadata)

        # update list of filenames associated with the agent
        agent_objects = vector_db.get_distinct_cmetadata(filter={"agent_id": agent.id})
        agent_files = [File(**obj) for obj in agent_objects]
        agent.update(filenames=[file.display_name for file in agent_files])

    exit_code = 0
    for agent_id, error_count in download_errors_for_agent.items():
        if error_count:
            log.error(
                "agent %d hit %d download errors during sync, check logs", agent_id, error_count
            )
            exit_code = 1
    for agent_id, error_count in embed_errors_for_agent.items():
        if error_count:
            log.error(
                "agent %d hit %d embedding errors during sync, check logs", agent_id, error_count
            )
            exit_code = 1

    return exit_code
