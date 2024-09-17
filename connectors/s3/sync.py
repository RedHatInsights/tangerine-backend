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
from connectors.db.agent import Agent
from connectors.db.common import File, add_files_to_agent, embed_files

s3 = boto3.client("s3")

log = logging.getLogger(__name__)


class AgentConfig(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    bucket: str
    prefix: str
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


def create_agent(agent_config: AgentConfig) -> Agent:
    return Agent.create(**dict(agent_config))


def sync_agent(agent: Agent, agent_config: AgentConfig) -> Agent:
    agent.update(**dict(agent_config))


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

    agent = Agent.get(agent_id)

    file_path = s3_object["Key"]
    path_on_disk = Path(tmpdir) / Path(s3_object["Key"])

    with open(path_on_disk, "r") as fp:
        file = File(source=f"s3-{bucket}", full_path=file_path, content=fp.read())

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


def add_all_docs_to_agent(agent_config: AgentConfig, agent: Agent):
    bucket = agent_config.bucket
    prefix = agent_config.prefix
    s3_objects = get_all_s3_objects(bucket, prefix)
    log.debug("%d objects found in bucket %s at prefix %s", len(s3_objects), bucket, prefix)
    extensions = agent_config.file_types

    def _filter(file_data):
        return any([file_data["Key"].endswith(f".{extension}") for extension in extensions])

    s3_objects = list(filter(_filter, s3_objects))

    log.debug(
        "%d objects to download after filtering for file type extensions %s",
        len(s3_objects),
        extensions,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        for download_success in download_objs_concurrent(bucket, s3_objects, tmpdir):
            if not download_success:
                raise Exception("hit errors during download, aborting")

        files = []
        for file in embed_files_concurrent(bucket, s3_objects, tmpdir, agent.id):
            if not file:
                raise Exception("hit errors creating embeddings, aborting")
            files.append(file)

        add_files_to_agent(files, agent)


def run() -> None:
    sync_config = get_sync_config()
    for agent_config in sync_config.agents:
        if not agent_config.file_types:
            agent_config.file_types = sync_config.default_file_types

        agent = Agent.get_by_name(agent_config.name)
        if agent:
            sync_agent(agent, agent_config)
            # sync docs...
        else:
            agent = create_agent(agent_config)
        # todo: indent this
        add_all_docs_to_agent(agent_config, agent)
