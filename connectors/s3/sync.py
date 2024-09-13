import logging
import tempfile
from concurrent import futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import List, Optional

import boto3
import boto3.session
import yaml
from pydantic import BaseModel

import connectors.config as cfg
from connectors.db.agent import Agent
from connectors.db.common import File, add_file

s3 = boto3.client("s3")


log = logging.getLogger(__name__)


DOWNLOAD_SUCCESS = "download success"
FILE_ADDED = "file added to agent"


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
    Agent.create(**dict(agent_config))


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

    return DOWNLOAD_SUCCESS


def download_objs(bucket, s3_objects, dest_dir):
    keys = [obj["Key"] for obj in s3_objects]
    with ProcessPoolExecutor() as executor:
        key_for_future = {executor.submit(download_obj, bucket, key, dest_dir): key for key in keys}

        for future in futures.as_completed(key_for_future):
            key = key_for_future[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception


def add_file_to_agent(bucket, s3_object, tmpdir, agent):
    """Adds an s3 object stored locally to agent"""
    file_path = s3_object["Key"]
    path_on_disk = Path(tmpdir) / Path(s3_object["Key"])

    with open(path_on_disk, "r") as fp:
        f = File(source=f"s3-{bucket}", full_path=file_path, content=fp.read())
    add_file(f, agent)

    return FILE_ADDED


def add_files_to_agent(bucket, s3_objects, tmpdir, agent):
    with ProcessPoolExecutor() as executor:
        key_for_future = {
            executor.submit(add_file_to_agent, bucket, s3_object, tmpdir, agent): s3_object["Key"]
            for s3_object in s3_objects
        }

        for future in futures.as_completed(key_for_future):
            key = key_for_future[future]
            exception = future.exception()

            if not exception:
                yield key, future.result()
            else:
                yield key, exception


def add_all_docs_to_agent(agent_config: AgentConfig, agent: Agent):
    bucket = agent_config.bucket
    prefix = agent_config.prefix
    s3_objects = get_all_s3_objects(bucket, prefix)
    log.debug("%d objects found in bucket %s at prefix %s", len(s3_objects), bucket, prefix)
    extensions = agent_config.file_types

    def _filter(file_data):
        return any([file_data["Key"].endswith(f".{extension}") for extension in extensions])

    s3_objects = filter(_filter, s3_objects)

    log.debug("%d objects to download after filtering for file type extensions %s", extensions)

    with tempfile.TemporaryDirectory() as tmpdir:
        for key, result in download_objs(bucket, s3_objects, tmpdir):
            log.debug("%s: %s", key, result)
        for key, result in add_files_to_agent(bucket, s3_objects, tmpdir, agent):
            log.debug("%s: %s"), key, result


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
            add_all_docs_to_agent(agent_config, agent)
