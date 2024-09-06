from typing import List, Optional

import boto3
import yaml
from pydantic import BaseModel

import connectors.config as cfg
import connectors.db.agent as db
from connectors.db.vector import vector_interface

s3 = boto3.client("s3")


class AgentConfig(BaseModel):
    name: str
    description: str
    system_prompt: Optional[str] = None
    bucket: str
    prefix: str
    file_types: List[str]


class SyncConfig(BaseModel):
    agents: List[AgentConfig]


def get_all_objects(bucket: str, prefix: str) -> List:
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


def create_agent(agent_config: AgentConfig) -> db.Agent:
    return db.create_agent(dict(agent_config))


def create_agents() -> None:
    sync_config = get_sync_config()
    for agent_config in sync_config.agents:
        create_agent(agent_config)
