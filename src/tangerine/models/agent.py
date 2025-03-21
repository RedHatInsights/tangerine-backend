import logging
from typing import Iterable, List, Optional, Self

import tangerine.config as cfg
from tangerine.db import db

log = logging.getLogger("tangerine.models.agent")


class Agent(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    agent_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    filenames = db.Column(db.ARRAY(db.String), default=[], nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self):
        return f"<Agent {self.id}>"

    @classmethod
    def create(cls, name: str, description: str, system_prompt: str = None, **kwargs) -> Self:
        new_agent = cls(
            agent_name=name,
            description=description,
            system_prompt=system_prompt or cfg.DEFAULT_SYSTEM_PROMPT,
        )
        db.session.add(new_agent)
        db.session.commit()
        db.session.refresh(new_agent)

        log.debug("agent %d created", new_agent.id)

        return new_agent

    @classmethod
    def list(cls) -> List[Self]:
        return db.session.scalars(db.select(cls)).all()

    @classmethod
    def get(cls, id: int) -> Optional[Self]:
        agent_id = int(id)
        agent = db.session.get(cls, agent_id)
        return agent

    @classmethod
    def get_by_name(cls, name: str) -> Optional[Self]:
        agent = db.session.scalar(db.select(cls).filter_by(agent_name=name))
        log.debug("get agent by name '%s' result: %s", name, agent)
        return agent

    def update(self, **kwargs) -> Self:
        updated_keys = []
        for key, val in kwargs.items():
            if key == "id":
                # do not allow updating of id
                continue
            setattr(self, key, val)
            updated_keys.append(key)
        db.session.add(self)
        db.session.commit()
        db.session.refresh(self)
        log.debug("updated attributes %s of agent %d", updated_keys, self.id)
        return self

    def add_files(self, file_display_names: Iterable[str]) -> Self:
        filenames = self.filenames.copy()
        file_display_names = set(file_display_names)
        for name in file_display_names:
            if name not in filenames:
                filenames.append(name)
        log.debug(
            "adding %d files to agent %d, total files now %d",
            len(file_display_names),
            self.id,
            len(filenames),
        )
        return self.update(filenames=filenames)

    def remove_files(self, file_display_names: Iterable[str]) -> Self:
        new_names = [name for name in self.filenames.copy() if name not in file_display_names]
        old_count = len(self.filenames)
        new_count = len(new_names)
        diff = old_count - new_count
        log.debug(
            "removing %d files from agent %d, old count %d, new count %d",
            diff,
            self.id,
            old_count,
            new_count,
        )
        if diff > 0:
            return self.update(filesnames=new_names)
        return self

    def delete(self) -> None:
        db.session.delete(self)
        db.session.commit()
        log.debug("agent with id %d deleted", self.id)
