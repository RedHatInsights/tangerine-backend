import logging
from typing import List, Optional, Self

from flask_sqlalchemy import SQLAlchemy

from connectors.config import DEFAULT_SYSTEM_PROMPT

db = SQLAlchemy()

log = logging.getLogger("tangerine.db.agent")


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
    def create(cls, name: str, description: str, system_prompt: str = None) -> Self:
        new_agent = cls(
            agent_name=name,
            description=description,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        db.session.add(new_agent)
        db.session.commit()
        db.session.refresh(new_agent)

        return new_agent

    @classmethod
    def list(cls) -> List[Self]:
        return cls.query.all()

    @classmethod
    def get(cls, id: int) -> Optional[Self]:
        agent_id = int(id)
        agent = cls.query.session.get(cls, agent_id)
        return agent or None

    def refresh(self) -> Self:
        return db.session.refresh(self)

    def update(self, **kwargs) -> Self:
        for key, val in kwargs.items():
            if key == "id":
                # do not allow updating of id
                continue
            setattr(self, key, val)
        db.session.commit()
        self.refresh()
        return self

    def add_files(self, file_display_names: List[str]) -> Self:
        new_names = self.filenames.copy()
        for name in file_display_names:
            new_names.append(name)
        return self.update(filenames=new_names)

    def remove_files(self, file_display_names: List[str]) -> Self:
        new_names = [name for name in self.filenames.copy() if name not in file_display_names]
        return self.update(filesnames=new_names)

    def delete(self) -> None:
        db.session.delete(self)
        db.session.commit()
