import logging
from typing import List

from connectors.config import DEFAULT_SYSTEM_PROMPT

from .common import db

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


def create_agent(name: str, description: str, system_prompt: str = None) -> Agent:
    new_agent = Agent(
        agent_name=name,
        description=description,
        system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
    )
    db.session.add(new_agent)
    db.session.commit()
    db.session.refresh(new_agent)

    return new_agent


def get_all_agents() -> List[Agent]:
    return Agent.query.all()
