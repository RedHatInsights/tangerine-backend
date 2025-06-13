import logging
from typing import Iterable, List, Optional, Self

import tangerine.config as cfg
from tangerine.db import db

log = logging.getLogger("tangerine.models.assistant")


class Assistant(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)
    filenames = db.Column(db.ARRAY(db.String), default=[], nullable=True)
    model = db.Column(db.String(50), default=None, nullable=True)

    def to_dict(self):
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}

    def __repr__(self):
        return f"<Assistant {self.id}>"

    @classmethod
    def create(cls, name: str, description: str, system_prompt: str = None, **kwargs) -> Self:
        new_assistant = cls(
            name=name,
            description=description,
            system_prompt=system_prompt or cfg.DEFAULT_SYSTEM_PROMPT,
        )
        db.session.add(new_assistant)
        db.session.commit()
        db.session.refresh(new_assistant)

        log.debug("assistant %d created", new_assistant.id)

        return new_assistant

    @classmethod
    def list(cls) -> List[Self]:
        return db.session.scalars(db.select(cls)).all()

    @classmethod
    def get(cls, id: int) -> Optional[Self]:
        assistant_id = int(id)
        assistant = db.session.get(cls, assistant_id)
        return assistant

    @classmethod
    def get_by_name(cls, name: str) -> Optional[Self]:
        assistant = db.session.scalar(db.select(cls).filter_by(name=name))
        log.debug("get assistant by name '%s' result: %s", name, assistant)
        return assistant

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
        log.debug("updated attributes %s of assistant %d", updated_keys, self.id)
        return self

    def add_files(self, file_display_names: Iterable[str]) -> Self:
        filenames = self.filenames.copy()
        file_display_names = set(file_display_names)
        for name in file_display_names:
            if name not in filenames:
                filenames.append(name)
        log.debug(
            "adding %d files to assistant %d, total files now %d",
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
            "removing %d files from assistant %d, old count %d, new count %d",
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
        log.debug("assistant with id %d deleted", self.id)
