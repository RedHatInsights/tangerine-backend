import logging
from typing import List, Optional, Self

from tangerine.db import db
from tangerine.utils import get_files_for_knowledgebase

log = logging.getLogger("tangerine.models.knowledgebase")


# Association table for many-to-many relationship between Assistant and KnowledgeBase
assistant_knowledgebase = db.Table(
    "assistant_knowledgebase",
    db.Column("assistant_id", db.Integer, db.ForeignKey("assistant.id"), primary_key=True),
    db.Column("knowledgebase_id", db.Integer, db.ForeignKey("knowledgebase.id"), primary_key=True),
)


class KnowledgeBase(db.Model):
    __tablename__ = "knowledgebase"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(50), nullable=False, unique=True)
    description = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=db.func.now(), nullable=False)
    updated_at = db.Column(
        db.DateTime, default=db.func.now(), onupdate=db.func.now(), nullable=False
    )

    # Many-to-many relationship with Assistant
    assistants = db.relationship(
        "Assistant",
        secondary=assistant_knowledgebase,
        back_populates="knowledgebases",
        lazy="dynamic",
    )

    def to_dict(self, get_filenames=False):
        result = {}
        for c in self.__table__.columns:
            value = getattr(self, c.name)
            # Convert datetime objects to ISO format strings for JSON serialization
            if hasattr(value, "isoformat"):
                value = value.isoformat()
            result[c.name] = value

        if get_filenames:
            # Add filenames from vector database
            result["filenames"] = get_files_for_knowledgebase(self.id)

        return result

    def __repr__(self):
        return f"<KnowledgeBase {self.id}: {self.name}>"

    @classmethod
    def create(cls, name: str, description: str, **kwargs) -> Self:
        new_kb = cls(
            name=name,
            description=description,
        )
        db.session.add(new_kb)
        db.session.commit()
        db.session.refresh(new_kb)

        log.debug("knowledgebase %d created", new_kb.id)
        return new_kb

    @classmethod
    def list(cls) -> List[Self]:
        return db.session.scalars(db.select(cls)).all()

    @classmethod
    def get(cls, id: int) -> Optional[Self]:
        kb_id = int(id)
        kb = db.session.get(cls, kb_id)
        return kb

    @classmethod
    def get_by_name(cls, name: str) -> Optional[Self]:
        kb = db.session.scalar(db.select(cls).filter_by(name=name))
        log.debug("get knowledgebase by name '%s' result: %s", name, kb)
        return kb

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
        log.debug("updated attributes %s of knowledgebase %d", updated_keys, self.id)
        return self

    def is_associated_with_assistants(self) -> bool:
        """Check if this knowledgebase is associated with any assistants."""
        return self.assistants.count() > 0

    def get_associated_assistants(self) -> List:
        """Get list of assistants associated with this knowledgebase."""
        return self.assistants.all()

    def delete(self) -> None:
        """Delete this knowledgebase. Raises ValueError if still associated with assistants."""
        if self.is_associated_with_assistants():
            associated = [a.name for a in self.get_associated_assistants()]
            raise ValueError(
                f"Cannot delete knowledgebase '{self.name}' - still associated with assistants: {associated}"
            )

        db.session.delete(self)
        db.session.commit()
        log.debug("knowledgebase with id %d deleted", self.id)
