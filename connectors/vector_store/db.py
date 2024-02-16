from sqlalchemy.orm import mapped_column
from flask_sqlalchemy import SQLAlchemy
from pgvector.sqlalchemy import Vector


db = SQLAlchemy()


class Agents(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_name = db.Column(db.String(50), nullable=False)
    description = db.Column(db.Text, nullable=False)
    system_prompt = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f'<Agents {self.id}>'


class VectorCollections(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    agent_id = db.Column(db.Integer, db.ForeignKey('agents.id'), nullable=False)
    embedding = mapped_column(Vector(4096), nullable=False)

    def __repr__(self):
        return f'<VectorCollections {self.id}>'
