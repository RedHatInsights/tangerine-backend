"""Add ts_vector column to langchain_pg_embeddings for full text search

Revision ID: d0fa16dacff5a
Revises: c87bdbcc45f5
Create Date: 2025-03-24 16:03:31.063433

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "d0fa16dacff5a"
down_revision = "c87bdbcc45f5"
branch_labels = None
depends_on = None


def upgrade():
    op.execute("""
        ALTER TABLE langchain_pg_embedding
        ADD COLUMN fts_vector tsvector
        GENERATED ALWAYS AS (to_tsvector('english', document)) STORED
    """)
    op.execute("""
        CREATE INDEX idx_langchain_pg_embedding_fts
        ON langchain_pg_embedding USING GIN (fts_vector)
    """)


def downgrade():
    op.execute("DROP INDEX IF EXISTS idx_langchain_pg_embedding_fts")
    op.execute("ALTER TABLE langchain_pg_embedding DROP COLUMN IF EXISTS fts_vector")
