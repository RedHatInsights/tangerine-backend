"""Add conversation history functionality

Revision ID: e8f9c2d4a5b7
Revises: b9791218a532
Create Date: 2025-01-16 10:00:00.000000

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "e8f9c2d4a5b7"
down_revision = "b9791218a532"
branch_labels = None
depends_on = None


def upgrade():
    # Add user_id column to interactions table
    with op.batch_alter_table("interactions", schema=None) as batch_op:
        batch_op.add_column(sa.Column("user_id", sa.String(length=256), nullable=True))

    # Create conversations table
    op.create_table(
        "conversations",
        sa.Column("id", sa.UUID(), nullable=False),
        sa.Column("user_id", sa.String(length=256), nullable=True),
        sa.Column("session_id", sa.UUID(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("title", sa.String(length=256), nullable=True),
        sa.Column("assistant_name", sa.String(length=256), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade():
    # Drop conversations table
    op.drop_table("conversations")

    # Remove user_id column from interactions table
    with op.batch_alter_table("interactions", schema=None) as batch_op:
        batch_op.drop_column("user_id")
