"""Add model column to assistant table

Revision ID: b9791218a532
Revises: 29d2f3c7e7ba
Create Date: 2025-06-12 08:45:34.843646

"""

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "b9791218a532"
down_revision = "29d2f3c7e7ba"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("assistant", schema=None) as batch_op:
        batch_op.add_column(sa.Column("model", sa.String(length=50), nullable=True))


def downgrade():
    with op.batch_alter_table("assistant", schema=None) as batch_op:
        batch_op.drop_column("model")
