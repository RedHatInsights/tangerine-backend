"""Merge migration heads

Revision ID: merge_heads_2025
Revises: b9791218a532, c88ac0c951d2
Create Date: 2025-01-12 10:48:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'merge_heads_2025'
down_revision = ('b9791218a532', 'c88ac0c951d2')
branch_labels = None
depends_on = None


def upgrade():
    # This is a merge migration - no schema changes needed
    # All schema changes are handled by the parent migrations
    pass


def downgrade():
    # This is a merge migration - no schema changes needed
    # All schema changes are handled by the parent migrations
    pass 