"""Merge filename removal migrations

Revision ID: 64251c3f53da
Revises: 896de87742d0, a1b2c3d4e5f6
Create Date: 2025-08-05 15:14:36.142645

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '64251c3f53da'
down_revision = ('896de87742d0', 'a1b2c3d4e5f6')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
