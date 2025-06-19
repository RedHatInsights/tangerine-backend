"""Merge feature branch and main branch migrations

Revision ID: 371a6bebcfb1
Revises: b9791218a532, c88ac0c951d2
Create Date: 2025-06-19 10:52:30.333502

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '371a6bebcfb1'
down_revision = ('b9791218a532', 'c88ac0c951d2')
branch_labels = None
depends_on = None


def upgrade():
    pass


def downgrade():
    pass
