"""Rename agent to assistant

Revision ID: 29d2f3c7e7ba
Revises: f996e3ef2db7
Create Date: 2025-03-28 13:28:54.547622

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '29d2f3c7e7ba'
down_revision = 'f996e3ef2db7'
branch_labels = None
depends_on = None


def upgrade():
    op.rename_table('agent', 'assistant')


def downgrade():
    op.rename_table('assistant', 'agent')
