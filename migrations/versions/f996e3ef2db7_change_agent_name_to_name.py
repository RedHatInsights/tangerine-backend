"""Change agent_name to name

Revision ID: f996e3ef2db7
Revises: d0fa16dacff5a
Create Date: 2025-03-28 12:52:46.163318

"""

from alembic import op


# revision identifiers, used by Alembic.
revision = "f996e3ef2db7"
down_revision = "d0fa16dacff5a"
branch_labels = None
depends_on = None


def upgrade():
    with op.batch_alter_table("agent", schema=None) as batch_op:
        # Rename the column 'agent_name' to 'name'
        batch_op.alter_column("agent_name", new_column_name="name")


def downgrade():
    with op.batch_alter_table("agent", schema=None) as batch_op:
        # Rename the column 'name' to 'agent_name'
        batch_op.alter_column("name", new_column_name="agent_name")
