"""Initial AgentForge schema.

Revision ID: 0001_initial
Revises:
Create Date: 2026-09-13
"""
from alembic import op

from agentforge.db import Base
from agentforge import models  # noqa: F401

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    Base.metadata.create_all(bind=bind)


def downgrade() -> None:
    Base.metadata.drop_all(bind=op.get_bind())