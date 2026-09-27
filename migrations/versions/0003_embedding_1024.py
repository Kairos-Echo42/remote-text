"""Align embedding columns with DashScope text-embedding-v4.

Revision ID: 0003_embedding_1024
Revises: 0002_model_kind
Create Date: 2026-09-14
"""
from alembic import op

revision = "0003_embedding_1024"
down_revision = "0002_model_kind"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute("UPDATE memory_records SET embedding = NULL")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1024)")
    op.execute("ALTER TABLE memory_records ALTER COLUMN embedding TYPE vector(1024)")


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name != "postgresql":
        return
    op.execute("UPDATE chunks SET embedding = NULL")
    op.execute("UPDATE memory_records SET embedding = NULL")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(1536)")
    op.execute("ALTER TABLE memory_records ALTER COLUMN embedding TYPE vector(1536)")