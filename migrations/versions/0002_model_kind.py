"""Add model profile role.

Revision ID: 0002_model_kind
Revises: 0001_initial
Create Date: 2026-09-14
"""
from alembic import op
import sqlalchemy as sa

revision = "0002_model_kind"
down_revision = "0001_initial"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "model_profiles",
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="chat"),
    )
    op.create_index("ix_model_profiles_kind", "model_profiles", ["kind"], unique=False)


def downgrade() -> None:
    op.drop_index("ix_model_profiles_kind", table_name="model_profiles")
    op.drop_column("model_profiles", "kind")