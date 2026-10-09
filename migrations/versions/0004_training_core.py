"""Add ML/DL training experiment subsystem.

Revision ID: 0004_training_core
Revises: 0003_embedding_1024
Create Date: 2026-09-27
"""

from alembic import op

from agentforge import models  # noqa: F401
from agentforge.db import Base

revision = "0004_training_core"
down_revision = "0003_embedding_1024"
branch_labels = None
depends_on = None

NEW_TABLES = [
    "datasets",
    "dataset_versions",
    "experiments",
    "training_jobs",
    "training_attempts",
    "training_events",
    "metric_points",
    "checkpoints",
    "model_versions",
    "lineage_nodes",
    "lineage_edges",
    "decision_records",
    "budget_reservations",
]


def upgrade() -> None:
    Base.metadata.create_all(bind=op.get_bind(), tables=[Base.metadata.tables[name] for name in NEW_TABLES])


def downgrade() -> None:
    for name in reversed(NEW_TABLES):
        Base.metadata.tables[name].drop(bind=op.get_bind(), checkfirst=True)
