"""Add cascade and validation fields to requests table

Revision ID: 007
Revises: 006
Create Date: 2026-10-13 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "007"
down_revision: Union[str, None] = "006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "requests",
        sa.Column("escalation_count", sa.Integer(), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "requests",
        sa.Column("attempts", sa.JSON(), nullable=True),
    )
    op.add_column(
        "requests",
        sa.Column("validation_failed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
    )
    op.add_column(
        "requests",
        sa.Column("wasted_latency_ms", sa.Float(precision=53), nullable=False, server_default=sa.text("0")),
    )
    op.add_column(
        "requests",
        sa.Column("wasted_shadow_cost_usd", sa.Float(precision=53), nullable=False, server_default=sa.text("0")),
    )


def downgrade() -> None:
    op.drop_column("requests", "wasted_shadow_cost_usd")
    op.drop_column("requests", "wasted_latency_ms")
    op.drop_column("requests", "validation_failed")
    op.drop_column("requests", "attempts")
    op.drop_column("requests", "escalation_count")
