"""Add tokens_saved, saved_latency_ms, and saved_shadow_cost_usd to requests

Revision ID: 004
Revises: 003
Create Date: 2026-10-12 10:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "004"
down_revision: Union[str, None] = "003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "requests",
        sa.Column("tokens_saved", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "requests",
        sa.Column(
            "saved_latency_ms",
            sa.Float(precision=53),
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column(
        "requests",
        sa.Column(
            "saved_shadow_cost_usd",
            sa.Float(precision=53),
            nullable=False,
            server_default="0",
        ),
    )


def downgrade() -> None:
    op.drop_column("requests", "saved_shadow_cost_usd")
    op.drop_column("requests", "saved_latency_ms")
    op.drop_column("requests", "tokens_saved")
