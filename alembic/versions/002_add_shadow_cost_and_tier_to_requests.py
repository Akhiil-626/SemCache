"""Add shadow_cost_usd and tier columns to requests and semantic_cache

Revision ID: 002
Revises: 001
Create Date: 2026-10-11 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add columns to requests table
    op.add_column(
        "requests",
        sa.Column("tier", sa.String(length=50), nullable=False, server_default="small"),
    )
    op.add_column(
        "requests",
        sa.Column("shadow_cost_usd", sa.Float(), nullable=False, server_default="0.0"),
    )

    # Add columns to semantic_cache table
    op.add_column(
        "semantic_cache",
        sa.Column("tier", sa.String(length=50), nullable=False, server_default="small"),
    )
    op.add_column(
        "semantic_cache",
        sa.Column("shadow_cost_usd", sa.Float(), nullable=False, server_default="0.0"),
    )


def downgrade() -> None:
    op.drop_column("semantic_cache", "shadow_cost_usd")
    op.drop_column("semantic_cache", "tier")
    op.drop_column("requests", "shadow_cost_usd")
    op.drop_column("requests", "tier")
