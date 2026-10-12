"""Add tenant_id, tokens, cache_status, status, and error_message to requests

Revision ID: 003
Revises: 002
Create Date: 2026-10-11 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "requests",
        sa.Column("tenant_id", sa.String(length=100), nullable=False, server_default="default"),
    )
    op.add_column(
        "requests",
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "requests",
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "requests",
        sa.Column("cache_status", sa.String(length=20), nullable=False, server_default="none"),
    )
    op.add_column(
        "requests",
        sa.Column("status", sa.String(length=20), nullable=False, server_default="ok"),
    )
    op.add_column(
        "requests",
        sa.Column("error_message", sa.Text(), nullable=True),
    )
    op.create_index(
        "ix_requests_tenant_id_created_at",
        "requests",
        ["tenant_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_requests_tenant_id_created_at", table_name="requests")
    op.drop_column("requests", "error_message")
    op.drop_column("requests", "status")
    op.drop_column("requests", "cache_status")
    op.drop_column("requests", "completion_tokens")
    op.drop_column("requests", "prompt_tokens")
    op.drop_column("requests", "tenant_id")
