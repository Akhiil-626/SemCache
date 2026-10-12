"""Rework semantic cache with 384d all-minilm vector, HNSW index, metadata filters, and request similarity

Revision ID: 005
Revises: 004
Create Date: 2026-10-12 11:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import pgvector.sqlalchemy
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "005"
down_revision: Union[str, None] = "004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Truncate existing semantic cache entries since old 1536d vectors are invalid
    op.execute("TRUNCATE TABLE semantic_cache")

    # 2. Recreate embedding column with 384 dimensions for all-minilm
    op.drop_column("semantic_cache", "embedding")
    op.add_column(
        "semantic_cache",
        sa.Column("embedding", pgvector.sqlalchemy.Vector(384), nullable=False),
    )

    # 3. Add filtering, metadata, and token tracking columns to semantic_cache
    op.add_column(
        "semantic_cache",
        sa.Column("tenant_id", sa.String(length=100), nullable=False, server_default="default"),
    )
    op.add_column(
        "semantic_cache",
        sa.Column("cache_version", sa.String(length=20), nullable=False, server_default="v1"),
    )
    op.add_column(
        "semantic_cache",
        sa.Column(
            "expires_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now() + interval '1 hour'"),
        ),
    )
    op.add_column(
        "semantic_cache",
        sa.Column("prompt_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "semantic_cache",
        sa.Column("completion_tokens", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column(
        "semantic_cache",
        sa.Column("latency_ms", sa.Float(precision=53), nullable=False, server_default="0"),
    )

    # 4. Create HNSW index using vector_cosine_ops
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_semantic_cache_embedding_hnsw "
        "ON semantic_cache USING hnsw (embedding vector_cosine_ops)"
    )
    op.create_index(
        "ix_semantic_cache_tenant_tier_version_expires",
        "semantic_cache",
        ["tenant_id", "tier", "cache_version", "expires_at"],
    )

    # 5. Add similarity column to requests table
    op.add_column(
        "requests",
        sa.Column("similarity", sa.Float(precision=53), nullable=True),
    )


def downgrade() -> None:
    # 1. Remove similarity from requests
    op.drop_column("requests", "similarity")

    # 2. Drop indexes on semantic_cache
    op.drop_index("ix_semantic_cache_tenant_tier_version_expires", table_name="semantic_cache")
    op.execute("DROP INDEX IF EXISTS ix_semantic_cache_embedding_hnsw")

    # 3. Drop added columns from semantic_cache
    op.drop_column("semantic_cache", "latency_ms")
    op.drop_column("semantic_cache", "completion_tokens")
    op.drop_column("semantic_cache", "prompt_tokens")
    op.drop_column("semantic_cache", "expires_at")
    op.drop_column("semantic_cache", "cache_version")
    op.drop_column("semantic_cache", "tenant_id")

    # 4. Revert embedding column to 1536 dimensions
    op.drop_column("semantic_cache", "embedding")
    op.add_column(
        "semantic_cache",
        sa.Column("embedding", pgvector.sqlalchemy.Vector(1536), nullable=False),
    )
