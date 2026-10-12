import uuid
from datetime import datetime, timezone
from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, DateTime, Float, Index, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.core.config import settings
from app.db.base import Base


class SemanticCacheEntry(Base):
    """Stores query prompts, their vector embeddings, and LLM generated responses."""

    __tablename__ = "semantic_cache"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    embedding: Mapped[list[float]] = mapped_column(
        Vector(settings.EMBEDDING_DIM), nullable=False
    )
    response: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(String(50), default="small", nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    tenant_id: Mapped[str] = mapped_column(
        String(100), default="default", server_default="default", nullable=False
    )
    cache_version: Mapped[str] = mapped_column(
        String(20), default="v1", server_default="v1", nullable=False
    )
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, server_default="0", nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    shadow_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    hit_count: Mapped[int] = mapped_column(Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    last_hit_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )


class RequestRecord(Base):
    """Stores all incoming query requests and their routing/caching execution metadata."""

    __tablename__ = "requests"
    __table_args__ = (
        Index("ix_requests_tenant_id_created_at", "tenant_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    response: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(String(50), default="small", nullable=False)
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    shadow_cost_usd: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tenant_id: Mapped[str] = mapped_column(
        String(100), default="default", server_default="default", nullable=False
    )
    prompt_tokens: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    completion_tokens: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    cache_status: Mapped[str] = mapped_column(
        String(20), default="none", server_default="none", nullable=False
    )
    tokens_saved: Mapped[int] = mapped_column(
        Integer, default=0, server_default="0", nullable=False
    )
    saved_latency_ms: Mapped[float] = mapped_column(
        Float, default=0.0, server_default="0", nullable=False
    )
    saved_shadow_cost_usd: Mapped[float] = mapped_column(
        Float, default=0.0, server_default="0", nullable=False
    )
    similarity: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), default="ok", server_default="ok", nullable=False
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
