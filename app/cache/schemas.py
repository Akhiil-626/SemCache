from datetime import datetime
from uuid import UUID
from pydantic import BaseModel, Field


class CacheHitResult(BaseModel):
    is_hit: bool
    similarity_score: float | None = None
    cached_response: str | None = None
    original_model: str | None = None
    tier: str | None = None
    shadow_cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    entry_id: UUID | None = None
    guard_rejected: bool = False
    guard_reason: str | None = None


class CacheSaveRequest(BaseModel):
    prompt: str
    embedding: list[float]
    response: str
    tier: str = "small"
    model_name: str
    cost_usd: float = 0.0
    shadow_cost_usd: float = 0.0


class CacheEntryResponse(BaseModel):
    id: UUID
    prompt: str
    response: str
    tier: str
    model_name: str
    cost_usd: float
    shadow_cost_usd: float
    hit_count: int
    created_at: datetime
    last_hit_at: datetime
