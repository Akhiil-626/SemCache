import logging
import math
from typing import Any
from fastapi import APIRouter, Depends, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.db.models import RequestRecord
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Stats"])


class TierStats(BaseModel):
    request_count: int = 0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_shadow_cost_usd: float = 0.0
    cache_status_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "exact_hit": 0,
            "semantic_hit": 0,
            "miss": 0,
            "bypass": 0,
            "skipped": 0,
        }
    )
    hit_rate: float = 0.0
    total_tokens_saved: int = 0
    total_saved_latency_ms: float = 0.0
    total_saved_shadow_cost_usd: float = 0.0
    semantic_hit_count: int = 0
    avg_similarity_hits: float = 0.0
    avg_similarity_misses: float = 0.0


class StatsResponse(BaseModel):
    total_requests: int
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens_saved: int = 0
    total_saved_latency_ms: float = 0.0
    total_saved_shadow_cost_usd: float = 0.0
    cache_status_counts: dict[str, int] = Field(
        default_factory=lambda: {
            "exact_hit": 0,
            "semantic_hit": 0,
            "miss": 0,
            "bypass": 0,
            "skipped": 0,
        }
    )
    hit_rate: float = 0.0
    semantic_hit_count: int = 0
    avg_similarity_hits: float = 0.0
    avg_similarity_misses: float = 0.0
    tiers: dict[str, TierStats]


def compute_p95(latencies: list[float]) -> float:
    """Computes the 95th percentile latency using standard linear interpolation."""
    if not latencies:
        return 0.0
    if len(latencies) == 1:
        return round(latencies[0], 2)
    sorted_vals = sorted(latencies)
    k = (len(sorted_vals) - 1) * 0.95
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return round(sorted_vals[f], 2)
    interpolated = sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)
    return round(interpolated, 2)


@router.get("/stats", response_model=StatsResponse, status_code=status.HTTP_200_OK)
async def get_stats(db: AsyncSession = Depends(get_db)) -> Any:
    """Returns aggregated request statistics, token usage, and cache savings per tier and overall."""
    try:
        stmt = select(RequestRecord)
        result = await db.execute(stmt)
        records = result.scalars().all()
    except Exception as exc:
        logger.warning("Database unreachable when querying /v1/stats", extra={"error": str(exc)})
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "detail": "Database is unreachable or unavailable.",
                "error": str(exc),
            },
        )

    # Base configured tiers (ensures small and large are always present)
    configured_tier_names = [t.name for t in settings.tiers]
    for fallback_name in ("small", "large"):
        if fallback_name not in configured_tier_names:
            configured_tier_names.append(fallback_name)

    # Initialize empty statistics with zeros
    tiers_map: dict[str, TierStats] = {
        name: TierStats() for name in configured_tier_names
    }

    if not records:
        return StatsResponse(
            total_requests=0,
            total_prompt_tokens=0,
            total_completion_tokens=0,
            total_tokens_saved=0,
            total_saved_latency_ms=0.0,
            total_saved_shadow_cost_usd=0.0,
            cache_status_counts={
                "exact_hit": 0,
                "semantic_hit": 0,
                "miss": 0,
                "bypass": 0,
                "skipped": 0,
            },
            hit_rate=0.0,
            semantic_hit_count=0,
            avg_similarity_hits=0.0,
            avg_similarity_misses=0.0,
            tiers=tiers_map,
        )

    # Group records by tier
    tier_groups: dict[str, list[RequestRecord]] = {}
    for r in records:
        tier_groups.setdefault(r.tier, []).append(r)

    total_requests = len(records)

    for tier_name, group in tier_groups.items():
        latencies = [float(r.latency_ms) for r in group]
        avg_lat = round(sum(latencies) / len(latencies), 2) if latencies else 0.0
        p95_lat = compute_p95(latencies)

        prompt_tokens = sum(int(getattr(r, "prompt_tokens", 0) or 0) for r in group)
        completion_tokens = sum(int(getattr(r, "completion_tokens", 0) or 0) for r in group)
        shadow_cost = round(sum(float(r.shadow_cost_usd) for r in group), 8)

        status_counts: dict[str, int] = {
            "exact_hit": 0,
            "semantic_hit": 0,
            "miss": 0,
            "bypass": 0,
            "skipped": 0,
        }
        for r in group:
            st = getattr(r, "cache_status", "none") or "none"
            status_counts[st] = status_counts.get(st, 0) + 1

        exact_hits = status_counts.get("exact_hit", 0)
        semantic_hits = status_counts.get("semantic_hit", 0)
        total_hits = exact_hits + semantic_hits
        hit_rate = round(total_hits / len(group), 4) if group else 0.0

        tokens_saved = sum(int(getattr(r, "tokens_saved", 0) or 0) for r in group)
        saved_latency = round(sum(float(getattr(r, "saved_latency_ms", 0.0) or 0.0) for r in group), 2)
        saved_shadow = round(sum(float(getattr(r, "saved_shadow_cost_usd", 0.0) or 0.0) for r in group), 8)

        group_hit_sims = [
            float(r.similarity)
            for r in group
            if getattr(r, "cache_status", None) in ("exact_hit", "semantic_hit")
            and getattr(r, "similarity", None) is not None
        ]
        group_miss_sims = [
            float(r.similarity)
            for r in group
            if getattr(r, "cache_status", None) == "miss"
            and getattr(r, "similarity", None) is not None
        ]
        avg_sim_hits = (
            round(sum(group_hit_sims) / len(group_hit_sims), 4) if group_hit_sims else 0.0
        )
        avg_sim_misses = (
            round(sum(group_miss_sims) / len(group_miss_sims), 4) if group_miss_sims else 0.0
        )

        tiers_map[tier_name] = TierStats(
            request_count=len(group),
            avg_latency_ms=avg_lat,
            p95_latency_ms=p95_lat,
            total_prompt_tokens=prompt_tokens,
            total_completion_tokens=completion_tokens,
            total_shadow_cost_usd=shadow_cost,
            cache_status_counts=status_counts,
            hit_rate=hit_rate,
            total_tokens_saved=tokens_saved,
            total_saved_latency_ms=saved_latency,
            total_saved_shadow_cost_usd=saved_shadow,
            semantic_hit_count=semantic_hits,
            avg_similarity_hits=avg_sim_hits,
            avg_similarity_misses=avg_sim_misses,
        )

    total_prompt_tokens = sum(t.total_prompt_tokens for t in tiers_map.values())
    total_completion_tokens = sum(t.total_completion_tokens for t in tiers_map.values())
    total_tokens_saved = sum(t.total_tokens_saved for t in tiers_map.values())
    total_saved_latency = round(sum(t.total_saved_latency_ms for t in tiers_map.values()), 2)
    total_saved_shadow = round(sum(t.total_saved_shadow_cost_usd for t in tiers_map.values()), 8)

    overall_status_counts: dict[str, int] = {
        "exact_hit": 0,
        "semantic_hit": 0,
        "miss": 0,
        "bypass": 0,
        "skipped": 0,
    }
    for t in tiers_map.values():
        for k, v in t.cache_status_counts.items():
            overall_status_counts[k] = overall_status_counts.get(k, 0) + v

    total_hits = overall_status_counts.get("exact_hit", 0) + overall_status_counts.get("semantic_hit", 0)
    total_hit_rate = round(total_hits / total_requests, 4) if total_requests else 0.0
    total_semantic_hits = overall_status_counts.get("semantic_hit", 0)

    all_hit_sims = [
        float(r.similarity)
        for r in records
        if getattr(r, "cache_status", None) in ("exact_hit", "semantic_hit")
        and getattr(r, "similarity", None) is not None
    ]
    all_miss_sims = [
        float(r.similarity)
        for r in records
        if getattr(r, "cache_status", None) == "miss"
        and getattr(r, "similarity", None) is not None
    ]
    overall_avg_sim_hits = (
        round(sum(all_hit_sims) / len(all_hit_sims), 4) if all_hit_sims else 0.0
    )
    overall_avg_sim_misses = (
        round(sum(all_miss_sims) / len(all_miss_sims), 4) if all_miss_sims else 0.0
    )

    return StatsResponse(
        total_requests=total_requests,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens_saved=total_tokens_saved,
        total_saved_latency_ms=total_saved_latency,
        total_saved_shadow_cost_usd=total_saved_shadow,
        cache_status_counts=overall_status_counts,
        hit_rate=total_hit_rate,
        semantic_hit_count=total_semantic_hits,
        avg_similarity_hits=overall_avg_sim_hits,
        avg_similarity_misses=overall_avg_sim_misses,
        tiers=tiers_map,
    )
