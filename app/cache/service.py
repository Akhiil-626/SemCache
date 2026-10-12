import logging
from datetime import datetime, timedelta, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.policies import is_informational_query
from app.cache.schemas import CacheHitResult
from app.cache.semantic_guard import check_semantic_guard
from app.core.config import settings
from app.db.models import SemanticCacheEntry

logger = logging.getLogger(__name__)


class SemanticCacheService:
    """Manages vector-similarity lookup and persistence in pgvector with metadata filtering and guard checks."""

    def __init__(self, threshold: float = settings.SIMILARITY_THRESHOLD) -> None:
        self.threshold = threshold

    async def lookup(
        self,
        db: AsyncSession,
        query_embedding: list[float] | None,
        query_text: str,
        tenant_id: str = "default",
        tier: str = "small",
    ) -> CacheHitResult:
        """Finds closest cached entry by cosine distance. Returns hit only if similarity >= threshold and guard passes."""
        if settings.SEMANTIC_INFO_ONLY and not is_informational_query(query_text):
            logger.info(
                "Semantic cache lookup skipped: reason=not_informational (query=%r)",
                query_text,
            )
            return CacheHitResult(is_hit=False, similarity_score=None)

        if query_embedding is None:
            return CacheHitResult(is_hit=False, similarity_score=None)

        try:
            now = datetime.now(timezone.utc)
            distance_col = SemanticCacheEntry.embedding.cosine_distance(query_embedding).label("distance")
            stmt = (
                select(SemanticCacheEntry, distance_col)
                .where(
                    SemanticCacheEntry.tenant_id == tenant_id,
                    SemanticCacheEntry.tier == tier,
                    SemanticCacheEntry.cache_version == settings.CACHE_VERSION,
                    SemanticCacheEntry.expires_at > now,
                )
                .order_by(distance_col)
                .limit(1)
            )

            result = await db.execute(stmt)
            row = result.first()

            if not row:
                logger.info(
                    "Semantic cache lookup: best_similarity=none (no candidates found for tenant=%s, tier=%s, version=%s)",
                    tenant_id,
                    tier,
                    settings.CACHE_VERSION,
                )
                return CacheHitResult(is_hit=False, similarity_score=None)

            entry, distance = row
            similarity = round(1.0 - float(distance), 4)

            # Log best similarity seen on EVERY lookup at INFO
            logger.info(
                "Semantic cache lookup: best_similarity=%.4f threshold=%.2f candidate_id=%s",
                similarity,
                self.threshold,
                str(entry.id),
            )

            if similarity >= self.threshold:
                # Run negation and number guard
                guard_ok, guard_reason = check_semantic_guard(query_text, entry.prompt)
                if not guard_ok:
                    logger.info(
                        "Semantic cache guard rejected candidate: reason=%s (query=%r vs cached=%r)",
                        guard_reason,
                        query_text,
                        entry.prompt,
                    )
                    return CacheHitResult(
                        is_hit=False,
                        similarity_score=similarity,
                        guard_rejected=True,
                        guard_reason="guard",
                    )

                entry.hit_count += 1
                entry.last_hit_at = datetime.now(timezone.utc)
                await db.commit()

                return CacheHitResult(
                    is_hit=True,
                    similarity_score=similarity,
                    cached_response=entry.response,
                    original_model=entry.model_name,
                    tier=entry.tier,
                    shadow_cost_usd=entry.shadow_cost_usd,
                    prompt_tokens=entry.prompt_tokens,
                    completion_tokens=entry.completion_tokens,
                    latency_ms=entry.latency_ms,
                    entry_id=entry.id,
                )

            return CacheHitResult(is_hit=False, similarity_score=similarity)

        except Exception as e:
            logger.warning("Error performing semantic cache lookup: %s: %s", type(e).__name__, str(e))
            return CacheHitResult(is_hit=False, similarity_score=None)

    async def store(
        self,
        db: AsyncSession,
        prompt: str,
        embedding: list[float],
        response: str,
        model_name: str,
        tier: str = "small",
        cost_usd: float = 0.0,
        shadow_cost_usd: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        tenant_id: str = "default",
        cache_version: str | None = None,
        ttl_seconds: int | None = None,
    ) -> SemanticCacheEntry | None:
        """Stores a new prompt-response pair and its vector embedding with metadata and TTL."""
        if settings.SEMANTIC_INFO_ONLY and not is_informational_query(prompt):
            logger.info(
                "Semantic cache store skipped: reason=not_informational (prompt=%r)",
                prompt,
            )
            return None

        ttl = ttl_seconds or settings.CACHE_TTL_SECONDS
        version = cache_version or settings.CACHE_VERSION
        now = datetime.now(timezone.utc)
        expires_at = now + timedelta(seconds=ttl)

        entry = SemanticCacheEntry(
            prompt=prompt,
            embedding=embedding,
            response=response,
            tier=tier,
            model_name=model_name,
            cost_usd=cost_usd,
            shadow_cost_usd=shadow_cost_usd,
            tenant_id=tenant_id,
            cache_version=version,
            expires_at=expires_at,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        db.add(entry)
        await db.commit()
        await db.refresh(entry)

        logger.info(
            "Persisted new semantic cache record: entry_id=%s tier=%s model=%s expires_at=%s",
            str(entry.id),
            tier,
            model_name,
            expires_at.isoformat(),
        )
        return entry


cache_service = SemanticCacheService()
