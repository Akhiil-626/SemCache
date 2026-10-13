import logging
import time
from dataclasses import dataclass
from typing import Any
from sqlalchemy.ext.asyncio import AsyncSession

from app.cache.backend import ExactCacheBackend, get_exact_cache_backend
from app.cache.key import build_exact_cache_key
from app.cache.policies import evaluate_cache_policy, is_informational_query
from app.cache.service import cache_service
from app.core.config import settings
from app.db.models import RequestRecord
from app.embeddings.service import embedding_service
from app.llm.client import llm_client
from app.routing.cascade import run_cascade
from app.routing.router import cost_router

logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    content: str
    cached: bool
    tier: str
    model_name: str
    similarity_score: float
    cost_usd: float
    shadow_cost_usd: float
    latency_ms: float
    routing_rationale: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_status: str = "none"
    tokens_saved: int = 0
    saved_latency_ms: float = 0.0
    saved_shadow_cost_usd: float = 0.0
    validation_failed: bool = False
    escalation_count: int = 0
    attempts: list[dict[str, Any]] | None = None



class CompletionService:
    """Core pipeline service encapsulating exact caching, semantic caching fallback,
    N-tier routing, LLM generation, cost tracking, and persistent request logging.
    """

    def __init__(self, exact_backend: ExactCacheBackend | None = None) -> None:
        self._exact_backend = exact_backend

    @property
    def exact_backend(self) -> ExactCacheBackend:
        if self._exact_backend is None:
            self._exact_backend = get_exact_cache_backend()
        return self._exact_backend

    @exact_backend.setter
    def exact_backend(self, backend: ExactCacheBackend) -> None:
        self._exact_backend = backend

    async def _safe_log_request(self, db: AsyncSession, **kwargs: Any) -> None:
        """Persists request execution log, ensuring DB write failures never crash the request."""
        try:
            req_log = RequestRecord(**kwargs)
            db.add(req_log)
            await db.commit()
        except Exception as exc:
            logger.warning("Failed to persist request log to database", extra={"error": str(exc)})
            try:
                await db.rollback()
            except Exception:
                pass

    async def execute_completion(
        self,
        prompt: str,
        db: AsyncSession,
        user_budget: float | None = None,
        bypass_cache: bool = False,
        preferred_tier: str | None = None,
        tenant_id: str = "default",
        messages: list[dict[str, Any]] | list[Any] | None = None,
        temperature: float | None = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        cache_control: str | None = None,
        response_format: Any = None,
    ) -> CompletionResult:
        start_time = time.perf_counter()

        # 1. Normalize messages and prompt
        if messages is None:
            normalized_msgs: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
        else:
            normalized_msgs = [
                {"role": m["role"], "content": m["content"]}
                if isinstance(m, dict)
                else {"role": getattr(m, "role", "user"), "content": getattr(m, "content", "")}
                for m in messages
            ]
            if not prompt and normalized_msgs:
                prompt = normalized_msgs[-1]["content"]

        # Extract normalized last user message for semantic embedding
        last_user_content = ""
        for m in reversed(normalized_msgs):
            if m.get("role") == "user":
                last_user_content = m.get("content", "")
                break
        if not last_user_content:
            last_user_content = prompt
        query_text = last_user_content.strip()

        # 2. Check cache bypass conditions
        is_bypass = bypass_cache or (
            cache_control is not None and "no-cache" in cache_control.lower()
        )

        # 3. Determine tier and model routing upfront
        if preferred_tier:
            routing_decision = cost_router.route(
                prompt=query_text or prompt,
                user_budget=user_budget,
                messages=normalized_msgs,
                model=preferred_tier,
            )
        else:
            routing_decision = cost_router.route(
                prompt=query_text or prompt,
                user_budget=user_budget,
                messages=normalized_msgs,
            )

        tier_cfg = routing_decision.selected_tier or settings.get_tier(routing_decision.tier)
        routing_rationale = routing_decision.reason
        tier_name = routing_decision.tier
        model_name = tier_cfg.model

        # 4. Exact & Semantic Cache: Policy Check, Key Generation & Lookup
        cache_key: str | None = None
        policy_decision = None
        query_vector: list[float] | None = None
        best_similarity: float | None = None
        is_info_query: bool = True

        cache_status = "miss"
        if is_bypass:
            cache_status = "bypass"
        else:
            try:
                policy_decision = evaluate_cache_policy(
                    messages=normalized_msgs,
                    temperature=temperature,
                    stream=stream,
                )

                if not policy_decision.cacheable:
                    cache_status = "skipped"
                    logger.info(
                        "Cache policy rejected caching: reason=%s (skipping exact and semantic lookup)",
                        policy_decision.reason,
                    )
                elif not settings.EXACT_CACHE_ENABLED and not settings.SEMANTIC_CACHE_ENABLED:
                    cache_status = "skipped"
                else:
                    if settings.EXACT_CACHE_ENABLED:
                        cache_key = build_exact_cache_key(
                            tenant_id=tenant_id,
                            tier=tier_name,
                            model=model_name,
                            messages=normalized_msgs,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            cache_version=settings.CACHE_VERSION,
                        )

                # 5. Exact Cache: Lookup
                if cache_key is not None and cache_status == "miss":
                    t_lookup_start = time.perf_counter()
                    cached_entry = await self.exact_backend.get(cache_key)
                    lookup_latency_ms = (time.perf_counter() - t_lookup_start) * 1000.0

                    if cached_entry is not None:
                        # EXACT CACHE HIT
                        orig_prompt_tokens = int(cached_entry.get("prompt_tokens", 0))
                        orig_completion_tokens = int(cached_entry.get("completion_tokens", 0))
                        orig_latency_ms = float(cached_entry.get("latency_ms", 0.0))
                        orig_shadow_cost = float(cached_entry.get("shadow_cost_usd", 0.0))
                        cached_content = str(cached_entry.get("content", ""))
                        tokens_saved = orig_prompt_tokens + orig_completion_tokens

                        # Log to DB: 0 consumed tokens/costs, record original metrics as saved
                        await self._safe_log_request(
                            db=db,
                            prompt=prompt,
                            response=cached_content,
                            tier=cached_entry.get("tier", tier_name),
                            model_name=cached_entry.get("model_name", model_name),
                            cached=True,
                            cost_usd=0.0,
                            shadow_cost_usd=0.0,
                            latency_ms=round(lookup_latency_ms, 2),
                            tenant_id=tenant_id,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cache_status="exact_hit",
                            tokens_saved=tokens_saved,
                            saved_latency_ms=round(orig_latency_ms, 2),
                            saved_shadow_cost_usd=round(orig_shadow_cost, 8),
                            similarity=None,
                            status="ok",
                            error_message=None,
                            routing_rationale="Retrieved from exact key-value cache.",
                            routing_score=routing_decision.score,
                            desired_tier=routing_decision.desired_tier,
                        )

                        # Return response with original tokens in usage and 0 for costs
                        return CompletionResult(
                            content=cached_content,
                            cached=True,
                            tier=cached_entry.get("tier", tier_name),
                            model_name=cached_entry.get("model_name", model_name),
                            similarity_score=1.0,
                            cost_usd=0.0,
                            shadow_cost_usd=0.0,
                            latency_ms=round(lookup_latency_ms, 2),
                            routing_rationale="Retrieved from exact key-value cache.",
                            prompt_tokens=orig_prompt_tokens,
                            completion_tokens=orig_completion_tokens,
                            cache_status="exact_hit",
                            tokens_saved=tokens_saved,
                            saved_latency_ms=round(orig_latency_ms, 2),
                            saved_shadow_cost_usd=round(orig_shadow_cost, 8),
                        )

                # 6. Semantic Cache: Lookup (after exact cache miss)
                if settings.SEMANTIC_INFO_ONLY:
                    is_info_query = is_informational_query(query_text)
                    if not is_info_query:
                        logger.info(
                            "Semantic cache lookup skipped: reason=not_informational (query=%r)",
                            query_text,
                        )

                if (
                    settings.SEMANTIC_CACHE_ENABLED
                    and is_info_query
                    and cache_status == "miss"
                    and not is_bypass
                    and (policy_decision is None or policy_decision.cacheable)
                ):
                    t_sem_start = time.perf_counter()
                    query_vector = await embedding_service.get_embedding(query_text)
                    sem_hit = await cache_service.lookup(
                        db=db,
                        query_embedding=query_vector,
                        query_text=query_text,
                        tenant_id=tenant_id,
                        tier=tier_name,
                    )
                    sem_latency_ms = (time.perf_counter() - t_sem_start) * 1000.0
                    best_similarity = sem_hit.similarity_score

                    if sem_hit.is_hit and sem_hit.cached_response:
                        # SEMANTIC CACHE HIT
                        orig_prompt_tokens = sem_hit.prompt_tokens
                        orig_completion_tokens = sem_hit.completion_tokens
                        orig_latency_ms = sem_hit.latency_ms
                        orig_shadow_cost = sem_hit.shadow_cost_usd
                        tokens_saved = orig_prompt_tokens + orig_completion_tokens

                        await self._safe_log_request(
                            db=db,
                            prompt=prompt,
                            response=sem_hit.cached_response,
                            tier=sem_hit.tier or tier_name,
                            model_name=sem_hit.original_model or model_name,
                            cached=True,
                            cost_usd=0.0,
                            shadow_cost_usd=0.0,
                            latency_ms=round(sem_latency_ms, 2),
                            tenant_id=tenant_id,
                            prompt_tokens=0,
                            completion_tokens=0,
                            cache_status="semantic_hit",
                            tokens_saved=tokens_saved,
                            saved_latency_ms=round(orig_latency_ms, 2),
                            saved_shadow_cost_usd=round(orig_shadow_cost, 8),
                            similarity=best_similarity,
                            status="ok",
                            error_message=None,
                            routing_rationale="Retrieved from semantic vector cache.",
                            routing_score=routing_decision.score,
                            desired_tier=routing_decision.desired_tier,
                        )

                        return CompletionResult(
                            content=sem_hit.cached_response,
                            cached=True,
                            tier=sem_hit.tier or tier_name,
                            model_name=sem_hit.original_model or model_name,
                            similarity_score=best_similarity or 0.0,
                            cost_usd=0.0,
                            shadow_cost_usd=0.0,
                            latency_ms=round(sem_latency_ms, 2),
                            routing_rationale="Retrieved from semantic vector cache.",
                            prompt_tokens=orig_prompt_tokens,
                            completion_tokens=orig_completion_tokens,
                            cache_status="semantic_hit",
                            tokens_saved=tokens_saved,
                            saved_latency_ms=round(orig_latency_ms, 2),
                            saved_shadow_cost_usd=round(orig_shadow_cost, 8),
                        )
                    elif sem_hit.guard_rejected:
                        logger.info(
                            "Semantic cache candidate rejected by guard: reason=%s (query=%r)",
                            sem_hit.guard_reason,
                            query_text,
                        )
            except Exception as exc:
                cache_status = "bypass"
                logger.warning(
                    "Exact/semantic cache lookup error: %s: %s",
                    type(exc).__name__,
                    str(exc),
                )

        # 7. LLM Provider Execution via Validation Cascade
        request_ctx = {
            "response_format": response_format,
            "messages": normalized_msgs,
        }
        bypass_cascade = bool(preferred_tier)

        try:
            cascade_result = await run_cascade(
                prompt=prompt,
                starting_tier=tier_cfg,
                llm_client=llm_client,
                request_context=request_ctx,
                bypass_cascade=bypass_cascade,
                confidence=routing_decision.confidence,
            )
        except Exception as llm_exc:
            total_err_latency = (time.perf_counter() - start_time) * 1000.0
            await self._safe_log_request(
                db=db,
                prompt=prompt,
                response="",
                tier=tier_name,
                model_name=model_name,
                cached=False,
                cost_usd=0.0,
                shadow_cost_usd=0.0,
                latency_ms=round(total_err_latency, 2),
                tenant_id=tenant_id,
                prompt_tokens=0,
                completion_tokens=0,
                cache_status=cache_status,
                tokens_saved=0,
                saved_latency_ms=0.0,
                saved_shadow_cost_usd=0.0,
                similarity=best_similarity,
                status="error",
                error_message=str(llm_exc),
                routing_rationale=routing_rationale,
                routing_score=routing_decision.score,
                desired_tier=routing_decision.desired_tier,
                escalation_count=0,
                attempts=None,
                validation_failed=False,
                wasted_latency_ms=0.0,
                wasted_shadow_cost_usd=0.0,
            )
            raise

        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        final_response = cascade_result.response
        final_tier_name = cascade_result.tier
        winning_model_name = final_response.model_name

        # 8. Store in Exact Cache on Miss (Only if validation passed)
        if (
            not cascade_result.validation_failed
            and cache_key is not None
            and cache_status == "miss"
            and settings.EXACT_CACHE_ENABLED
            and not is_bypass
            and (policy_decision is None or policy_decision.cacheable)
        ):
            try:
                store_payload = {
                    "content": final_response.content,
                    "model_name": winning_model_name,
                    "tier": final_tier_name,
                    "prompt_tokens": cascade_result.total_prompt_tokens,
                    "completion_tokens": cascade_result.total_completion_tokens,
                    "latency_ms": round(cascade_result.total_latency_ms, 2),
                    "shadow_cost_usd": cascade_result.total_shadow_cost,
                }
                await self.exact_backend.set(
                    cache_key,
                    store_payload,
                    ttl_seconds=settings.CACHE_TTL_SECONDS,
                )
            except Exception as exc:
                cache_status = "bypass"
                logger.warning(
                    "Exact cache store error: %s: %s",
                    type(exc).__name__,
                    str(exc),
                )

        # 9. Store in Semantic Cache on Miss (Only if validation passed)
        if (
            not cascade_result.validation_failed
            and settings.SEMANTIC_CACHE_ENABLED
            and is_info_query
            and cache_status == "miss"
            and not is_bypass
            and (policy_decision is None or policy_decision.cacheable)
        ):
            try:
                if query_vector is None:
                    query_vector = await embedding_service.get_embedding(query_text)
                if query_vector is not None:
                    await cache_service.store(
                        db=db,
                        prompt=query_text,
                        embedding=query_vector,
                        response=final_response.content,
                        model_name=winning_model_name,
                        tier=final_tier_name,
                        cost_usd=final_response.real_cost,
                        shadow_cost_usd=cascade_result.total_shadow_cost,
                        prompt_tokens=cascade_result.total_prompt_tokens,
                        completion_tokens=cascade_result.total_completion_tokens,
                        latency_ms=round(cascade_result.total_latency_ms, 2),
                        tenant_id=tenant_id,
                        cache_version=settings.CACHE_VERSION,
                        ttl_seconds=settings.CACHE_TTL_SECONDS,
                    )
            except Exception as cache_exc:
                cache_status = "bypass"
                logger.warning(
                    "Semantic cache store error: %s: %s",
                    type(cache_exc).__name__,
                    str(cache_exc),
                )

        # 10. Persist Successful Request Log
        await self._safe_log_request(
            db=db,
            prompt=prompt,
            response=final_response.content,
            tier=final_tier_name,
            model_name=winning_model_name,
            cached=False,
            cost_usd=final_response.real_cost,
            shadow_cost_usd=cascade_result.total_shadow_cost,
            latency_ms=round(total_latency_ms, 2),
            tenant_id=tenant_id,
            prompt_tokens=cascade_result.total_prompt_tokens,
            completion_tokens=cascade_result.total_completion_tokens,
            cache_status=cache_status,
            tokens_saved=0,
            saved_latency_ms=0.0,
            saved_shadow_cost_usd=0.0,
            similarity=best_similarity,
            status="ok",
            error_message=None,
            routing_rationale=routing_rationale,
            routing_score=routing_decision.score,
            desired_tier=routing_decision.desired_tier,
            escalation_count=cascade_result.escalation_count,
            attempts=cascade_result.attempts,
            validation_failed=cascade_result.validation_failed,
            wasted_latency_ms=round(cascade_result.wasted_latency_ms, 2),
            wasted_shadow_cost_usd=round(cascade_result.wasted_shadow_cost_usd, 8),
        )

        return CompletionResult(
            content=final_response.content,
            cached=False,
            tier=final_tier_name,
            model_name=winning_model_name,
            similarity_score=best_similarity or 0.0,
            cost_usd=final_response.real_cost,
            shadow_cost_usd=cascade_result.total_shadow_cost,
            latency_ms=round(total_latency_ms, 2),
            routing_rationale=routing_rationale,
            prompt_tokens=cascade_result.total_prompt_tokens,
            completion_tokens=cascade_result.total_completion_tokens,
            cache_status=cache_status,
            tokens_saved=0,
            saved_latency_ms=0.0,
            saved_shadow_cost_usd=0.0,
            validation_failed=cascade_result.validation_failed,
            escalation_count=cascade_result.escalation_count,
            attempts=cascade_result.attempts,
        )


completion_service = CompletionService()
