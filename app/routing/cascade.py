import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import ModelTierConfig, settings
from app.llm.client import LLMResponse
from app.routing.validators import validate_answer

logger = logging.getLogger(__name__)


@dataclass
class CascadeResult:
    """Result of running LLM generation through validation and tier cascade."""

    response: LLMResponse
    tier: str
    attempts: list[dict[str, Any]]
    escalation_count: int
    validation_failed: bool
    total_latency_ms: float
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    total_shadow_cost: float
    wasted_latency_ms: float
    wasted_shadow_cost_usd: float


def _resolve_tier_name(tier_input: ModelTierConfig | str) -> str:
    if isinstance(tier_input, ModelTierConfig):
        return tier_input.name
    return str(tier_input)


async def run_cascade(
    prompt: str,
    starting_tier: ModelTierConfig | str,
    llm_client: Any,
    request_context: dict[str, Any] | None = None,
    bypass_cascade: bool = False,
    confidence: float = 0.5,
) -> CascadeResult:
    """Executes prompt against starting tier, escalating to higher tiers if validation fails.

    Rules:
    1. If user explicitly requested a model/tier (bypass_cascade=True), call once without escalation.
    2. Timeouts and connection errors raise immediately (no escalation).
    3. If validation fails, escalate sequentially up to the highest configured tier.
    4. If the highest tier fails validation, return its answer marked validation_failed=True.
    5. Returns cumulative totals and wasted metrics across all attempts.
    """
    configured_tiers = settings.tiers
    start_tier_name = _resolve_tier_name(starting_tier)

    # Find starting tier index in configured tiers
    start_idx = 0
    for idx, t in enumerate(configured_tiers):
        if t.name == start_tier_name or t.model == start_tier_name:
            start_idx = idx
            break

    # If explicit model was requested, execute single attempt on starting tier
    if bypass_cascade:
        target_cfg = (
            starting_tier
            if isinstance(starting_tier, ModelTierConfig)
            else settings.get_tier(start_tier_name)
        )
        candidates = [target_cfg]
    else:
        # Cascade through starting tier and any higher tiers
        candidates = configured_tiers[start_idx:]
        if not candidates:
            target_cfg = (
                starting_tier
                if isinstance(starting_tier, ModelTierConfig)
                else settings.get_tier(start_tier_name)
            )
            candidates = [target_cfg]

    attempts: list[dict[str, Any]] = []
    failed_attempts: list[dict[str, Any]] = []
    winning_response: LLMResponse | None = None
    winning_tier: str = start_tier_name
    validation_failed: bool = False
    total_prompt_tokens_accum: int = 0

    for idx, candidate in enumerate(candidates):
        cand_name = candidate.name if isinstance(candidate, ModelTierConfig) else str(candidate)

        # Generate response (Timeouts and connection errors bubble up immediately)
        response: LLMResponse = await llm_client.generate_response(
            prompt=prompt, tier=candidate
        )

        total_prompt_tokens_accum += response.prompt_tokens

        # Validate answer
        ctx = dict(request_context or {})
        ctx["tier"] = cand_name
        passed, failure_reasons = validate_answer(
            response, request_context=ctx, tier=cand_name
        )

        attempt_record = {
            "tier": cand_name,
            "latency_ms": response.latency_ms,
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "shadow_cost": response.shadow_cost,
            "validation_passed": passed,
            "failure_reasons": failure_reasons,
        }
        attempts.append(attempt_record)

        if passed:
            winning_response = response
            winning_tier = cand_name
            validation_failed = False
            break
        else:
            failed_attempts.append(attempt_record)
            if idx == len(candidates) - 1:
                # Reached highest tier without passing validation
                winning_response = response
                winning_tier = cand_name
                validation_failed = True
                break
            else:
                logger.info(
                    "Validation failed on tier '%s' (reasons: %s); escalating to next tier...",
                    cand_name,
                    failure_reasons,
                )

    if winning_response is None:
        raise RuntimeError("Cascade completed without producing an LLM response.")

    total_latency_ms = round(sum(a["latency_ms"] for a in attempts), 2)
    total_completion_tokens = sum(a["completion_tokens"] for a in attempts)
    total_tokens = total_prompt_tokens_accum + total_completion_tokens
    total_shadow_cost = round(sum(a["shadow_cost"] for a in attempts), 8)
    escalation_count = max(0, len(attempts) - 1)

    wasted_latency_ms = round(sum(a["latency_ms"] for a in failed_attempts), 2)
    wasted_shadow_cost_usd = round(sum(a["shadow_cost"] for a in failed_attempts), 8)

    return CascadeResult(
        response=winning_response,
        tier=winning_tier,
        attempts=attempts,
        escalation_count=escalation_count,
        validation_failed=validation_failed,
        total_latency_ms=total_latency_ms,
        total_prompt_tokens=total_prompt_tokens_accum,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        total_shadow_cost=total_shadow_cost,
        wasted_latency_ms=wasted_latency_ms,
        wasted_shadow_cost_usd=wasted_shadow_cost_usd,
    )
