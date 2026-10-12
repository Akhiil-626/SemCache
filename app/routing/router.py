import logging
from dataclasses import dataclass

from app.core.config import ModelTierConfig, settings

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    selected_tier: ModelTierConfig
    selected_model: str
    tier: str
    rationale: str
    estimated_shadow_cost: float


class CostAwareRouter:
    """Selects the optimal LLM model tier dynamically across N ordered tiers from configuration."""

    REASONING_KEYWORDS = {
        "analyze", "proof", "prove", "architect", "refactor",
        "compare", "critique", "derivation", "step by step",
        "optimize", "algorithm", "complex", "reasoning",
    }

    def __init__(self, tiers: list[ModelTierConfig] | None = None) -> None:
        self._tiers_override = tiers

    @property
    def tiers(self) -> list[ModelTierConfig]:
        """Ordered list of tiers, cheapest first."""
        return self._tiers_override or settings.tiers

    def _estimate_tier_shadow_cost(self, prompt_tokens: int, tier: ModelTierConfig) -> float:
        """Estimates request shadow cost assuming ~2x completion token ratio."""
        est_completion = prompt_tokens * 2
        return (prompt_tokens * tier.input_price / 1_000_000.0) + (
            est_completion * tier.output_price / 1_000_000.0
        )

    def route(self, prompt: str, user_budget: float | None = None) -> RoutingDecision:
        tiers = self.tiers
        if not tiers:
            raise ValueError("No model tiers configured.")

        lowest_tier = tiers[0]
        highest_tier = tiers[-1]

        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        has_complex_keywords = any(kw in prompt_lower for kw in self.REASONING_KEYWORDS)
        is_lengthy = word_count > 60
        is_complex = has_complex_keywords or is_lengthy

        est_tokens = max(1, len(prompt) // 4)

        # Standard simple queries route to the lowest tier
        if not is_complex:
            est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, lowest_tier), 8)
            logger.info("Routed to baseline tier", extra={"tier": lowest_tier.name, "model": lowest_tier.model})
            return RoutingDecision(
                selected_tier=lowest_tier,
                selected_model=lowest_tier.model,
                tier=lowest_tier.name,
                rationale=f"Standard complexity query routed to lowest tier ({lowest_tier.name}).",
                estimated_shadow_cost=est_shadow,
            )

        # For complex queries, attempt highest capability tier down to lowest if constrained by budget
        target_tier = highest_tier
        if user_budget is not None and len(tiers) > 1:
            for candidate in reversed(tiers):
                candidate_shadow_cost = self._estimate_tier_shadow_cost(est_tokens, candidate)
                if candidate_shadow_cost <= user_budget or candidate == lowest_tier:
                    target_tier = candidate
                    break

            if target_tier != highest_tier:
                est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, target_tier), 8)
                logger.info(
                    "Demoted tier due to budget constraint",
                    extra={"target_tier": target_tier.name, "user_budget": user_budget},
                )
                return RoutingDecision(
                    selected_tier=target_tier,
                    selected_model=target_tier.model,
                    tier=target_tier.name,
                    rationale=f"Downgraded to tier '{target_tier.name}' to respect budget limit ${user_budget}.",
                    estimated_shadow_cost=est_shadow,
                )

        est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, target_tier), 8)
        logger.info("Routed to highest capability tier", extra={"tier": target_tier.name, "model": target_tier.model})
        return RoutingDecision(
            selected_tier=target_tier,
            selected_model=target_tier.model,
            tier=target_tier.name,
            rationale=f"High complexity query routed to tier '{target_tier.name}'.",
            estimated_shadow_cost=est_shadow,
        )


cost_router = CostAwareRouter()
