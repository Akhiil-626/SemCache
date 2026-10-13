import logging
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Protocol
import yaml

from app.core.config import ModelTierConfig, settings

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    tier: str
    desired_tier: str
    reason: str
    score: float
    confidence: float
    rule_hits: list[str] = field(default_factory=list)
    selected_tier: ModelTierConfig | None = None
    selected_model: str | None = None
    estimated_shadow_cost: float = 0.0

    @property
    def rationale(self) -> str:
        return self.reason


class Router(Protocol):
    def route(
        self,
        prompt: str,
        user_budget: float | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RoutingDecision:
        ...


class RuleRouter:
    """Selects the optimal LLM model tier dynamically based on external rule configuration."""

    def __init__(
        self,
        config_path: str | Path | list[ModelTierConfig] | None = None,
        tiers: list[ModelTierConfig] | None = None,
    ) -> None:
        if isinstance(config_path, (list, tuple)):
            tiers = list(config_path)
            config_path = None
        self.config_path = Path(config_path or settings.ROUTING_CONFIG_PATH)
        self._tiers_override = tiers
        self.large_threshold: float = 2.0
        self.default_tier: str = "small"
        self.large_tier: str = "large"
        self.concept_groups: list[dict[str, Any]] = []
        self.feature_rules: list[dict[str, Any]] = []
        self.load_config()

    def load_config(self, raw_data: dict[str, Any] | None = None) -> None:
        """Loads thresholds, tiers, concept groups, and feature rules from YAML or dictionary."""
        if raw_data is not None:
            data = raw_data
        elif self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        else:
            data = {}

        thresholds = data.get("thresholds", {})
        self.large_threshold = float(thresholds.get("large_threshold", 2.0))

        tier_settings = data.get("tiers", {})
        self.default_tier = str(tier_settings.get("default_tier", "small"))
        self.large_tier = str(tier_settings.get("large_tier", "large"))

        self.concept_groups = data.get("concept_groups", [])
        self.feature_rules = data.get("feature_rules", data.get("rules", []))

        # Startup validation: verify each keyword appears in at most one list/group
        seen_words: dict[str, str] = {}
        for group in self.concept_groups:
            group_name = group.get("name", "unnamed")
            for word in group.get("words", []):
                cleaned = word.strip().lower()
                if cleaned in seen_words:
                    logger.warning(
                        "Duplicate routing keyword '%s' found in group '%s' (already in '%s')",
                        cleaned,
                        group_name,
                        seen_words[cleaned],
                    )
                else:
                    seen_words[cleaned] = group_name

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

    def evaluate_features(
        self,
        prompt: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> tuple[float, list[str]]:
        """Evaluates prompt against configured concept groups and feature rules.

        Each group and rule fires at most once per request.
        """
        text = prompt.strip()
        if messages:
            for m in reversed(messages):
                if (isinstance(m, dict) and m.get("role") == "user") or getattr(m, "role", None) == "user":
                    content = m.get("content") if isinstance(m, dict) else getattr(m, "content", "")
                    if content:
                        text = content.strip()
                        break

        words = text.split()
        word_count = len(words)
        msg_count = len(messages) if messages else 1
        non_ascii_chars = sum(1 for c in text if ord(c) > 127)
        non_ascii_ratio = (non_ascii_chars / len(text)) if text else 0.0

        q_count = text.count("?")
        numbered_items = len(
            re.findall(
                r"(?:^|\s)(?:\d+[\)\.]|[a-zA-Z][\)\.]|\b(?:first|second|third|1st|2nd|3rd)\b)",
                text,
                re.IGNORECASE,
            )
        )
        subtask_count = q_count + numbered_items

        total_score = 0.0
        fired_hits: list[str] = []

        # 1. Concept groups: fires AT MOST ONCE per group regardless of variant occurrences
        for group in self.concept_groups:
            g_name = group.get("name", "unnamed")
            weight = float(group.get("weight", 2.0))
            w_str = f"+{int(weight)}" if weight.is_integer() else f"+{weight}"
            words_list = group.get("words", [])

            matched = False
            for kw in words_list:
                if not kw:
                    continue
                # Word boundary check for alphanumeric ASCII words; substring check for non-ASCII/symbols
                if kw.strip() in ("def", "class"):
                    pattern = r"\b" + re.escape(kw.strip()) + r"\s+"
                    if re.search(pattern, text, re.IGNORECASE):
                        matched = True
                        break
                elif kw.isascii() and all(c.isalnum() or c in ' -_' for c in kw):
                    pattern = r"\b" + re.escape(kw) + r"\b"
                    if re.search(pattern, text, re.IGNORECASE):
                        matched = True
                        break
                else:
                    if kw.lower() in text.lower():
                        matched = True
                        break

            if matched:
                total_score += weight
                fired_hits.append(f"group '{g_name}' ({w_str})")

        # 2. Feature rules: fires AT MOST ONCE per rule
        for rule in self.feature_rules:
            rule_type = rule.get("type")
            weight = float(rule.get("weight", 0.0))
            w_str = f"+{int(weight)}" if weight.is_integer() else f"+{weight}"

            if rule_type == "word_count":
                min_w = rule.get("min_words", 0)
                max_w = rule.get("max_words", float("inf"))
                if min_w <= word_count <= max_w:
                    total_score += weight
                    fired_hits.append(f"{rule.get('label', 'word count')} ({w_str})")

            elif rule_type == "regex":
                pattern = rule.get("pattern", "")
                if pattern and re.search(pattern, text, re.IGNORECASE):
                    total_score += weight
                    fired_hits.append(f"{rule.get('label', 'regex pattern')} ({w_str})")

            elif rule_type == "subtasks":
                min_count = rule.get("min_count", 2)
                if subtask_count >= min_count:
                    total_score += weight
                    fired_hits.append(f"{rule.get('label', 'multiple subtasks')} ({w_str})")

            elif rule_type == "message_count":
                min_msgs = rule.get("min_messages", 3)
                if msg_count >= min_msgs:
                    total_score += weight
                    fired_hits.append(f"{rule.get('label', 'multi-turn conversation')} ({w_str})")

            elif rule_type == "non_ascii_ratio":
                min_ratio = float(rule.get("min_ratio", 0.15))
                if non_ascii_ratio >= min_ratio:
                    total_score += weight
                    fired_hits.append(f"{rule.get('label', 'non-ascii prompt')} ({w_str})")

        return round(total_score, 4), fired_hits

    def route(
        self,
        prompt: str,
        user_budget: float | None = None,
        messages: list[dict[str, Any]] | None = None,
        model: str | None = None,
    ) -> RoutingDecision:
        tiers = self.tiers
        if not tiers:
            raise ValueError("No model tiers configured.")

        lowest_tier = tiers[0]
        highest_tier = tiers[-1]
        est_tokens = max(1, len(prompt) // 4)

        # Explicit model bypasses scoring
        if model:
            target_tier = next((t for t in tiers if t.name == model or t.model == model), None)
            if not target_tier:
                target_tier = settings.get_tier(model)
            est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, target_tier), 8)
            return RoutingDecision(
                tier=target_tier.name,
                desired_tier=target_tier.name,
                reason=f"Explicit model '{model}' bypasses scoring.",
                score=0.0,
                confidence=1.0,
                rule_hits=[],
                selected_tier=target_tier,
                selected_model=target_tier.model,
                estimated_shadow_cost=est_shadow,
            )

        # Feature extraction and scoring
        score, rule_hits = self.evaluate_features(prompt=prompt, messages=messages)

        if score >= self.large_threshold:
            desired_tier_cfg = highest_tier
            desired_tier_name = self.large_tier
            confidence = min(1.0, score / max(1.0, self.large_threshold * 1.5))
            reason = f"score {score:.1f} >= {self.large_threshold:.1f}: {', '.join(rule_hits)}"
        else:
            desired_tier_cfg = lowest_tier
            desired_tier_name = self.default_tier
            confidence = max(0.5, 1.0 - (score / max(1.0, self.large_threshold)))
            if rule_hits:
                reason = f"score {score:.1f} < {self.large_threshold:.1f}: {', '.join(rule_hits)}"
            else:
                reason = f"score 0.0 < {self.large_threshold:.1f}: default {lowest_tier.name} tier"

        target_tier = desired_tier_cfg
        final_tier_name = target_tier.name

        # Budget constraints demote down if needed
        if user_budget is not None and len(tiers) > 1 and desired_tier_cfg != lowest_tier:
            for candidate in reversed(tiers):
                candidate_shadow_cost = self._estimate_tier_shadow_cost(est_tokens, candidate)
                if candidate_shadow_cost <= user_budget or candidate == lowest_tier:
                    target_tier = candidate
                    break

            if target_tier != desired_tier_cfg:
                est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, target_tier), 8)
                downgrade_reason = (
                    f"Downgraded to tier '{target_tier.name}' to respect budget limit ${user_budget} "
                    f"(desired: '{desired_tier_name}', {reason})."
                )
                logger.info(
                    "Demoted tier due to budget constraint",
                    extra={"target_tier": target_tier.name, "user_budget": user_budget},
                )
                return RoutingDecision(
                    tier=target_tier.name,
                    desired_tier=desired_tier_name,
                    reason=downgrade_reason,
                    score=score,
                    confidence=confidence,
                    rule_hits=rule_hits,
                    selected_tier=target_tier,
                    selected_model=target_tier.model,
                    estimated_shadow_cost=est_shadow,
                )

        est_shadow = round(self._estimate_tier_shadow_cost(est_tokens, target_tier), 8)
        logger.info(
            "Routed request",
            extra={"tier": target_tier.name, "model": target_tier.model, "score": score},
        )
        return RoutingDecision(
            tier=target_tier.name,
            desired_tier=desired_tier_name,
            reason=reason,
            score=score,
            confidence=confidence,
            rule_hits=rule_hits,
            selected_tier=target_tier,
            selected_model=target_tier.model,
            estimated_shadow_cost=est_shadow,
        )


CostAwareRouter = RuleRouter
cost_router: Router = RuleRouter()
