import json
import logging
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Callable
import zlib
import yaml

from app.core.config import settings

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of an answer validation check."""

    passed: bool
    reason: str | None = None

    def __iter__(self):
        yield self.passed
        yield self.reason

    def __bool__(self) -> bool:
        return self.passed


def _extract_answer_metadata(
    answer: Any, request_context: dict[str, Any] | None = None
) -> tuple[str, int, str]:
    """Extracts content, completion_tokens, and finish_reason from answer and context."""
    context = request_context or {}
    if hasattr(answer, "content"):
        content = getattr(answer, "content", "") or ""
        completion_tokens = getattr(answer, "completion_tokens", 0) or 0
        finish_reason = getattr(answer, "finish_reason", "stop") or "stop"
    elif isinstance(answer, dict):
        content = answer.get("content", "") or ""
        completion_tokens = answer.get(
            "completion_tokens", context.get("completion_tokens", len(content.split()))
        )
        finish_reason = answer.get(
            "finish_reason", context.get("finish_reason", "stop")
        )
    else:
        content = str(answer) if answer is not None else ""
        completion_tokens = context.get(
            "completion_tokens", len(content.split()) if content else 0
        )
        finish_reason = context.get("finish_reason", "stop")

    return content, completion_tokens, finish_reason


def validate_non_empty(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Verifies that completion is non-empty and meets minimum completion token threshold."""
    cfg = config or {}
    context = request_context or {}
    content, completion_tokens, _ = _extract_answer_metadata(answer, context)

    tier = context.get("tier", "default")
    min_tokens_cfg = cfg.get("min_completion_tokens", {})
    if isinstance(min_tokens_cfg, dict):
        min_tokens = min_tokens_cfg.get(tier, min_tokens_cfg.get("default", 1))
    else:
        min_tokens = int(min_tokens_cfg) if min_tokens_cfg else 1

    if not content or not content.strip() or completion_tokens < min_tokens:
        return ValidationResult(False, "empty_response")
    return ValidationResult(True, None)


def validate_not_truncated(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Verifies that response generation completed naturally without token truncation."""
    context = request_context or {}
    _, _, finish_reason = _extract_answer_metadata(answer, context)
    if finish_reason == "length":
        return ValidationResult(False, "truncated")
    return ValidationResult(True, None)


def validate_not_refusal(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Detects model refusals for concise responses (< 60 tokens)."""
    cfg = config or {}
    context = request_context or {}
    content, completion_tokens, _ = _extract_answer_metadata(answer, context)
    max_tokens = cfg.get("max_tokens", 60)

    # Refusals only trigger for short answers under max_tokens
    if completion_tokens >= max_tokens:
        return ValidationResult(True, None)

    patterns = cfg.get("patterns", [
        "i cannot fulfill",
        "i can't fulfill",
        "as an ai",
        "i am an ai",
        "i am unable to",
        "i cannot assist",
        "i apologize, but i cannot",
        "i cannot comply",
        "against my safety guidelines",
    ])
    content_lower = content.lower()
    for pat in patterns:
        if pat.lower() in content_lower:
            return ValidationResult(False, "refusal")

    return ValidationResult(True, None)


def validate_valid_json(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validates JSON structure and optional JSON Schema if response_format was requested."""
    context = request_context or {}
    response_format = context.get("response_format")
    if not response_format:
        return ValidationResult(True, None)

    content, _, _ = _extract_answer_metadata(answer, context)

    try:
        parsed = json.loads(content)
    except Exception:
        return ValidationResult(False, "invalid_json")

    # If schema is given in response_format or request_context, validate schema
    schema = None
    if isinstance(response_format, dict):
        if response_format.get("type") == "json_schema":
            schema = response_format.get("json_schema", {}).get("schema")
        elif "schema" in response_format:
            schema = response_format["schema"]
    if not schema and "schema" in context:
        schema = context["schema"]

    if schema:
        try:
            import jsonschema
            jsonschema.validate(instance=parsed, schema=schema)
        except Exception:
            return ValidationResult(False, "schema_mismatch")

    return ValidationResult(True, None)


def validate_no_repetition(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> ValidationResult:
    """Detects degenerative repetition loops via zlib compression and repeated n-grams."""
    cfg = config or {}
    context = request_context or {}
    content, _, _ = _extract_answer_metadata(answer, context)

    min_chars = cfg.get("min_characters", 200)
    if len(content) < min_chars:
        return ValidationResult(True, None)

    # 1. zlib compression ratio check
    zlib_threshold = cfg.get("zlib_compression_threshold", 0.20)
    raw_bytes = content.encode("utf-8")
    if raw_bytes:
        compressed = zlib.compress(raw_bytes)
        ratio = len(compressed) / len(raw_bytes)
        if ratio < zlib_threshold:
            return ValidationResult(False, "repetition_loop")

    # 2. Repeated n-gram check
    ngram_size = cfg.get("ngram_size", 4)
    max_repeats = cfg.get("max_ngram_repeats", 3)
    words = re.findall(r"\w+", content.lower())
    if len(words) >= ngram_size * 2:
        ngram_counts: dict[tuple[str, ...], int] = {}
        consecutive_repeats = 1
        last_ngram = None
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i : i + ngram_size])
            if ngram == last_ngram:
                consecutive_repeats += 1
                if consecutive_repeats >= max_repeats:
                    return ValidationResult(False, "repetition_ngram")
            else:
                consecutive_repeats = 1
                last_ngram = ngram

            ngram_counts[ngram] = ngram_counts.get(ngram, 0) + 1
            if ngram_counts[ngram] >= max_repeats * 2:
                return ValidationResult(False, "repetition_ngram")

    return ValidationResult(True, None)


VALIDATOR_REGISTRY: dict[str, Callable[..., ValidationResult]] = {
    "non_empty": validate_non_empty,
    "not_truncated": validate_not_truncated,
    "not_refusal": validate_not_refusal,
    "valid_json": validate_valid_json,
    "no_repetition": validate_no_repetition,
}


class ValidationPipeline:
    """Manages dynamic validator configuration and tier-specific execution."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.config_path = Path(config_path or "config/validators.yaml")
        self.validators_config: list[dict[str, Any]] = []
        self.tier_rules: dict[str, dict[str, Any]] = {}
        self.load_config()

    def load_config(self) -> None:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            self.validators_config = data.get("validators", [])
            self.tier_rules = data.get("tier_rules", {})
        else:
            self.validators_config = [
                {"name": name, "enabled": True} for name in VALIDATOR_REGISTRY
            ]
            self.tier_rules = {}

    def is_validator_enabled_for_tier(self, validator_name: str, tier: str) -> bool:
        if tier in self.tier_rules:
            enabled_list = self.tier_rules[tier].get("enabled_validators", [])
            return validator_name in enabled_list
        # If no tier-specific rule, check validator level config
        for v in self.validators_config:
            if v.get("name") == validator_name:
                return v.get("enabled", True)
        return True

    def validate(
        self,
        answer: Any,
        request_context: dict[str, Any] | None = None,
        tier: str | None = None,
    ) -> tuple[bool, list[str]]:
        context = dict(request_context or {})
        resolved_tier = tier or context.get("tier", "small")
        context["tier"] = resolved_tier

        failure_reasons: list[str] = []

        for v_cfg in self.validators_config:
            name = v_cfg.get("name")
            if not name or name not in VALIDATOR_REGISTRY:
                continue

            if not self.is_validator_enabled_for_tier(name, resolved_tier):
                continue

            validator_fn = VALIDATOR_REGISTRY[name]
            result = validator_fn(answer, context, v_cfg)
            if not result.passed:
                reason = result.reason or f"{name}_failed"
                failure_reasons.append(reason)

        return (len(failure_reasons) == 0, failure_reasons)


validation_pipeline = ValidationPipeline()


def validate_answer(
    answer: Any,
    request_context: dict[str, Any] | None = None,
    tier: str | None = None,
) -> tuple[bool, list[str]]:
    return validation_pipeline.validate(answer, request_context, tier=tier)


run_validators = validate_answer

