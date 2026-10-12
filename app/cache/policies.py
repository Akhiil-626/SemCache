import re
from dataclasses import dataclass
from typing import Any

TIME_SENSITIVE_WORDS = {"today", "now", "latest", "current", "price", "weather"}
TIME_SENSITIVE_REGEX = re.compile(
    r"\b(" + "|".join(re.escape(w) for w in sorted(TIME_SENSITIVE_WORDS)) + r")\b",
    re.IGNORECASE,
)


@dataclass
class PolicyDecision:
    cacheable: bool
    reason: str


def evaluate_cache_policy(
    messages: list[dict[str, Any]] | list[Any],
    temperature: float | None = 0.7,
    stream: bool = False,
    tools: list[Any] | None = None,
    tool_choice: Any | None = None,
) -> PolicyDecision:
    """Evaluates whether an incoming LLM completion request is eligible for exact caching."""
    if stream:
        return PolicyDecision(cacheable=False, reason="streaming")

    if temperature is not None and temperature > 0.7:
        return PolicyDecision(cacheable=False, reason="high_temperature")

    if tools or tool_choice:
        return PolicyDecision(cacheable=False, reason="tool_calls")

    for msg in messages:
        if isinstance(msg, dict):
            content = str(msg.get("content", ""))
        else:
            content = str(getattr(msg, "content", ""))

        if TIME_SENSITIVE_REGEX.search(content):
            return PolicyDecision(cacheable=False, reason="time_sensitive_keywords")

    return PolicyDecision(cacheable=True, reason="ok")


INFORMATIONAL_PREFIXES = (
    "what",
    "why",
    "how",
    "who",
    "when",
    "where",
    "which",
    "is",
    "are",
    "does",
    "do",
    "can",
    "should",
    "explain",
    "describe",
    "summarize",
    "define",
    "tell me about",
    "steps for",
    "difference",
)


def is_informational_query(text: str) -> bool:
    """Checks whether the query starts with an informational question prefix.
    
    Allowlisted prefixes include:
    what, why, how, who, when, where, which, is, are, does, do, can, should,
    explain, describe, summarize, define, tell me about, steps for, difference.
    """
    clean_text = text.strip().lower()
    # Strip leading punctuation/quotes/bullets e.g. "what...", "- how..."
    clean_text = re.sub(r"^[\s\"'#*\->]+", "", clean_text)
    for prefix in INFORMATIONAL_PREFIXES:
        if clean_text.startswith(prefix):
            rest = clean_text[len(prefix):]
            # Match if prefix is followed by word boundary, space, punctuation, or end of string
            if not rest or not rest[0].isalpha():
                return True
    return False

