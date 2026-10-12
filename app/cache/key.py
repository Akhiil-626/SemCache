import hashlib
import json
import re
from typing import Any

from app.core.config import settings


def collapse_whitespace(text: str) -> str:
    """Trims and collapses contiguous whitespace into single spaces without lowercasing."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_messages(messages: list[dict[str, Any]] | list[Any]) -> list[dict[str, str]]:
    """Normalizes message objects to canonical dictionaries preserving letter casing."""
    canonical: list[dict[str, str]] = []
    for msg in messages:
        if isinstance(msg, dict):
            role = str(msg.get("role", "")).strip()
            content = collapse_whitespace(str(msg.get("content", "")))
        else:
            role = str(getattr(msg, "role", "")).strip()
            content = collapse_whitespace(str(getattr(msg, "content", "")))
        canonical.append({"role": role, "content": content})
    return canonical


def build_exact_cache_key(
    tenant_id: str,
    tier: str,
    model: str,
    messages: list[dict[str, Any]] | list[Any],
    temperature: float | None = 0.7,
    max_tokens: int | None = None,
    cache_version: str | None = None,
) -> str:
    """Computes sha256 of canonical JSON representation for exact matching."""
    version = cache_version or settings.CACHE_VERSION
    canonical_msgs = normalize_messages(messages)

    canonical_payload = {
        "cache_version": version,
        "max_tokens": max_tokens,
        "messages": canonical_msgs,
        "model": model.strip(),
        "temperature": round(temperature, 4) if temperature is not None else None,
        "tenant_id": tenant_id.strip(),
        "tier": tier.strip(),
    }

    canonical_json = json.dumps(canonical_payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
