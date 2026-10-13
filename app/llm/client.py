import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Protocol

import httpx
import litellm
from litellm import acompletion

from app.core.config import ModelTierConfig, settings

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    real_cost: float
    shadow_cost: float
    model_name: str
    tier: str
    finish_reason: str = "stop"


class LLMClientInterface(Protocol):
    async def generate_response(
        self, prompt: str, tier: ModelTierConfig | str
    ) -> LLMResponse:
        ...


class LLMClient:
    """Invokes local Ollama models via LiteLLM with timeout, retry, and shadow cost calculation."""

    def __init__(self, retry_count: int = settings.OLLAMA_RETRY_COUNT) -> None:
        self.retry_count = retry_count

    def _resolve_tier(self, tier_input: ModelTierConfig | str) -> ModelTierConfig:
        if isinstance(tier_input, ModelTierConfig):
            return tier_input
        # Look up by tier name or model name
        for t in settings.tiers:
            if t.name == tier_input or t.model == tier_input:
                return t
        return settings.get_tier(tier_input)

    async def generate_response(
        self, prompt: str, tier: ModelTierConfig | str = "small"
    ) -> LLMResponse:
        tier_cfg = self._resolve_tier(tier)
        api_base = tier_cfg.api_base or settings.OLLAMA_BASE_URL
        start_time = time.perf_counter()

        last_error: Exception | None = None
        response: Any = None

        for attempt in range(self.retry_count + 1):
            try:
                response = await acompletion(
                    model=tier_cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    api_base=api_base,
                    timeout=settings.OLLAMA_REQUEST_TIMEOUT,
                )
                break
            except (
                litellm.exceptions.Timeout,
                httpx.TimeoutException,
                asyncio.TimeoutError,
                TimeoutError,
            ) as exc:
                timeout_s = (
                    int(settings.OLLAMA_REQUEST_TIMEOUT)
                    if float(settings.OLLAMA_REQUEST_TIMEOUT).is_integer()
                    else settings.OLLAMA_REQUEST_TIMEOUT
                )
                err_msg = f"timeout after {timeout_s}s"
                logger.error(
                    "Ollama request timed out; not retrying: %s",
                    err_msg,
                    extra={"tier": tier_cfg.name, "error": err_msg},
                )
                raise TimeoutError(err_msg) from exc
            except (
                litellm.exceptions.APIConnectionError,
                httpx.ConnectError,
                ConnectionError,
            ) as exc:
                if "timeout" in str(exc).lower() or "timed out" in str(exc).lower():
                    timeout_s = (
                        int(settings.OLLAMA_REQUEST_TIMEOUT)
                        if float(settings.OLLAMA_REQUEST_TIMEOUT).is_integer()
                        else settings.OLLAMA_REQUEST_TIMEOUT
                    )
                    err_msg = f"timeout after {timeout_s}s"
                    logger.error(
                        "Ollama request timed out; not retrying: %s",
                        err_msg,
                        extra={"tier": tier_cfg.name, "error": err_msg},
                    )
                    raise TimeoutError(err_msg) from exc

                last_error = exc
                if attempt < self.retry_count:
                    logger.warning(
                        "Ollama connection error; retrying once...",
                        extra={"attempt": attempt + 1, "tier": tier_cfg.name, "error": str(exc)},
                    )
                    await asyncio.sleep(0.2)
                    continue
                raise last_error

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract content & token metrics
        content = ""
        finish_reason = "stop"
        if response and hasattr(response, "choices") and response.choices:
            choice = response.choices[0]
            if hasattr(choice, "message") and hasattr(choice.message, "content"):
                content = choice.message.content or ""
            finish_reason = getattr(choice, "finish_reason", "stop") or "stop"
        elif isinstance(response, dict) and response.get("choices"):
            choice = response["choices"][0]
            if isinstance(choice, dict):
                msg = choice.get("message", {})
                if isinstance(msg, dict):
                    content = msg.get("content", "") or ""
                finish_reason = choice.get("finish_reason", "stop") or "stop"

        usage = getattr(response, "usage", None) if response else None
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        else:
            prompt_tokens = max(1, len(prompt) // 4)
            completion_tokens = max(1, len(content) // 4)

        # Real cost is always 0.0 for local Ollama
        real_cost = 0.0

        # Shadow cost derived from tier's shadow_price_per_1m input and output rates
        shadow_cost = (prompt_tokens * tier_cfg.input_price / 1_000_000.0) + (
            completion_tokens * tier_cfg.output_price / 1_000_000.0
        )
        shadow_cost = round(shadow_cost, 8)

        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=round(latency_ms, 2),
            real_cost=real_cost,
            shadow_cost=shadow_cost,
            model_name=tier_cfg.model,
            tier=tier_cfg.name,
            finish_reason=finish_reason,
        )


llm_client = LLMClient()
