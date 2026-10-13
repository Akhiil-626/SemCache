from unittest.mock import AsyncMock, patch
import pytest

from app.cache.backend import InMemoryExactCacheBackend
from app.llm.client import LLMResponse
from app.routing.cascade import CascadeResult, run_cascade
from app.services.completion_service import completion_service


class MockDBSession:
    def __init__(self):
        self.added = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, obj):
        pass


@pytest.fixture
def mock_db():
    return MockDBSession()


@pytest.fixture
def in_memory_backend():
    backend = InMemoryExactCacheBackend()
    original_backend = completion_service.exact_backend
    completion_service.exact_backend = backend
    yield backend
    completion_service.exact_backend = original_backend


def _make_resp(
    content: str,
    tier: str = "small",
    finish_reason: str = "stop",
    prompt_tokens: int = 20,
    completion_tokens: int = 30,
    latency_ms: float = 100.0,
    shadow_cost: float = 0.0001,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        real_cost=0.0,
        shadow_cost=shadow_cost,
        model_name=f"ollama_{tier}",
        tier=tier,
        finish_reason=finish_reason,
    )


@pytest.mark.asyncio
async def test_cascade_scenario_a_small_passes_validation():
    """Scenario (a): small passes validation -> 1 call, 0 escalations, validation_failed=false."""
    mock_llm = AsyncMock()
    mock_llm.generate_response.return_value = _make_resp(
        content="Photosynthesis is the process by which green plants make food.",
        tier="small",
        finish_reason="stop",
        latency_ms=120.0,
        shadow_cost=0.00015,
    )

    result = await run_cascade(
        prompt="What is photosynthesis?",
        starting_tier="small",
        llm_client=mock_llm,
    )

    assert mock_llm.generate_response.call_count == 1
    assert result.escalation_count == 0
    assert result.validation_failed is False
    assert result.tier == "small"
    assert len(result.attempts) == 1
    assert result.attempts[0]["validation_passed"] is True
    assert result.wasted_latency_ms == 0.0
    assert result.wasted_shadow_cost_usd == 0.0
    assert result.total_latency_ms == 120.0
    assert result.total_shadow_cost == 0.00015


@pytest.mark.asyncio
async def test_cascade_scenario_b_small_fails_large_passes():
    """Scenario (b): small fails (truncated), large passes -> 2 calls, 1 escalation, totals sum both calls, wasted reflects small."""
    resp_small_truncated = _make_resp(
        content="Incomplete sentence that cut off",
        tier="small",
        finish_reason="length",
        prompt_tokens=15,
        completion_tokens=25,
        latency_ms=110.0,
        shadow_cost=0.0002,
    )
    resp_large_passed = _make_resp(
        content="Complete and well-formed technical explanation of the architecture.",
        tier="large",
        finish_reason="stop",
        prompt_tokens=15,
        completion_tokens=60,
        latency_ms=350.0,
        shadow_cost=0.0012,
    )

    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = [resp_small_truncated, resp_large_passed]

    result = await run_cascade(
        prompt="Explain distributed consistency.",
        starting_tier="small",
        llm_client=mock_llm,
    )

    assert mock_llm.generate_response.call_count == 2
    assert result.escalation_count == 1
    assert result.validation_failed is False
    assert result.tier == "large"
    assert len(result.attempts) == 2

    # Attempt 1 (small) failed validation
    assert result.attempts[0]["tier"] == "small"
    assert result.attempts[0]["validation_passed"] is False
    assert "truncated" in result.attempts[0]["failure_reasons"]

    # Attempt 2 (large) passed validation
    assert result.attempts[1]["tier"] == "large"
    assert result.attempts[1]["validation_passed"] is True

    # Total metrics sum both calls
    assert result.total_latency_ms == 460.0  # 110.0 + 350.0
    assert result.total_completion_tokens == 85  # 25 + 60
    assert result.total_shadow_cost == 0.0014  # 0.0002 + 0.0012

    # Wasted metrics reflect small's failed attempt
    assert result.wasted_latency_ms == 110.0
    assert result.wasted_shadow_cost_usd == 0.0002


@pytest.mark.asyncio
async def test_cascade_scenario_c_all_tiers_fail_returns_highest():
    """Scenario (c): all tiers fail -> stops at large, returns large's answer, validation_failed=true."""
    resp_small_fail = _make_resp(
        content="Cut off 1",
        tier="small",
        finish_reason="length",
        latency_ms=100.0,
        shadow_cost=0.0002,
    )
    resp_large_fail = _make_resp(
        content="Cut off 2",
        tier="large",
        finish_reason="length",
        latency_ms=300.0,
        shadow_cost=0.0010,
    )

    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = [resp_small_fail, resp_large_fail]

    result = await run_cascade(
        prompt="Design a fault-tolerant database.",
        starting_tier="small",
        llm_client=mock_llm,
    )

    assert mock_llm.generate_response.call_count == 2
    assert result.escalation_count == 1
    assert result.validation_failed is True
    assert result.tier == "large"
    assert result.response.content == "Cut off 2"
    assert len(result.attempts) == 2


@pytest.mark.asyncio
async def test_cascade_scenario_d_timeout_fails_immediately_no_escalation():
    """Scenario (d): timeout on small -> fails immediately, no escalation to large."""
    mock_llm = AsyncMock()
    mock_llm.generate_response.side_effect = TimeoutError("timeout after 30s")

    with pytest.raises(TimeoutError, match="timeout after 30s"):
        await run_cascade(
            prompt="Complex query that hangs",
            starting_tier="small",
            llm_client=mock_llm,
        )

    # Must NOT call large tier on timeout
    assert mock_llm.generate_response.call_count == 1


@pytest.mark.asyncio
async def test_cascade_scenario_e_explicit_model_skips_cascade():
    """Scenario (e): explicit model requested -> no cascade, single call even if invalid."""
    resp_truncated = _make_resp(
        content="Truncated output",
        tier="small",
        finish_reason="length",
    )
    mock_llm = AsyncMock()
    mock_llm.generate_response.return_value = resp_truncated

    result = await run_cascade(
        prompt="My query",
        starting_tier="small",
        llm_client=mock_llm,
        bypass_cascade=True,
    )

    assert mock_llm.generate_response.call_count == 1
    assert result.escalation_count == 0
    assert result.validation_failed is True
    assert len(result.attempts) == 1


@pytest.mark.asyncio
async def test_cascade_scenario_f_invalid_answer_never_written_to_cache(
    mock_db, in_memory_backend
):
    """Scenario (f): invalid answer is never written to exact or semantic cache."""
    # Both tiers return truncated answers (validation fails)
    resp_small_fail = _make_resp("Short fail", tier="small", finish_reason="length")
    resp_large_fail = _make_resp("Large fail", tier="large", finish_reason="length")

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.store",
        new_callable=AsyncMock,
    ) as mock_sem_store:
        mock_llm.side_effect = [resp_small_fail, resp_large_fail]

        res = await completion_service.execute_completion(
            prompt="What is an invalid response?",
            db=mock_db,
        )

        assert res.validation_failed is True
        # Exact cache must not have been populated
        assert in_memory_backend.store == {}
        # Semantic cache store must never be called
        mock_sem_store.assert_not_called()


@pytest.mark.asyncio
async def test_cascade_scenario_g_successful_escalation_cached_with_cumulative_metrics(
    mock_db, in_memory_backend
):
    """Scenario (g): successful escalation result is cached with cumulative metrics under winning tier."""
    resp_small_fail = _make_resp(
        "Cut off",
        tier="small",
        finish_reason="length",
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=100.0,
        shadow_cost=0.0001,
    )
    resp_large_ok = _make_resp(
        "Complete and thorough answer for an informational question.",
        tier="large",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=50,
        latency_ms=300.0,
        shadow_cost=0.0010,
    )

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.store",
        new_callable=AsyncMock,
    ) as mock_sem_store, patch(
        "app.services.completion_service.embedding_service.get_embedding",
        new_callable=AsyncMock,
    ) as mock_embed:
        mock_llm.side_effect = [resp_small_fail, resp_large_ok]
        mock_embed.return_value = [0.1] * 384

        res = await completion_service.execute_completion(
            prompt="What is the difference between TCP and UDP?",
            db=mock_db,
        )

        assert res.validation_failed is False
        assert res.tier == "large"
        assert res.escalation_count == 1

        # Check exact cache store payload
        assert len(in_memory_backend.store) == 1
        cached_entry = list(in_memory_backend.store.values())[0]
        assert cached_entry["tier"] == "large"
        assert cached_entry["completion_tokens"] == 70  # 20 + 50
        assert cached_entry["latency_ms"] == 400.0  # 100.0 + 300.0
        assert cached_entry["shadow_cost_usd"] == 0.0011  # 0.0001 + 0.0010

        # Check semantic cache store was called with cumulative metrics
        mock_sem_store.assert_called_once()
        _, sem_kwargs = mock_sem_store.call_args
        assert sem_kwargs["tier"] == "large"
        assert sem_kwargs["completion_tokens"] == 70
        assert sem_kwargs["latency_ms"] == 400.0
        assert sem_kwargs["shadow_cost_usd"] == 0.0011
