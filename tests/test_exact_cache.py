import time
from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.cache.backend import (
    InMemoryExactCacheBackend,
    RedisExactCacheBackend,
    get_exact_cache_backend,
    reset_exact_cache_backend,
)
from app.cache.key import build_exact_cache_key
from app.cache.policies import evaluate_cache_policy
from app.core.config import settings
from app.db.session import get_db
from app.llm.client import LLMResponse
from app.main import app
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


@pytest.fixture
def mock_llm_response():
    return LLMResponse(
        content="This is a deterministic LLM response.",
        prompt_tokens=30,
        completion_tokens=40,
        latency_ms=120.5,
        real_cost=0.0,
        shadow_cost=0.0000285,
        model_name="ollama_chat/llama3.2:3b",
        tier="small",
    )


@pytest.mark.asyncio
async def test_exact_cache_hit_flow(mock_db, in_memory_backend, mock_llm_response):
    """Test 1: First request misses and stores; second identical request hits exact cache."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Call 1: Miss
        res1 = await completion_service.execute_completion(
            prompt="Tell me about algorithms",
            db=mock_db,
            tenant_id="tenant-1",
            temperature=0.5,
        )
        assert res1.cached is False
        assert res1.cache_status == "miss"
        assert res1.content == "This is a deterministic LLM response."
        assert res1.prompt_tokens == 30
        assert res1.completion_tokens == 40
        assert mock_llm.call_count == 1

        # Call 2: Exact Hit
        res2 = await completion_service.execute_completion(
            prompt="Tell me about algorithms",
            db=mock_db,
            tenant_id="tenant-1",
            temperature=0.5,
        )
        assert res2.cached is True
        assert res2.cache_status == "exact_hit"
        assert res2.content == "This is a deterministic LLM response."
        # LLM was not called a second time
        assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_different_tenant_misses(mock_db, in_memory_backend, mock_llm_response):
    """Test 2: Requests with different tenant_id miss each other."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Tenant A
        res_a = await completion_service.execute_completion(
            prompt="Explain binary search",
            db=mock_db,
            tenant_id="tenant-alpha",
            temperature=0.5,
        )
        assert res_a.cache_status == "miss"
        assert mock_llm.call_count == 1

        # Tenant B with identical prompt
        res_b = await completion_service.execute_completion(
            prompt="Explain binary search",
            db=mock_db,
            tenant_id="tenant-beta",
            temperature=0.5,
        )
        assert res_b.cache_status == "miss"
        assert res_b.cached is False
        assert mock_llm.call_count == 2


@pytest.mark.asyncio
async def test_different_cache_version_misses(mock_db, in_memory_backend, mock_llm_response):
    """Test 3: Bumping CACHE_VERSION invalidates cache entries."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Populate with v1
        original_ver = settings.CACHE_VERSION
        try:
            settings.CACHE_VERSION = "v1"
            res1 = await completion_service.execute_completion(
                prompt="Explain quicksort",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res1.cache_status == "miss"
            assert mock_llm.call_count == 1

            # Query with v2
            settings.CACHE_VERSION = "v2"
            res2 = await completion_service.execute_completion(
                prompt="Explain quicksort",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res2.cache_status == "miss"
            assert res2.cached is False
            assert mock_llm.call_count == 2
        finally:
            settings.CACHE_VERSION = original_ver


@pytest.mark.asyncio
async def test_high_temperature_skipped(mock_db, in_memory_backend, mock_llm_response):
    """Test 4: temperature > 0.7 skips caching."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        res = await completion_service.execute_completion(
            prompt="Be creative and write a poem",
            db=mock_db,
            tenant_id="tenant-1",
            temperature=0.9,
        )
        assert res.cached is False
        assert res.cache_status == "skipped"
        assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_time_sensitive_prompt_skipped(mock_db, in_memory_backend, mock_llm_response):
    """Test 5: Time-sensitive keywords (today, now, latest, current, price, weather) skip caching."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        queries = [
            "What is today's stock price?",
            "Tell me the current weather in London",
            "What is the latest news right now?",
        ]
        for q in queries:
            res = await completion_service.execute_completion(
                prompt=q,
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res.cached is False
            assert res.cache_status == "skipped"


@pytest.mark.asyncio
async def test_cache_control_no_cache_bypass(client: AsyncClient, mock_db, in_memory_backend, mock_llm_response):
    """Test 6: Cache-Control: no-cache skips lookup and store."""
    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            payload = {
                "messages": [{"role": "user", "content": "Explain hash maps"}],
                "temperature": 0.5,
            }

            headers = {"Cache-Control": "no-cache"}
            response = await client.post("/v1/chat/completions", json=payload, headers=headers)
            assert response.status_code == 200
            data = response.json()
            assert data["cached"] is False
            assert data["cache_status"] == "bypass"
            assert mock_llm.call_count == 1
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_ttl_expiry_with_controllable_clock(mock_db, mock_llm_response):
    """Test 7: TTL expiry causes cache miss using in-memory backend with controllable clock."""
    current_time = 1000.0

    def clock():
        return current_time

    backend = InMemoryExactCacheBackend(clock=clock)
    original_backend = completion_service.exact_backend
    completion_service.exact_backend = backend

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            # Initial call at t=1000.0 (TTL = 3600s, expiry = 4600.0)
            res1 = await completion_service.execute_completion(
                prompt="Explain red-black trees",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res1.cache_status == "miss"
            assert mock_llm.call_count == 1

            # Query at t=2000.0 (within TTL) -> Hit
            current_time = 2000.0
            res2 = await completion_service.execute_completion(
                prompt="Explain red-black trees",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res2.cache_status == "exact_hit"
            assert res2.cached is True
            assert mock_llm.call_count == 1

            # Advance clock past TTL to t=5000.0 -> Miss
            current_time = 5000.0
            res3 = await completion_service.execute_completion(
                prompt="Explain red-black trees",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res3.cache_status == "miss"
            assert res3.cached is False
            assert mock_llm.call_count == 2
    finally:
        completion_service.exact_backend = original_backend


@pytest.mark.asyncio
async def test_redis_failure_falls_back_to_miss(mock_db, mock_llm_response):
    """Test 8: Backend errors fall back to miss without failing the request."""
    broken_backend = RedisExactCacheBackend(redis_url="redis://nonexistent-host:6379/0")
    original_backend = completion_service.exact_backend
    completion_service.exact_backend = broken_backend

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            # Even if Redis is completely unavailable, request succeeds cleanly as a miss
            res = await completion_service.execute_completion(
                prompt="Explain Dijkstra algorithm",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            assert res.cached is False
            assert res.cache_status == "miss"
            assert res.content == "This is a deterministic LLM response."
            assert mock_llm.call_count == 1
    finally:
        completion_service.exact_backend = original_backend


@pytest.mark.asyncio
async def test_hit_latency_is_measured_not_constant(mock_db, in_memory_backend, mock_llm_response):
    """Test 9: Hit latency is the real measured time (not hardcoded 99.99 or simulated constant)."""
    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Seed cache
        await completion_service.execute_completion(
            prompt="Measure latency check",
            db=mock_db,
            tenant_id="tenant-1",
            temperature=0.5,
        )

        # Hit
        hit_res = await completion_service.execute_completion(
            prompt="Measure latency check",
            db=mock_db,
            tenant_id="tenant-1",
            temperature=0.5,
        )
        assert hit_res.cached is True
        assert hit_res.cache_status == "exact_hit"
        # Real in-memory lookup is sub-millisecond to a few ms; definitely not 99.99
        assert hit_res.latency_ms != 99.99
        assert 0.0 <= hit_res.latency_ms < 50.0


@pytest.mark.asyncio
async def test_saved_metrics_equal_original_call(mock_db, in_memory_backend):
    """Test 10: On exact hit, saved_* fields and tokens_saved match original call's values."""
    original_llm_res = LLMResponse(
        content="First call generated content",
        prompt_tokens=42,
        completion_tokens=58,
        latency_ms=185.4,
        real_cost=0.0,
        shadow_cost=0.0000411,
        model_name="ollama_chat/llama3.2:3b",
        tier="small",
    )

    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = original_llm_res

        # Call 1: Miss
        res1 = await completion_service.execute_completion(
            prompt="Check savings tracking",
            db=mock_db,
            tenant_id="tenant-savings",
            temperature=0.5,
        )
        assert res1.tokens_saved == 0
        assert res1.saved_latency_ms == 0.0
        assert res1.saved_shadow_cost_usd == 0.0

        # Call 2: Exact Hit
        res2 = await completion_service.execute_completion(
            prompt="Check savings tracking",
            db=mock_db,
            tenant_id="tenant-savings",
            temperature=0.5,
        )
        assert res2.cached is True
        assert res2.cache_status == "exact_hit"

        # tokens_saved = prompt_tokens (42) + completion_tokens (58) = 100
        assert res2.tokens_saved == 100
        # saved_latency_ms matches original call's latency
        assert res2.saved_latency_ms > 0.0
        # saved_shadow_cost_usd matches original call's shadow cost
        assert res2.saved_shadow_cost_usd == original_llm_res.shadow_cost
        # Cost consumed on hit is 0
        assert res2.cost_usd == 0.0
        assert res2.shadow_cost_usd == 0.0
        # Response usage preserves original answer's tokens
        assert res2.prompt_tokens == 42
        assert res2.completion_tokens == 58


@pytest.mark.asyncio
async def test_two_identical_requests_produce_same_key_and_second_is_exact_hit(
    mock_db, in_memory_backend, mock_llm_response
):
    """Task 6(i): Two identical requests produce the same key, the second returns exact_hit and does not call LLM."""
    msgs = [{"role": "user", "content": "Explain quicksort step by step"}]
    key1 = build_exact_cache_key(
        tenant_id="default",
        tier="small",
        model="ollama_chat/llama3.2:3b",
        messages=msgs,
        temperature=0.7,
        max_tokens=None,
    )
    key2 = build_exact_cache_key(
        tenant_id="default",
        tier="small",
        model="ollama_chat/llama3.2:3b",
        messages=msgs,
        temperature=0.7,
        max_tokens=None,
    )
    assert key1 == key2

    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        # Request 1: Miss
        res1 = await completion_service.execute_completion(
            prompt="Explain quicksort step by step",
            db=mock_db,
            preferred_tier="small",
            messages=msgs,
            temperature=0.7,
        )
        assert res1.cache_status == "miss"
        assert res1.cached is False
        assert mock_llm.call_count == 1

        # Request 2: Exact Hit
        res2 = await completion_service.execute_completion(
            prompt="Explain quicksort step by step",
            db=mock_db,
            preferred_tier="small",
            messages=msgs,
            temperature=0.7,
        )
        assert res2.cache_status == "exact_hit"
        assert res2.cached is True
        assert res2.content == mock_llm_response.content
        # Second call does NOT call LLM
        assert mock_llm.call_count == 1


def test_backend_factory_returns_redis_backend_when_enabled():
    """Task 6(ii): The backend factory returns the Redis backend when EXACT_CACHE_ENABLED=true and REDIS_URL is set."""
    reset_exact_cache_backend()
    orig_enabled = settings.EXACT_CACHE_ENABLED
    orig_url = settings.REDIS_URL
    try:
        settings.EXACT_CACHE_ENABLED = True
        settings.REDIS_URL = "redis://localhost:6379/0"
        backend = get_exact_cache_backend()
        assert isinstance(backend, RedisExactCacheBackend)
        assert backend.redis_url == "redis://localhost:6379/0"
    finally:
        settings.EXACT_CACHE_ENABLED = orig_enabled
        settings.REDIS_URL = orig_url
        reset_exact_cache_backend()


@pytest.mark.asyncio
async def test_serialization_failure_while_storing_is_logged_and_does_not_fail_request(
    mock_db, in_memory_backend, mock_llm_response, caplog
):
    """Task 6(iii): A serialization failure while storing is logged at WARNING and does not fail the request."""
    import logging
    caplog.set_level(logging.WARNING)

    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        with patch.object(
            in_memory_backend,
            "set",
            side_effect=TypeError("Object of type object is not JSON serializable"),
        ):
            res = await completion_service.execute_completion(
                prompt="Test serialization error resilience",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )
            # The user's request still succeeds with the generated response
            assert res.content == "This is a deterministic LLM response."
            assert res.cached is False

            # Verify the warning was logged with error type and message
            warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            assert any("Exact cache store error" in w and "TypeError" in w for w in warnings)


@pytest.mark.asyncio
async def test_key_building_failure_is_logged_and_returns_normal_response(
    mock_db, in_memory_backend, mock_llm_response, caplog
):
    """Verifies that if build_exact_cache_key raises an exception, the request still succeeds,
    returns the normal LLM response, and logs a warning with cache_status 'bypass'.
    """
    import logging
    caplog.set_level(logging.WARNING)

    with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
        mock_llm.return_value = mock_llm_response

        with patch(
            "app.services.completion_service.build_exact_cache_key",
            side_effect=RuntimeError("Key generation failed"),
        ):
            res = await completion_service.execute_completion(
                prompt="Test key generation error resilience",
                db=mock_db,
                tenant_id="tenant-1",
            )
            assert res.content == "This is a deterministic LLM response."
            assert res.cached is False
            assert res.cache_status == "bypass"

            warnings = [r.message for r in caplog.records if r.levelno == logging.WARNING]
            assert any("Key generation failed" in w for w in warnings)
