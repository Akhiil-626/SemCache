from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.cache.backend import InMemoryExactCacheBackend
from app.cache.schemas import CacheHitResult
from app.db.models import RequestRecord
from app.db.session import get_db
from app.llm.client import LLMResponse
from app.main import app
from app.services.completion_service import CompletionResult, completion_service


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

    async def execute(self, stmt):
        class MockResult:
            def scalars(self):
                class MockScalars:
                    def all(self):
                        return []
                    def first(self):
                        return None
                return MockScalars()
        return MockResult()


@pytest.fixture
def mock_db_session():
    mock_db = MockDBSession()
    app.dependency_overrides[get_db] = lambda: mock_db
    yield mock_db
    app.dependency_overrides.pop(get_db, None)


@pytest.fixture(autouse=True)
def in_memory_cache_backend():
    backend = InMemoryExactCacheBackend()
    orig = completion_service.exact_backend
    completion_service.exact_backend = backend
    yield backend
    completion_service.exact_backend = orig


def _make_llm_resp(
    content: str,
    tier: str = "small",
    finish_reason: str = "stop",
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> LLMResponse:
    return LLMResponse(
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=80.0,
        real_cost=0.0,
        shadow_cost=0.0001,
        model_name=f"ollama_{tier}",
        tier=tier,
        finish_reason=finish_reason,
    )



@pytest.mark.asyncio
async def test_openai_chat_completions_endpoint(client: AsyncClient):
    """Verifies POST /v1/chat/completions returns OpenAI-compatible JSON structure."""
    mock_result = CompletionResult(
        content="OpenAI format response",
        cached=False,
        tier="small",
        model_name="ollama_chat/llama3.2:3b",
        similarity_score=0.0,
        cost_usd=0.0,
        shadow_cost_usd=0.00015,
        latency_ms=45.2,
        routing_rationale="Routed to baseline tier.",
        prompt_tokens=20,
        completion_tokens=30,
    )

    with patch("app.api.v1.chat.completion_service.execute_completion", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_result

        payload = {
            "messages": [
                {"role": "user", "content": "Hello, how does caching work?"}
            ],
            "model": "small",
        }

        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["object"] == "chat.completion"
        assert data["id"].startswith("chatcmpl-")
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert data["choices"][0]["message"]["content"] == "OpenAI format response"
        assert data["usage"]["prompt_tokens"] == 20
        assert data["usage"]["completion_tokens"] == 30
        assert data["usage"]["total_tokens"] == 50
        assert data["tier"] == "small"
        assert data["shadow_cost_usd"] == 0.00015


@pytest.mark.asyncio
async def test_query_endpoint_wrapper(client: AsyncClient):
    """Verifies that existing POST /api/v1/query works as a thin wrapper."""
    mock_result = CompletionResult(
        content="Legacy query response",
        cached=True,
        tier="small",
        model_name="cached",
        similarity_score=0.95,
        cost_usd=0.0,
        shadow_cost_usd=0.0,
        latency_ms=8.5,
        routing_rationale="Retrieved from cache.",
        prompt_tokens=10,
        completion_tokens=20,
    )

    with patch("app.api.v1.query.completion_service.execute_completion", new_callable=AsyncMock) as mock_exec:
        mock_exec.return_value = mock_result

        payload = {
            "prompt": "Test legacy query prompt"
        }

        response = await client.post("/api/v1/query", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["response"] == "Legacy query response"
        assert data["cached"] is True
        assert data["similarity_score"] == 0.95
        assert data["tier"] == "small"


@pytest.mark.asyncio
async def test_chat_model_auto_easy_prompt_routes_small_with_scoring_rationale(
    client: AsyncClient, mock_db_session
):
    """1. model 'auto' with an easy prompt: starts at small, rationale built from scores."""
    resp_small = _make_llm_resp("Paris is the capital of France.", tier="small")

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.lookup",
        new_callable=AsyncMock,
    ) as mock_lookup:
        mock_lookup.return_value = CacheHitResult(is_hit=False)
        mock_llm.return_value = resp_small

        payload = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}],
            "model": "auto",
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "small"
        assert "score" in data["routing_rationale"]
        assert "bypasses scoring" not in data["routing_rationale"]
        assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_chat_model_missing_routes_same_as_auto(
    client: AsyncClient, mock_db_session
):
    """2. model missing: same as auto (scoring + cascade, not a 422)."""
    resp_small = _make_llm_resp("Paris is the capital of France.", tier="small")

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.lookup",
        new_callable=AsyncMock,
    ) as mock_lookup:
        mock_lookup.return_value = CacheHitResult(is_hit=False)
        mock_llm.return_value = resp_small

        # Missing "model" entirely
        payload = {
            "messages": [{"role": "user", "content": "What is the capital of France?"}]
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "small"
        assert "score" in data["routing_rationale"]
        assert "bypasses scoring" not in data["routing_rationale"]


@pytest.mark.asyncio
async def test_chat_model_auto_hard_prompt_routes_large(
    client: AsyncClient, mock_db_session
):
    """3. model 'auto' with a hard prompt: starts at large."""
    resp_large = _make_llm_resp(
        "Detailed architectural comparison between Redis and PostgreSQL.",
        tier="large",
    )

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.lookup",
        new_callable=AsyncMock,
    ) as mock_lookup:
        mock_lookup.return_value = CacheHitResult(is_hit=False)
        mock_llm.return_value = resp_large

        payload = {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze and compare the architectural trade-offs between Redis and PostgreSQL.",
                }
            ],
            "model": "auto",
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "large"
        assert "score" in data["routing_rationale"]
        assert "bypasses scoring" not in data["routing_rationale"]
        assert mock_llm.call_count == 1


@pytest.mark.asyncio
async def test_chat_explicit_model_small_and_large_never_cascade(
    client: AsyncClient, mock_db_session
):
    """4. model 'small' and model 'large': explicit, rationale says explicit, never cascade."""
    resp_small = _make_llm_resp("Explicit small output", tier="small")
    resp_large = _make_llm_resp("Explicit large output", tier="large")

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.lookup",
        new_callable=AsyncMock,
    ) as mock_lookup:
        mock_lookup.return_value = CacheHitResult(is_hit=False)

        # 4a. Explicit small on hard query
        mock_llm.return_value = resp_small
        payload_small = {
            "messages": [
                {
                    "role": "user",
                    "content": "Analyze and architect a distributed system with proof.",
                }
            ],
            "model": "small",
        }
        res_small = await client.post("/v1/chat/completions", json=payload_small)
        assert res_small.status_code == 200
        data_small = res_small.json()
        assert data_small["tier"] == "small"
        assert "Explicit model 'small' bypasses scoring." in data_small["routing_rationale"]
        assert data_small["escalation_count"] == 0

        # 4b. Explicit large on easy query
        mock_llm.return_value = resp_large
        payload_large = {
            "messages": [{"role": "user", "content": "Hello world"}],
            "model": "large",
        }
        res_large = await client.post("/v1/chat/completions", json=payload_large)
        assert res_large.status_code == 200
        data_large = res_large.json()
        assert data_large["tier"] == "large"
        assert "Explicit model 'large' bypasses scoring." in data_large["routing_rationale"]
        assert data_large["escalation_count"] == 0

        # 4c. Never cascade even when validation fails on explicit tier
        resp_truncated = _make_llm_resp("Truncated", tier="small", finish_reason="length")
        mock_llm.reset_mock()
        mock_llm.return_value = resp_truncated
        res_no_cascade = await client.post("/v1/chat/completions", json=payload_small)
        assert res_no_cascade.status_code == 200
        assert mock_llm.call_count == 1
        assert res_no_cascade.json()["escalation_count"] == 0


@pytest.mark.asyncio
async def test_chat_model_auto_small_fails_validator_escalates_to_large(
    client: AsyncClient, mock_db_session
):
    """5. model 'auto' where small tier fails validator: second attempt on large; response and logged row show escalation_count 1."""
    resp_small_fail = _make_llm_resp("Cut off text", tier="small", finish_reason="length")
    resp_large_ok = _make_llm_resp("Complete response", tier="large", finish_reason="stop")

    with patch(
        "app.services.completion_service.llm_client.generate_response",
        new_callable=AsyncMock,
    ) as mock_llm, patch(
        "app.services.completion_service.cache_service.lookup",
        new_callable=AsyncMock,
    ) as mock_lookup:
        mock_lookup.return_value = CacheHitResult(is_hit=False)
        mock_llm.side_effect = [resp_small_fail, resp_large_ok]

        payload = {
            "messages": [{"role": "user", "content": "Explain how DNS resolution works."}],
            "model": "auto",
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        data = response.json()
        assert data["tier"] == "large"
        assert data["escalation_count"] == 1
        assert mock_llm.call_count == 2

        # Check logged record in database
        logged_records = [r for r in mock_db_session.added if isinstance(r, RequestRecord)]
        assert len(logged_records) > 0
        last_record = logged_records[-1]
        assert last_record.escalation_count == 1
        assert last_record.tier == "large"
        assert len(last_record.attempts) == 2


@pytest.mark.asyncio
async def test_chat_unknown_model_returns_400_listing_valid_values(client: AsyncClient):
    """6. unknown model name: HTTP 400 listing the valid values ('auto' plus tier names)."""
    payload = {
        "messages": [{"role": "user", "content": "Hello"}],
        "model": "unknown-gpt-model",
    }
    response = await client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 400

    body = response.text.lower()
    assert "auto" in body
    assert "small" in body
    assert "large" in body

