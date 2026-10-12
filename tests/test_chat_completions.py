from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient

from app.services.completion_service import CompletionResult


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
