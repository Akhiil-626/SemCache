from unittest.mock import AsyncMock, patch
import litellm.exceptions
import pytest
from app.core.config import settings
from app.llm.client import LLMClient


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockMessage:
    def __init__(self, content: str):
        self.content = content


class MockChoice:
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockLiteLLMResponse:
    def __init__(self, content: str, prompt_tokens: int, completion_tokens: int):
        self.choices = [MockChoice(content)]
        self.usage = MockUsage(prompt_tokens, completion_tokens)


@pytest.mark.asyncio
async def test_llm_client_generation_and_cost_calculation():
    """Verifies LiteLLM call invocation, real cost (0), and shadow cost calculation."""
    client = LLMClient()
    mock_resp = MockLiteLLMResponse(
        content="Response from small model",
        prompt_tokens=400,
        completion_tokens=600,
    )

    with patch("app.llm.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        result = await client.generate_response(prompt="Hello Ollama", tier="small")

        assert result.content == "Response from small model"
        assert result.prompt_tokens == 400
        assert result.completion_tokens == 600
        assert result.real_cost == 0.0

        # Small tier shadow price: input $0.15 / 1M, output $0.60 / 1M
        # 400 * 0.15 / 1e6 + 600 * 0.60 / 1e6 = 0.00042
        expected_shadow = round((400 * 0.15 / 1_000_000.0) + (600 * 0.60 / 1_000_000.0), 8)
        assert result.shadow_cost == expected_shadow
        assert result.tier == "small"
        assert result.model_name == "ollama_chat/llama3.2:3b"

        # Verify acompletion called with ollama_chat prefix and api_base
        mock_acompletion.assert_called_once()
        _, kwargs = mock_acompletion.call_args
        assert kwargs["model"] == "ollama_chat/llama3.2:3b"
        assert kwargs["api_base"] == settings.OLLAMA_BASE_URL


@pytest.mark.asyncio
async def test_llm_client_large_tier_shadow_cost():
    """Verifies shadow cost calculation for large tier."""
    client = LLMClient()
    mock_resp = MockLiteLLMResponse(
        content="Reasoning response from large model",
        prompt_tokens=5000,
        completion_tokens=5000,
    )

    with patch("app.llm.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        result = await client.generate_response(prompt="Explain calculus", tier="large")

        assert result.real_cost == 0.0
        # Large tier: input $2.00 / 1M, output $8.00 / 1M
        # 5000 * 2.0 / 1e6 + 5000 * 8.0 / 1e6 = 0.05
        expected_shadow = round((5000 * 2.00 / 1_000_000.0) + (5000 * 8.00 / 1_000_000.0), 8)
        assert result.shadow_cost == expected_shadow
        assert result.tier == "large"


@pytest.mark.asyncio
async def test_shadow_cost_input_output_calculation():
    """TASK 3 unit test: 35 prompt + 38 completion tokens on small (0.15 / 0.60) must equal 0.00002805."""
    client = LLMClient()
    mock_resp = MockLiteLLMResponse(
        content="Shadow cost calculation test",
        prompt_tokens=35,
        completion_tokens=38,
    )

    with patch("app.llm.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = mock_resp

        result = await client.generate_response(prompt="Test prompt", tier="small")

        # 35 * 0.15 / 1e6 = 0.00000525
        # 38 * 0.60 / 1e6 = 0.0000228
        # total = 0.00002805
        assert result.prompt_tokens == 35
        assert result.completion_tokens == 38
        assert result.shadow_cost == 0.00002805


@pytest.mark.asyncio
async def test_llm_client_retry_on_connection_error():
    """Verifies client retries once on connection error before succeeding."""
    client = LLMClient(retry_count=1)
    mock_success = MockLiteLLMResponse("Recovered response", 50, 50)
    connection_err = litellm.exceptions.APIConnectionError(
        message="Failed to connect to host",
        model="ollama_chat/llama3.2:3b",
        llm_provider="ollama",
    )

    with patch("app.llm.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        # First call fails, second call succeeds
        mock_acompletion.side_effect = [connection_err, mock_success]

        result = await client.generate_response(prompt="Test retry", tier="small")

        assert result.content == "Recovered response"
        assert mock_acompletion.call_count == 2


@pytest.mark.asyncio
async def test_llm_client_timeout_does_not_retry():
    """Verifies client never retries on read timeout and raises TimeoutError with 'timeout after <N>s'."""
    client = LLMClient(retry_count=1)
    timeout_err = litellm.exceptions.Timeout(
        message="Request timed out",
        model="ollama_chat/llama3.2:3b",
        llm_provider="ollama",
    )

    with patch("app.llm.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.side_effect = timeout_err

        with pytest.raises(TimeoutError) as exc_info:
            await client.generate_response(prompt="Heavy prompt", tier="small")

        assert "timeout after" in str(exc_info.value)
        assert mock_acompletion.call_count == 1

