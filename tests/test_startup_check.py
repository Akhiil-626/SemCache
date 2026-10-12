import httpx
import pytest
from app.core.config import ModelTierConfig
from app.llm.startup_check import verify_ollama_models


@pytest.mark.asyncio
async def test_startup_check_unreachable_ollama():
    """Verifies that an unreachable Ollama instance fails with clear error."""
    test_tiers = [
        ModelTierConfig(name="small", model="ollama_chat/llama3.2:3b"),
        ModelTierConfig(name="large", model="ollama_chat/llama3.1:8b"),
    ]

    # Handler that simulates connection failure
    def connection_failure_handler(request: httpx.Request):
        raise httpx.ConnectError("Connection refused to host:11434")

    transport = httpx.MockTransport(connection_failure_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        with pytest.raises(RuntimeError) as exc_info:
            await verify_ollama_models(
                ollama_base_url="http://localhost:11434",
                tiers=test_tiers,
                client=mock_client,
            )

        err_msg = str(exc_info.value)
        assert "Ollama server is unreachable at 'http://localhost:11434'" in err_msg
        assert "ollama serve" in err_msg


@pytest.mark.asyncio
async def test_startup_check_missing_model_fail_with_pull_command():
    """Verifies that a missing model fails with exact 'ollama pull <model>' command."""
    test_tiers = [
        ModelTierConfig(name="small", model="ollama_chat/llama3.2:3b"),
        ModelTierConfig(name="large", model="ollama_chat/llama3.1:8b"),
    ]

    # Ollama returns only llama3.2:3b; llama3.1:8b is missing
    def mock_tags_handler(request: httpx.Request):
        return httpx.Response(
            status_code=200,
            json={"models": [{"name": "llama3.2:3b", "model": "llama3.2:3b"}]},
        )

    transport = httpx.MockTransport(mock_tags_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        with pytest.raises(RuntimeError) as exc_info:
            await verify_ollama_models(
                ollama_base_url="http://localhost:11434",
                tiers=test_tiers,
                client=mock_client,
            )

        err_msg = str(exc_info.value)
        assert "Missing required models" in err_msg
        assert "ollama pull llama3.1:8b" in err_msg


@pytest.mark.asyncio
async def test_startup_check_all_models_present():
    """Verifies that startup succeeds when all required models are pulled."""
    test_tiers = [
        ModelTierConfig(name="small", model="ollama_chat/llama3.2:3b"),
        ModelTierConfig(name="large", model="ollama_chat/llama3.1:8b"),
    ]

    def mock_tags_handler(request: httpx.Request):
        return httpx.Response(
            status_code=200,
            json={
                "models": [
                    {"name": "llama3.2:3b"},
                    {"name": "llama3.1:8b"},
                ]
            },
        )

    transport = httpx.MockTransport(mock_tags_handler)
    async with httpx.AsyncClient(transport=transport) as mock_client:
        # Should not raise any exception
        await verify_ollama_models(
            ollama_base_url="http://localhost:11434",
            tiers=test_tiers,
            client=mock_client,
        )
