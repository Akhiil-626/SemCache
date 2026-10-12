from unittest.mock import AsyncMock, patch
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_liveness_endpoint(client: AsyncClient) -> None:
    """Verifies that /health is a pure liveness check returning HTTP 200 and {'status': 'ok'}."""
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.asyncio
async def test_readiness_endpoint_unready_offline(client: AsyncClient) -> None:
    """Verifies that /health/ready checks dependencies and returns HTTP 503 when down."""
    with patch("app.api.v1.health.check_db_health", new_callable=AsyncMock) as mock_db_health:
        mock_db_health.return_value = (False, "Connection refused: database down")

        response = await client.get("/health/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["status"] == "unready"
        assert "dependencies" in payload
        assert "database" in payload["dependencies"]
        assert payload["dependencies"]["database"]["status"] == "down"
        assert "ollama" in payload["dependencies"]
        assert "redis" in payload["dependencies"]


@pytest.mark.asyncio
async def test_readiness_endpoint_ready(client: AsyncClient) -> None:
    """Verifies that /health/ready returns HTTP 200 when all dependencies are up."""
    with (
        patch("app.api.v1.health.check_db_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.v1.health.check_ollama_health", new_callable=AsyncMock) as mock_ollama,
        patch("app.api.v1.health.check_redis_health", new_callable=AsyncMock) as mock_redis,
    ):
        mock_db.return_value = (True, None)
        mock_ollama.return_value = (True, None)
        mock_redis.return_value = (True, None)

        response = await client.get("/health/ready")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ready"
        assert payload["dependencies"]["database"]["status"] == "up"
        assert payload["dependencies"]["ollama"]["status"] == "up"
        assert payload["dependencies"]["redis"]["status"] == "ok"


@pytest.mark.asyncio
async def test_readiness_endpoint_redis_offline(client: AsyncClient) -> None:
    """Verifies that /health/ready returns HTTP 503 when Redis is down."""
    with (
        patch("app.api.v1.health.check_db_health", new_callable=AsyncMock) as mock_db,
        patch("app.api.v1.health.check_ollama_health", new_callable=AsyncMock) as mock_ollama,
        patch("app.api.v1.health.check_redis_health", new_callable=AsyncMock) as mock_redis,
    ):
        mock_db.return_value = (True, None)
        mock_ollama.return_value = (True, None)
        mock_redis.return_value = (False, "Connection refused: Redis offline")

        response = await client.get("/health/ready")
        assert response.status_code == 503
        payload = response.json()
        assert payload["status"] == "unready"
        assert "Connection refused" in payload["dependencies"]["redis"]["status"]
        assert payload["dependencies"]["redis"]["error"] == "Connection refused: Redis offline"
