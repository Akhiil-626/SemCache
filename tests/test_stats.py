import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock
import pytest
from httpx import AsyncClient

from app.db.models import RequestRecord
from app.db.session import get_db
from app.main import app


class MockScalarResult:
    def __init__(self, items):
        self._items = items

    def all(self):
        return self._items


class MockExecuteResult:
    def __init__(self, items):
        self._items = items

    def scalars(self):
        return MockScalarResult(self._items)


@pytest.mark.asyncio
async def test_stats_empty_table(client: AsyncClient):
    """Verifies that GET /v1/stats returns zeros and HTTP 200 when table is empty."""
    mock_db = AsyncMock()
    mock_db.execute.return_value = MockExecuteResult([])

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = await client.get("/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 0
        assert data["hit_rate"] == 0.0
        assert data["total_tokens_saved"] == 0
        assert data["total_saved_latency_ms"] == 0.0
        assert data["total_saved_shadow_cost_usd"] == 0.0
        assert "small" in data["tiers"]
        assert "large" in data["tiers"]

        small_stats = data["tiers"]["small"]
        assert small_stats["request_count"] == 0
        assert small_stats["avg_latency_ms"] == 0.0
        assert small_stats["p95_latency_ms"] == 0.0
        assert small_stats["total_prompt_tokens"] == 0
        assert small_stats["total_completion_tokens"] == 0
        assert small_stats["total_shadow_cost_usd"] == 0.0
        assert small_stats["hit_rate"] == 0.0
        assert small_stats["total_tokens_saved"] == 0

        large_stats = data["tiers"]["large"]
        assert large_stats["request_count"] == 0
        assert large_stats["avg_latency_ms"] == 0.0
        assert large_stats["p95_latency_ms"] == 0.0
        assert large_stats["total_prompt_tokens"] == 0
        assert large_stats["total_completion_tokens"] == 0
        assert large_stats["total_shadow_cost_usd"] == 0.0
        assert large_stats["hit_rate"] == 0.0
        assert large_stats["total_tokens_saved"] == 0
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_stats_with_populated_records(client: AsyncClient):
    """Verifies aggregation math for count, average, p95, tokens, and cache savings."""
    now = datetime.now(timezone.utc)
    mock_records = [
        RequestRecord(
            id=uuid.uuid4(),
            prompt="Simple query 1",
            response="Short answer 1",
            tier="small",
            model_name="ollama_chat/llama3.2:3b",
            cached=False,
            cost_usd=0.0,
            shadow_cost_usd=0.0001,
            latency_ms=10.0,
            prompt_tokens=100,
            completion_tokens=50,
            cache_status="miss",
            tokens_saved=0,
            saved_latency_ms=0.0,
            saved_shadow_cost_usd=0.0,
            created_at=now,
        ),
        RequestRecord(
            id=uuid.uuid4(),
            prompt="Simple query 2 (cached hit)",
            response="Short answer 2",
            tier="small",
            model_name="ollama_chat/llama3.2:3b",
            cached=True,
            cost_usd=0.0,
            shadow_cost_usd=0.0,
            latency_ms=2.0,
            prompt_tokens=0,
            completion_tokens=0,
            cache_status="exact_hit",
            tokens_saved=150,
            saved_latency_ms=15.0,
            saved_shadow_cost_usd=0.00015,
            created_at=now,
        ),
        RequestRecord(
            id=uuid.uuid4(),
            prompt="Complex query",
            response="Long reasoning response",
            tier="large",
            model_name="ollama_chat/llama3.1:8b",
            cached=False,
            cost_usd=0.0,
            shadow_cost_usd=0.005,
            latency_ms=250.0,
            prompt_tokens=500,
            completion_tokens=400,
            cache_status="miss",
            tokens_saved=0,
            saved_latency_ms=0.0,
            saved_shadow_cost_usd=0.0,
            created_at=now,
        ),
    ]

    mock_db = AsyncMock()
    mock_db.execute.return_value = MockExecuteResult(mock_records)

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = await client.get("/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 3
        # 1 hit out of 3 total requests = 0.3333
        assert data["hit_rate"] == 0.3333
        assert data["total_tokens_saved"] == 150
        assert data["total_saved_latency_ms"] == 15.0
        assert data["total_saved_shadow_cost_usd"] == 0.00015

        small = data["tiers"]["small"]
        assert small["request_count"] == 2
        assert small["avg_latency_ms"] == 6.0
        assert small["total_prompt_tokens"] == 100
        assert small["total_completion_tokens"] == 50
        assert small["total_shadow_cost_usd"] == 0.0001
        assert small["cache_status_counts"]["exact_hit"] == 1
        assert small["cache_status_counts"]["miss"] == 1
        assert small["hit_rate"] == 0.5
        assert small["total_tokens_saved"] == 150
        assert small["total_saved_latency_ms"] == 15.0
        assert small["total_saved_shadow_cost_usd"] == 0.00015

        large = data["tiers"]["large"]
        assert large["request_count"] == 1
        assert large["avg_latency_ms"] == 250.0
        assert large["p95_latency_ms"] == 250.0
        assert large["total_prompt_tokens"] == 500
        assert large["total_completion_tokens"] == 400
        assert large["total_shadow_cost_usd"] == 0.005
        assert large["hit_rate"] == 0.0
        assert large["total_tokens_saved"] == 0
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_stats_database_unreachable(client: AsyncClient):
    """Verifies that GET /v1/stats returns HTTP 503 when the database is unreachable."""
    mock_db = AsyncMock()
    mock_db.execute.side_effect = Exception("Connection refused: PostgreSQL host is down")

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = await client.get("/v1/stats")
        assert response.status_code == 503

        data = response.json()
        assert "detail" in data
        assert "Database is unreachable or unavailable." in data["detail"]
        assert "error" in data
        assert "Connection refused" in data["error"]
    finally:
        app.dependency_overrides.pop(get_db, None)


@pytest.mark.asyncio
async def test_stats_semantic_cache_metrics(client: AsyncClient):
    """Verifies that GET /v1/stats calculates semantic_hit_count and avg similarity of hits and misses."""
    now = datetime.now(timezone.utc)
    mock_records = [
        RequestRecord(
            id=uuid.uuid4(),
            prompt="Tell me about python",
            response="Python is high-level.",
            tier="small",
            model_name="ollama_chat/llama3.2:3b",
            cached=True,
            cost_usd=0.0,
            shadow_cost_usd=0.0,
            latency_ms=12.0,
            prompt_tokens=0,
            completion_tokens=0,
            cache_status="semantic_hit",
            tokens_saved=80,
            saved_latency_ms=110.0,
            saved_shadow_cost_usd=0.00003,
            similarity=0.94,
            created_at=now,
        ),
        RequestRecord(
            id=uuid.uuid4(),
            prompt="Tell me about rust",
            response="Rust is memory safe.",
            tier="small",
            model_name="ollama_chat/llama3.2:3b",
            cached=False,
            cost_usd=0.0,
            shadow_cost_usd=0.00003,
            latency_ms=130.0,
            prompt_tokens=30,
            completion_tokens=50,
            cache_status="miss",
            tokens_saved=0,
            saved_latency_ms=0.0,
            saved_shadow_cost_usd=0.0,
            similarity=0.72,
            created_at=now,
        ),
    ]

    mock_db = AsyncMock()
    mock_db.execute.return_value = MockExecuteResult(mock_records)

    app.dependency_overrides[get_db] = lambda: mock_db
    try:
        response = await client.get("/v1/stats")
        assert response.status_code == 200

        data = response.json()
        assert data["total_requests"] == 2
        assert data["semantic_hit_count"] == 1
        assert data["avg_similarity_hits"] == 0.94
        assert data["avg_similarity_misses"] == 0.72

        small = data["tiers"]["small"]
        assert small["semantic_hit_count"] == 1
        assert small["avg_similarity_hits"] == 0.94
        assert small["avg_similarity_misses"] == 0.72
        assert small["cache_status_counts"]["semantic_hit"] == 1
        assert small["cache_status_counts"]["miss"] == 1
    finally:
        app.dependency_overrides.pop(get_db, None)
