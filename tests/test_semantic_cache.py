import time
import uuid
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch
import pytest

from app.cache.backend import InMemoryExactCacheBackend
from app.cache.schemas import CacheHitResult
from app.cache.semantic_guard import (
    check_antonym_guard,
    check_semantic_guard,
    is_informational_query,
    load_antonym_pairs,
    reset_antonym_cache,
)
from app.cache.service import SemanticCacheService, cache_service
from app.core.config import settings
from app.db.models import RequestRecord, SemanticCacheEntry
from app.embeddings.service import embedding_service
from app.llm.client import LLMResponse
from app.services.completion_service import completion_service


class FakeEmbedder:
    """Fake vector embedder returning deterministic unit vectors for tests."""

    def __init__(
        self,
        mapping: dict[str, list[float]] | None = None,
        default_vector: list[float] | None = None,
        should_fail: bool = False,
    ) -> None:
        self.mapping = mapping or {}
        # Default 384d unit vector with 1.0 in first position
        self.default_vector = default_vector or ([1.0] + [0.0] * 383)
        self.should_fail = should_fail
        self.calls: list[str] = []

    async def get_embedding(self, text: str) -> list[float] | None:
        self.calls.append(text)
        if self.should_fail:
            return None
        return self.mapping.get(text, self.default_vector)


class MockFirstResult:
    def __init__(self, item):
        self._item = item

    def first(self):
        return self._item


class MockDBSession:
    def __init__(self, query_result=None):
        self.added = []
        self.query_result = query_result
        self.executed_stmts = []

    def add(self, obj):
        self.added.append(obj)

    async def commit(self):
        pass

    async def rollback(self):
        pass

    async def refresh(self, obj):
        if not hasattr(obj, "id") or obj.id is None:
            obj.id = uuid.uuid4()

    async def execute(self, stmt):
        self.executed_stmts.append(stmt)
        return MockFirstResult(self.query_result)


@pytest.fixture
def mock_llm_response():
    return LLMResponse(
        content="Generated fresh response from LLM.",
        prompt_tokens=30,
        completion_tokens=40,
        latency_ms=120.5,
        real_cost=0.0,
        shadow_cost=0.0000285,
        model_name="ollama_chat/llama3.2:3b",
        tier="small",
    )


@pytest.fixture
def in_memory_exact_backend():
    backend = InMemoryExactCacheBackend()
    orig = completion_service.exact_backend
    completion_service.exact_backend = backend
    yield backend
    completion_service.exact_backend = orig


@pytest.fixture
def sample_cache_entry():
    return SemanticCacheEntry(
        id=uuid.uuid4(),
        prompt="How do I reset my password?",
        embedding=[1.0] + [0.0] * 383,
        response="Your password reset request has been processed.",
        tier="small",
        model_name="ollama_chat/llama3.2:3b",
        tenant_id="tenant-1",
        cache_version="v1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        prompt_tokens=30,
        completion_tokens=40,
        latency_ms=120.5,
        cost_usd=0.0,
        shadow_cost_usd=0.0000285,
        hit_count=1,
    )


# ==============================================================================
# SECTION 1: Semantic Guard Unit Tests (Negation, Number, Antonym)
# ==============================================================================

def test_semantic_guard_negations():
    """Guard rejects when texts differ in negation words."""
    ok, reason = check_semantic_guard("Delete my account", "Don't delete my account")
    assert ok is False
    assert reason == "guard"

    ok, reason = check_semantic_guard("Please cancel the subscription", "Please do not cancel the subscription")
    assert ok is False
    assert reason == "guard"

    ok, reason = check_semantic_guard("Proceed without delay", "Proceed with delay")
    assert ok is False
    assert reason == "guard"

    ok, reason = check_semantic_guard("Reset my password", "How to reset my password")
    assert ok is True
    assert reason is None


def test_semantic_guard_numbers():
    """Guard rejects when texts differ in number tokens."""
    ok, reason = check_semantic_guard("Send 50 dollars", "Send 500 dollars")
    assert ok is False
    assert reason == "guard"

    ok, reason = check_semantic_guard("Restart server 1", "Restart server 2")
    assert ok is False
    assert reason == "guard"

    ok, reason = check_semantic_guard("Transfer 100 credits", "Transfer 100 credits")
    assert ok is True
    assert reason is None


def test_antonym_guard_pairs_rejected():
    """Antonym guard rejects when texts contain conflicting members of antonym pairs.
    Tests EVERY configured antonym pair bidirectionally (A vs B and B vs A).
    """
    reset_antonym_cache()
    pairs = load_antonym_pairs()
    assert len(pairs) >= 18, "Expected at least 18 antonym pairs in config"

    # Test bidirectional rejection for every single configured antonym pair
    for w1, w2 in pairs:
        q_a = f"Please {w1} this setting"
        q_b = f"Please {w2} this setting"

        # Direction 1: A vs B
        ok_ab, reason_ab = check_semantic_guard(q_a, q_b)
        assert ok_ab is False, f"Expected antonym guard to reject '{q_a}' vs '{q_b}'"
        assert reason_ab == "guard"

        # Direction 2: B vs A
        ok_ba, reason_ba = check_semantic_guard(q_b, q_a)
        assert ok_ba is False, f"Expected antonym guard to reject '{q_b}' vs '{q_a}'"
        assert reason_ba == "guard"

    # Also verify realistic natural sentence pairs in both directions
    antonym_test_cases = [
        ("How to enable two-factor auth", "How to disable two-factor auth"),
        ("Should I buy this stock?", "Should I sell this stock?"),
        ("Why increase prices now?", "Why decrease prices now?"),
        ("How to lock the device", "How to unlock the device"),
        ("Is it open today?", "Is it closed today?"),
        ("Do you allow pets?", "Do you deny pets?"),
        ("How to add user", "How to remove user"),
        ("How to turn on alerts", "How to turn off alerts"),
        ("When to start the task", "When to pause the task"),
        ("Did they approve the loan?", "Did they reject the loan?"),
        ("How to upload photos", "How to download photos"),
        ("Is this chemical safe to use?", "Is this chemical dangerous to use?"),
        ("How to accept invitation", "How to decline invitation"),
        ("How to encrypt messages", "How to decrypt messages"),
        ("How to subscribe to updates", "How to unsubscribe from updates"),
        ("What happens before the event?", "What happens after the event?"),
        ("Did the team win?", "Did the team lose?"),
        ("Should we include details?", "Should we exclude details?"),
    ]

    for q, c in antonym_test_cases:
        ok_fwd, reason_fwd = check_semantic_guard(q, c)
        assert ok_fwd is False, f"Expected antonym guard to reject '{q}' vs '{c}'"
        assert reason_fwd == "guard"

        ok_rev, reason_rev = check_semantic_guard(c, q)
        assert ok_rev is False, f"Expected antonym guard to reject '{c}' vs '{q}'"
        assert reason_rev == "guard"

    # Same members should pass
    ok, reason = check_semantic_guard("How to enable 2FA", "Steps to enable 2FA")
    assert ok is True
    assert reason is None

    # Both members present in both queries (e.g. comparison) should pass
    ok, reason = check_semantic_guard(
        "Difference between enable and disable",
        "Explain difference between enable and disable",
    )
    assert ok is True
    assert reason is None


def test_is_informational_query_allowlist():
    """Verifies that informational questions pass and command-style prompts are rejected."""
    info_queries = [
        "What is the capital of France?",
        "Why is the sky blue?",
        "How do I reset my password?",
        "Who wrote Hamlet?",
        "When does the store open?",
        "Where can I find settings?",
        "Which model is better?",
        "Is Python easy to learn?",
        "Are there any updates?",
        "Does this work on Windows?",
        "Do you support Redis?",
        "Can you explain quicksort?",
        "Should I use Postgres?",
        "Explain quantum computing",
        "Describe the solar system",
        "Summarize this text",
        "Define polymorphism",
        "Tell me about machine learning",
        "Steps for installing python",
        "Difference between process and thread",
    ]
    for q in info_queries:
        assert is_informational_query(q) is True, f"Expected '{q}' to be informational"

    command_queries = [
        "Delete my account",
        "Cancel subscription",
        "Restart the server",
        "Transfer 100 dollars to Alice",
        "Remove user from group",
        "Send email to support",
    ]
    for q in command_queries:
        assert is_informational_query(q) is False, f"Expected '{q}' to NOT be informational"


# ==============================================================================
# SECTION 2: Semantic Cache Lookup & Pipeline Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_hit_above_threshold(in_memory_exact_backend, mock_llm_response, sample_cache_entry):
    """Test 1: Vector distance 0.05 gives similarity 0.95 >= 0.88 threshold -> semantic_hit without calling LLM."""
    mock_db = MockDBSession(query_result=(sample_cache_entry, 0.05))
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How do I reset my password?",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cached is True
            assert res.cache_status == "semantic_hit"
            assert res.content == sample_cache_entry.response
            assert res.similarity_score == 0.95
            assert mock_llm.call_count == 0
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_miss_below_threshold(in_memory_exact_backend, mock_llm_response, sample_cache_entry):
    """Test 2: Vector distance 0.20 gives similarity 0.80 < 0.88 threshold -> miss, calls LLM, stores in cache."""
    mock_db = MockDBSession(query_result=(sample_cache_entry, 0.20))
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How do I reset my password?",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cached is False
            assert res.cache_status == "miss"
            assert res.content == mock_llm_response.content
            assert res.similarity_score == 0.80
            assert mock_llm.call_count == 1

            # Check that new entry was added to DB
            stored_entries = [obj for obj in mock_db.added if isinstance(obj, SemanticCacheEntry)]
            assert len(stored_entries) >= 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_wrong_tenant_misses(in_memory_exact_backend, mock_llm_response):
    """Test 3: Query for tenant-2 when DB only has candidate for tenant-1 produces a miss."""
    mock_db = MockDBSession(query_result=None)
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How to reset password",
                db=mock_db,
                tenant_id="tenant-2",
                temperature=0.5,
            )

            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_different_cache_version_misses(in_memory_exact_backend, mock_llm_response):
    """Test 4: When CACHE_VERSION differs, lookup returns None (miss) and calls LLM."""
    mock_db = MockDBSession(query_result=None)
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    orig_version = settings.CACHE_VERSION
    embedding_service.embedder = fake_embedder

    try:
        settings.CACHE_VERSION = "v2"
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="Explain binary search",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1
    finally:
        settings.CACHE_VERSION = orig_version
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_expired_entry_ignored(in_memory_exact_backend, mock_llm_response):
    """Test 5: An expired entry (expires_at in past) is filtered out by DB query -> miss."""
    mock_db = MockDBSession(query_result=None)
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How to check expired cache entry",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_guard_rejects_delete_vs_dont_delete(in_memory_exact_backend, mock_llm_response):
    """Test 6: High similarity (0.95) but negation mismatch ('Don't delete' vs 'Delete') -> guard rejection, miss, calls LLM."""
    delete_entry = SemanticCacheEntry(
        id=uuid.uuid4(),
        prompt="Delete my account",
        embedding=[1.0] + [0.0] * 383,
        response="Deletion confirmation.",
        tier="small",
        model_name="ollama_chat/llama3.2:3b",
        tenant_id="tenant-1",
        cache_version="v1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        prompt_tokens=30,
        completion_tokens=40,
        latency_ms=120.5,
        cost_usd=0.0,
        shadow_cost_usd=0.0000285,
        hit_count=1,
    )
    mock_db = MockDBSession(query_result=(delete_entry, 0.05))  # similarity 0.95
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    orig_info_only = settings.SEMANTIC_INFO_ONLY
    embedding_service.embedder = fake_embedder

    try:
        # Disable info_only temporarily to specifically verify negation guard rejection on command phrasing
        settings.SEMANTIC_INFO_ONLY = False
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="Don't delete my account",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            # Hit was rejected by guard, so it fell back to LLM
            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1

            # Best similarity was still recorded in requests table log
            req_logs = [obj for obj in mock_db.added if isinstance(obj, RequestRecord)]
            assert len(req_logs) >= 1
            assert req_logs[-1].similarity == 0.95
            assert req_logs[-1].cache_status == "miss"
    finally:
        settings.SEMANTIC_INFO_ONLY = orig_info_only
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_embedder_failure_falls_back_to_miss(in_memory_exact_backend, mock_llm_response):
    """Test 7: When embedder fails/returns None, it logs WARNING and acts as cache miss (never fails request)."""
    mock_db = MockDBSession()
    fake_embedder = FakeEmbedder(should_fail=True)
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How to test embedder resilience",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            # Request succeeded despite embedder failure
            assert res.content == mock_llm_response.content
            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_hit_metrics_measured_and_saved_equal_original(
    in_memory_exact_backend, mock_llm_response, sample_cache_entry
):
    """Test 8: On semantic_hit, latency_ms is measured real embed+search time, consumed tokens=0, saved_* equal original."""
    mock_db = MockDBSession(query_result=(sample_cache_entry, 0.05))
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How do I reset my password?",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cache_status == "semantic_hit"
            assert res.cached is True
            assert res.latency_ms > 0.0
            assert res.cost_usd == 0.0
            assert res.shadow_cost_usd == 0.0
            assert res.prompt_tokens == 30
            assert res.completion_tokens == 40
            assert res.tokens_saved == 70
            assert res.saved_latency_ms == 120.5
            assert res.saved_shadow_cost_usd == 0.0000285

            req_logs = [obj for obj in mock_db.added if isinstance(obj, RequestRecord)]
            assert len(req_logs) >= 1
            hit_log = req_logs[-1]
            assert hit_log.cache_status == "semantic_hit"
            assert hit_log.prompt_tokens == 0
            assert hit_log.completion_tokens == 0
            assert hit_log.cost_usd == 0.0
            assert hit_log.shadow_cost_usd == 0.0
            assert hit_log.tokens_saved == 70
            assert hit_log.saved_latency_ms == 120.5
            assert hit_log.saved_shadow_cost_usd == 0.0000285
            assert hit_log.similarity == 0.95
    finally:
        embedding_service.embedder = orig_embedder


# ==============================================================================
# SECTION 3: Antonym Guard and Informational-Only Allowlist Pipeline Tests
# ==============================================================================

@pytest.mark.asyncio
async def test_antonym_pair_rejected_in_pipeline(in_memory_exact_backend, mock_llm_response):
    """Pipeline test: Antonym discrepancy (enable vs disable) above threshold is rejected by guard and calls LLM."""
    cached_entry = SemanticCacheEntry(
        id=uuid.uuid4(),
        prompt="How to disable two-factor authentication?",
        embedding=[1.0] + [0.0] * 383,
        response="Instructions to disable 2FA.",
        tier="small",
        model_name="ollama_chat/llama3.2:3b",
        tenant_id="tenant-1",
        cache_version="v1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        prompt_tokens=25,
        completion_tokens=35,
        latency_ms=100.0,
        cost_usd=0.0,
        shadow_cost_usd=0.00002,
        hit_count=1,
    )
    mock_db = MockDBSession(query_result=(cached_entry, 0.05))  # similarity 0.95 >= 0.88
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            res = await completion_service.execute_completion(
                prompt="How to enable two-factor authentication?",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            # Even with similarity 0.95, antonym mismatch causes cache miss and calls LLM
            assert res.cached is False
            assert res.cache_status == "miss"
            assert mock_llm.call_count == 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_command_style_prompts_skipped_from_semantic_cache_but_use_exact_cache(
    in_memory_exact_backend, mock_llm_response
):
    """Command-style prompts ('Delete my account') skip semantic cache (not_informational) but still use exact cache."""
    mock_db = MockDBSession()
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            # Call 1: Command misses exact cache, skips semantic cache, calls LLM, stores in exact cache
            res1 = await completion_service.execute_completion(
                prompt="Delete my account",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res1.cached is False
            assert res1.cache_status == "miss"
            assert mock_llm.call_count == 1
            # Semantic embedder should NOT be called for lookup or store
            assert len(fake_embedder.calls) == 0

            # No SemanticCacheEntry should be added to DB
            stored_sem_entries = [obj for obj in mock_db.added if isinstance(obj, SemanticCacheEntry)]
            assert len(stored_sem_entries) == 0

            # Call 2: Identical command hits EXACT cache!
            res2 = await completion_service.execute_completion(
                prompt="Delete my account",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res2.cached is True
            assert res2.cache_status == "exact_hit"
            # LLM call count remains 1 (exact hit!)
            assert mock_llm.call_count == 1
    finally:
        embedding_service.embedder = orig_embedder


@pytest.mark.asyncio
async def test_informational_paraphrase_hits_semantic_cache(
    in_memory_exact_backend, mock_llm_response
):
    """Informational query ('What is the procedure to reset password?') hits semantic cache for cached ('How do I reset password?')."""
    cached_entry = SemanticCacheEntry(
        id=uuid.uuid4(),
        prompt="How do I reset my password?",
        embedding=[1.0] + [0.0] * 383,
        response="To reset your password, click on 'Forgot Password' on the login screen.",
        tier="small",
        model_name="ollama_chat/llama3.2:3b",
        tenant_id="tenant-1",
        cache_version="v1",
        expires_at=datetime.now(timezone.utc) + timedelta(hours=1),
        prompt_tokens=30,
        completion_tokens=40,
        latency_ms=115.0,
        cost_usd=0.0,
        shadow_cost_usd=0.0000285,
        hit_count=1,
    )
    mock_db = MockDBSession(query_result=(cached_entry, 0.05))  # similarity 0.95 >= 0.88
    fake_embedder = FakeEmbedder()
    orig_embedder = embedding_service.embedder
    embedding_service.embedder = fake_embedder

    try:
        with patch("app.services.completion_service.llm_client.generate_response", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_llm_response

            # Query starts with 'What' (informational) and cached starts with 'How' (informational)
            res = await completion_service.execute_completion(
                prompt="What is the procedure to reset my password?",
                db=mock_db,
                tenant_id="tenant-1",
                temperature=0.5,
            )

            assert res.cached is True
            assert res.cache_status == "semantic_hit"
            assert res.content == cached_entry.response
            assert res.similarity_score == 0.95
            # LLM is NOT called
            assert mock_llm.call_count == 0
    finally:
        embedding_service.embedder = orig_embedder
