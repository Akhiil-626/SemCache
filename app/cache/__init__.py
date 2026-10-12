from app.cache.backend import (
    ExactCacheBackend,
    InMemoryExactCacheBackend,
    RedisExactCacheBackend,
    get_exact_cache_backend,
    reset_exact_cache_backend,
)
from app.cache.key import build_exact_cache_key
from app.cache.policies import (
    INFORMATIONAL_PREFIXES,
    PolicyDecision,
    evaluate_cache_policy,
    is_informational_query,
)
from app.cache.schemas import CacheEntryResponse, CacheHitResult, CacheSaveRequest
from app.cache.semantic_guard import (
    check_antonym_guard,
    check_semantic_guard,
    load_antonym_pairs,
    reset_antonym_cache,
)
from app.cache.service import SemanticCacheService, cache_service

__all__ = [
    "CacheHitResult",
    "CacheSaveRequest",
    "CacheEntryResponse",
    "SemanticCacheService",
    "cache_service",
    "ExactCacheBackend",
    "RedisExactCacheBackend",
    "InMemoryExactCacheBackend",
    "get_exact_cache_backend",
    "reset_exact_cache_backend",
    "build_exact_cache_key",
    "evaluate_cache_policy",
    "PolicyDecision",
    "check_semantic_guard",
    "check_antonym_guard",
    "load_antonym_pairs",
    "reset_antonym_cache",
    "is_informational_query",
    "INFORMATIONAL_PREFIXES",
]
