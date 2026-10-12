import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Callable

from app.core.config import settings

logger = logging.getLogger(__name__)


class ExactCacheBackend(ABC):
    """Abstract interface for exact key-value cache backends with TTL support."""

    @abstractmethod
    async def get(self, key: str) -> dict[str, Any] | None:
        """Retrieves and deserializes JSON cache payload for key. Returns None on miss or error."""
        ...

    @abstractmethod
    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        """Serializes and stores JSON cache payload with expiration TTL."""
        ...

    @abstractmethod
    async def delete(self, key: str) -> None:
        """Deletes key from cache."""
        ...

    @abstractmethod
    async def clear(self) -> None:
        """Evicts all cached items."""
        ...


class RedisExactCacheBackend(ExactCacheBackend):
    """Async Redis implementation of ExactCacheBackend with visible logging."""

    def __init__(self, redis_url: str) -> None:
        self.redis_url = redis_url
        self._client: Any = None

    async def _get_client(self) -> Any:
        if self._client is None:
            try:
                import redis.asyncio as aioredis
                self._client = aioredis.from_url(
                    self.redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except Exception as exc:
                logger.warning(
                    "Redis client initialization failed: %s: %s",
                    type(exc).__name__,
                    str(exc),
                )
                raise
        return self._client

    async def get(self, key: str) -> dict[str, Any] | None:
        key_hash = key[:8] if key else "none"
        backend_name = self.__class__.__name__
        try:
            client = await self._get_client()
            raw = await client.get(key)
            if raw:
                data = json.loads(raw)
                logger.info(
                    "Exact cache lookup: backend=%s key=%s outcome=hit ttl=%s",
                    backend_name,
                    key_hash,
                    settings.CACHE_TTL_SECONDS,
                )
                return data
            logger.info(
                "Exact cache lookup: backend=%s key=%s outcome=miss ttl=%s",
                backend_name,
                key_hash,
                "-",
            )
            return None
        except Exception as exc:
            logger.warning(
                "Exact cache lookup error: %s: %s",
                type(exc).__name__,
                str(exc),
                extra={"backend": backend_name, "key": key_hash},
            )
            return None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        key_hash = key[:8] if key else "none"
        backend_name = self.__class__.__name__
        try:
            client = await self._get_client()
            raw = json.dumps(value)
            await client.set(key, raw, ex=ttl_seconds)
            logger.info(
                "Exact cache store: backend=%s key=%s outcome=stored ttl=%s",
                backend_name,
                key_hash,
                ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Exact cache store error: %s: %s",
                type(exc).__name__,
                str(exc),
                extra={"backend": backend_name, "key": key_hash, "ttl": ttl_seconds},
            )

    async def delete(self, key: str) -> None:
        try:
            client = await self._get_client()
            if client is not None:
                await client.delete(key)
        except Exception as exc:
            logger.warning("Redis error on key delete: %s: %s", type(exc).__name__, str(exc))

    async def clear(self) -> None:
        try:
            client = await self._get_client()
            if client is not None:
                await client.flushdb()
        except Exception as exc:
            logger.warning("Redis error on flushdb: %s: %s", type(exc).__name__, str(exc))

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
            self._client = None


class InMemoryExactCacheBackend(ExactCacheBackend):
    """In-memory TTL cache backend with controllable clock function for deterministic tests."""

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._store: dict[str, tuple[dict[str, Any], float]] = {}

    async def get(self, key: str) -> dict[str, Any] | None:
        key_hash = key[:8] if key else "none"
        backend_name = self.__class__.__name__
        try:
            entry = self._store.get(key)
            if entry is None:
                logger.info(
                    "Exact cache lookup: backend=%s key=%s outcome=miss ttl=%s",
                    backend_name,
                    key_hash,
                    "-",
                )
                return None
            val, expiry = entry
            now = self.clock()
            if now >= expiry:
                del self._store[key]
                logger.info(
                    "Exact cache lookup: backend=%s key=%s outcome=miss ttl=%s",
                    backend_name,
                    key_hash,
                    "-",
                )
                return None
            remaining_ttl = max(0, int(expiry - now))
            logger.info(
                "Exact cache lookup: backend=%s key=%s outcome=hit ttl=%s",
                backend_name,
                key_hash,
                remaining_ttl,
            )
            return val
        except Exception as exc:
            logger.warning(
                "Exact cache lookup error: %s: %s",
                type(exc).__name__,
                str(exc),
                extra={"backend": backend_name, "key": key_hash},
            )
            return None

    async def set(self, key: str, value: dict[str, Any], ttl_seconds: int = 3600) -> None:
        key_hash = key[:8] if key else "none"
        backend_name = self.__class__.__name__
        try:
            # Validate JSON serialization to mirror Redis behavior
            json.dumps(value)
            expiry = self.clock() + ttl_seconds
            self._store[key] = (value, expiry)
            logger.info(
                "Exact cache store: backend=%s key=%s outcome=stored ttl=%s",
                backend_name,
                key_hash,
                ttl_seconds,
            )
        except Exception as exc:
            logger.warning(
                "Exact cache store error: %s: %s",
                type(exc).__name__,
                str(exc),
                extra={"backend": backend_name, "key": key_hash, "ttl": ttl_seconds},
            )

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def clear(self) -> None:
        self._store.clear()


_backend_singleton: ExactCacheBackend | None = None


def get_exact_cache_backend(force_type: str | None = None) -> ExactCacheBackend:
    """Backend factory returning RedisExactCacheBackend when enabled, else InMemoryExactCacheBackend."""
    global _backend_singleton

    if force_type == "memory":
        return InMemoryExactCacheBackend()
    if force_type == "redis":
        return RedisExactCacheBackend(redis_url=settings.REDIS_URL)

    if _backend_singleton is not None:
        return _backend_singleton

    if settings.EXACT_CACHE_ENABLED and settings.REDIS_URL:
        _backend_singleton = RedisExactCacheBackend(redis_url=settings.REDIS_URL)
    else:
        _backend_singleton = InMemoryExactCacheBackend()

    return _backend_singleton


def reset_exact_cache_backend() -> None:
    """Resets backend singleton for tests."""
    global _backend_singleton
    _backend_singleton = None
