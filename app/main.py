import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

import urllib.parse
from app.api.router import api_router, v1_router
from app.cache.backend import RedisExactCacheBackend, get_exact_cache_backend
from app.core.config import settings
from app.core.logging import setup_logging
from app.db.session import init_db
from app.llm.startup_check import verify_ollama_models

setup_logging(log_level=settings.LOG_LEVEL)
logger = logging.getLogger("semcache.main")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manages application startup and shutdown lifecycle events without fatal crashes."""
    logger.info("Starting up semcache service", extra={"environment": settings.ENVIRONMENT})

    # Non-fatal Ollama verification check
    try:
        await verify_ollama_models()
    except Exception as exc:
        logger.warning(
            "Ollama startup verification warning (service started, but Ollama is not ready)",
            extra={"error": str(exc)},
        )

    # Non-fatal database initialization check
    try:
        await init_db()
        logger.info("Database initialized with pgvector extension and schemas.")
    except Exception as exc:
        logger.warning(
            "Database connection deferred (service started, but PostgreSQL is not ready)",
            extra={"error": str(exc)},
        )

    # Exact cache startup check & Redis ping
    backend = get_exact_cache_backend()
    backend_kind = "redis" if isinstance(backend, RedisExactCacheBackend) else "memory"
    parsed = urllib.parse.urlparse(settings.REDIS_URL)
    redis_host = parsed.hostname or "localhost"
    redis_port = parsed.port or 6379
    redis_db = parsed.path.lstrip("/") or "0"

    logger.info(
        "Exact cache backend active: %s (host=%s, port=%s, db=%s)",
        backend_kind,
        redis_host,
        redis_port,
        redis_db,
    )

    if backend_kind == "redis":
        try:
            import redis.asyncio as aioredis
            test_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
            )
            await test_client.ping()
            await test_client.aclose()
            logger.info("Redis ping successful at %s:%s db=%s", redis_host, redis_port, redis_db)
        except Exception as exc:
            logger.warning(
                "Redis ping failed at startup (%s:%s db=%s): %s: %s",
                redis_host,
                redis_port,
                redis_db,
                type(exc).__name__,
                str(exc),
            )

    yield
    logger.info("Shutting down semcache service.")


def create_application() -> FastAPI:
    """FastAPI application factory."""
    application = FastAPI(
        title="SemCache API",
        description="Production Semantic Caching and Cost-Aware LLM Routing Service with Ollama",
        version="0.2.0",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @application.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            "HTTP Request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": round(process_time, 2),
            },
        )
        return response

    # Mount routers
    application.include_router(api_router)
    application.include_router(v1_router)

    return application


app = create_application()
