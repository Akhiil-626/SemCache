from fastapi import APIRouter, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.db.session import check_db_health
from app.llm.startup_check import check_ollama_health

router = APIRouter(tags=["Health"])


async def check_redis_health() -> tuple[bool, str | None]:
    """Validates Redis dependency connectivity."""
    if not settings.EXACT_CACHE_ENABLED:
        return True, None
    try:
        import redis.asyncio as aioredis
        client = aioredis.from_url(settings.REDIS_URL, encoding="utf-8", decode_responses=True)
        await client.ping()
        await client.aclose()
        return True, None
    except Exception as exc:
        return False, str(exc)


@router.get("/health", status_code=status.HTTP_200_OK)
async def liveness_check() -> dict[str, str]:
    """Pure liveness check: does not touch database, redis, or ollama."""
    return {"status": "ok"}


@router.get("/health/ready")
async def readiness_check() -> JSONResponse:
    """Readiness probe: validates dependencies (Postgres, Ollama, and Redis)."""
    db_ok, db_err = await check_db_health()
    ollama_ok, ollama_err = await check_ollama_health()
    redis_ok, redis_err = await check_redis_health()

    dependencies = {
        "database": {
            "status": "up" if db_ok else "down",
            **({"error": db_err} if not db_ok else {}),
        },
        "ollama": {
            "status": "up" if ollama_ok else "down",
            **({"error": ollama_err} if not ollama_ok else {}),
        },
        "redis": {
            "status": "ok" if redis_ok else (redis_err or "down"),
            **({"error": redis_err} if not redis_ok else {}),
        },
    }

    if db_ok and ollama_ok and redis_ok:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"status": "ready", "dependencies": dependencies},
        )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"status": "unready", "dependencies": dependencies},
    )
