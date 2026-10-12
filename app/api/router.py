from fastapi import APIRouter

from app.api.v1.chat import router as chat_router
from app.api.v1.health import router as health_router
from app.api.v1.query import router as query_router
from app.api.v1.stats import router as stats_router

# Existing api router with /health and /api/v1/query
api_router = APIRouter()
api_router.include_router(health_router, prefix="", tags=["Health"])
api_router.include_router(query_router, prefix="/api/v1", tags=["Query"])

# Separate v1 router mounted at /v1 (no /api prefix)
v1_router = APIRouter(prefix="/v1")
v1_router.include_router(chat_router, prefix="", tags=["OpenAI Chat Completions"])
v1_router.include_router(stats_router, prefix="", tags=["Stats"])
