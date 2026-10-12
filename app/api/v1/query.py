from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.session import get_db
from app.services.completion_service import completion_service

router = APIRouter(tags=["Query"])


class QueryRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt query to evaluate and execute")
    user_budget: float | None = Field(default=None, description="Optional per-query budget in USD")
    bypass_cache: bool = Field(default=False, description="Force LLM generation bypassing cache")


class QueryResponse(BaseModel):
    response: str
    cached: bool
    tier: str
    model_used: str
    similarity_score: float
    cost_usd: float
    shadow_cost_usd: float
    latency_ms: float
    routing_rationale: str | None = None


@router.post("/query", response_model=QueryResponse, status_code=status.HTTP_200_OK)
async def execute_query(
    request: QueryRequest,
    db: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """Executes a query as a thin wrapper over the unified CompletionService."""
    result = await completion_service.execute_completion(
        prompt=request.prompt,
        db=db,
        user_budget=request.user_budget,
        bypass_cache=request.bypass_cache,
    )

    return QueryResponse(
        response=result.content,
        cached=result.cached,
        tier=result.tier,
        model_used=result.model_name,
        similarity_score=result.similarity_score,
        cost_usd=result.cost_usd,
        shadow_cost_usd=result.shadow_cost_usd,
        latency_ms=result.latency_ms,
        routing_rationale=result.routing_rationale,
    )
