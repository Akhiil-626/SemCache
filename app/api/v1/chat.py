import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, status
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import AUTO_MODEL, settings
from app.db.session import get_db
from app.services.completion_service import completion_service

router = APIRouter(tags=["OpenAI Chat Completions"])


class ChatMessage(BaseModel):
    role: str = Field(..., description="Role of the message author (e.g. user, system, assistant)")
    content: str = Field(..., description="Content of the message")
    name: str | None = Field(default=None, description="Optional name of the participant")


class ChatCompletionRequest(BaseModel):
    messages: list[ChatMessage] = Field(..., min_length=1, description="List of messages in conversation")
    model: str | None = Field(default=None, description="Target model or tier (e.g. 'small', 'large')")
    temperature: float | None = Field(default=0.7, description="Sampling temperature")
    max_tokens: int | None = Field(default=None, description="Maximum completion tokens")
    stream: bool = Field(default=False, description="Streaming response flag")
    user_budget: float | None = Field(default=None, description="Optional budget limit in USD")
    bypass_cache: bool = Field(default=False, description="Force LLM generation bypassing cache")
    response_format: Any = Field(default=None, description="Optional response format (e.g. JSON schema)")


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str = "stop"


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo
    system_fingerprint: str = "semcache-fp"
    tier: str
    cached: bool
    cost_usd: float
    shadow_cost_usd: float
    latency_ms: float
    routing_rationale: str | None = None
    cache_status: str | None = None
    escalation_count: int = 0


def format_messages_to_prompt(messages: list[ChatMessage]) -> str:
    """Flattens OpenAI chat messages into a structured prompt."""
    if len(messages) == 1 and messages[0].role == "user":
        return messages[0].content
    return "\n\n".join(f"{msg.role.capitalize()}: {msg.content}" for msg in messages)


@router.post("/chat/completions", response_model=ChatCompletionResponse, status_code=status.HTTP_200_OK)
async def create_chat_completion(
    request: ChatCompletionRequest,
    cache_control: str | None = Header(default=None, alias="Cache-Control"),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    db: AsyncSession = Depends(get_db),
) -> ChatCompletionResponse:
    """OpenAI-compatible chat completion endpoint with exact key-value caching and cost routing."""
    valid_tier_names = [t.name for t in settings.tiers]
    valid_models = [AUTO_MODEL] + valid_tier_names

    req_model = request.model
    if req_model is not None and req_model != "" and req_model != AUTO_MODEL:
        if req_model not in valid_tier_names:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid model '{req_model}'. Valid values are: {', '.join(valid_models)}",
            )

    prompt = format_messages_to_prompt(request.messages)

    result = await completion_service.execute_completion(
        prompt=prompt,
        db=db,
        user_budget=request.user_budget,
        bypass_cache=request.bypass_cache,
        preferred_tier=request.model,
        tenant_id=x_tenant_id or "default",
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=request.stream,
        cache_control=cache_control,
        response_format=request.response_format,
    )

    created_timestamp = int(time.time())
    completion_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    return ChatCompletionResponse(
        id=completion_id,
        object="chat.completion",
        created=created_timestamp,
        model=result.model_name,
        choices=[
            ChatChoice(
                index=0,
                message=ChatMessage(role="assistant", content=result.content),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.prompt_tokens + result.completion_tokens,
        ),
        tier=result.tier,
        cached=result.cached,
        cost_usd=result.cost_usd,
        shadow_cost_usd=result.shadow_cost_usd,
        latency_ms=result.latency_ms,
        routing_rationale=result.routing_rationale,
        cache_status=result.cache_status,
        escalation_count=getattr(result, "escalation_count", 0),
    )
