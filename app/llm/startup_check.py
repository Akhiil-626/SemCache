import logging
import httpx
from app.core.config import ModelTierConfig, settings

logger = logging.getLogger(__name__)


async def verify_ollama_models(
    ollama_base_url: str | None = None,
    tiers: list[ModelTierConfig] | None = None,
    client: httpx.AsyncClient | None = None,
) -> None:
    """Verifies that the Ollama instance is reachable and all configured tier models are present.

    If Ollama is unreachable or a model is missing, raises RuntimeError with exact
    'ollama pull <model>' commands.
    """
    base_url = (ollama_base_url or settings.OLLAMA_BASE_URL).rstrip("/")
    tags_url = f"{base_url}/api/tags"
    target_tiers = tiers or settings.tiers
    timeout = float(settings.OLLAMA_REQUEST_TIMEOUT)

    try:
        if client is not None:
            response = await client.get(tags_url)
        else:
            async with httpx.AsyncClient(timeout=timeout) as http_client:
                response = await http_client.get(tags_url)

        if response.status_code != 200:
            raise RuntimeError(
                f"Ollama health check failed: GET {tags_url} returned HTTP {response.status_code}. "
                f"Response: {response.text}"
            )
        data = response.json()
    except httpx.RequestError as exc:
        raise RuntimeError(
            f"Ollama server is unreachable at '{base_url}'. "
            f"Ensure Ollama is running and accessible (e.g., run `ollama serve`). "
            f"Connection error: {exc}"
        ) from exc

    raw_models = data.get("models", [])
    available_tags: set[str] = set()
    for m in raw_models:
        if isinstance(m, dict):
            if "name" in m and m["name"]:
                available_tags.add(m["name"])
            if "model" in m and m["model"]:
                available_tags.add(m["model"])

    missing_models: list[str] = []
    for tier in target_tiers:
        # Strip provider prefix (e.g., 'ollama_chat/' -> 'llama3.2:3b')
        ollama_model_tag = tier.model.split("/", 1)[-1]
        is_available = (
            ollama_model_tag in available_tags
            or f"{ollama_model_tag}:latest" in available_tags
            or (":" not in ollama_model_tag and any(tag.startswith(f"{ollama_model_tag}:") for tag in available_tags))
        )
        if not is_available:
            missing_models.append(ollama_model_tag)

    if missing_models:
        pull_commands = "\n".join(f"  ollama pull {model}" for model in missing_models)
        raise RuntimeError(
            f"Ollama startup verification failed! Missing required models:\n"
            f"{pull_commands}\n\n"
            f"Please run the command(s) above on your host machine."
        )

    logger.info("Ollama startup check passed: all configured tier models are available.")


async def check_ollama_health() -> tuple[bool, str | None]:
    """Non-fatal check for readiness probe returning (is_healthy, error_details)."""
    try:
        await verify_ollama_models()
        return True, None
    except Exception as exc:
        return False, str(exc)
