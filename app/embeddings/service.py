import logging
import math
from typing import Any, Protocol

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbedderInterface(Protocol):
    """Protocol for vector embedders."""

    async def get_embedding(self, text: str) -> list[float] | None:
        """Returns L2-normalized vector embedding for given text, or None on failure."""
        ...


class OllamaEmbedder:
    """Invokes local Ollama embed API (POST /api/embed) and L2-normalizes the result."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = 5.0,
    ) -> None:
        self.base_url = (base_url or settings.OLLAMA_BASE_URL).rstrip("/")
        self.model = model or settings.EMBEDDING_MODEL
        self.timeout = timeout

    async def get_embedding(self, text: str) -> list[float] | None:
        if not text:
            return None

        url = f"{self.base_url}/api/embed"
        payload = {"model": self.model, "input": text}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data: dict[str, Any] = response.json()

            embeddings = data.get("embeddings", [])
            if not embeddings or not isinstance(embeddings, list):
                logger.warning(
                    "Ollama embedding response missing embeddings array: model=%s",
                    self.model,
                )
                return None

            raw_vector = embeddings[0]
            if not isinstance(raw_vector, list) or not raw_vector:
                return None

            # L2-normalization: v / ||v||_2
            norm = math.sqrt(sum(x * x for x in raw_vector))
            if norm > 0.0:
                normalized = [float(x) / norm for x in raw_vector]
            else:
                normalized = [float(x) for x in raw_vector]

            return normalized

        except Exception as exc:
            logger.warning(
                "Ollama embedding failed (treating as cache miss): %s: %s",
                type(exc).__name__,
                str(exc),
                extra={"model": self.model, "url": url},
            )
            return None


class EmbeddingService:
    """Manages active text embedder with test injection support."""

    def __init__(self, embedder: EmbedderInterface | None = None) -> None:
        self._embedder = embedder

    @property
    def embedder(self) -> EmbedderInterface:
        if self._embedder is None:
            self._embedder = OllamaEmbedder()
        return self._embedder

    @embedder.setter
    def embedder(self, value: EmbedderInterface) -> None:
        self._embedder = value

    async def get_embedding(self, text: str) -> list[float] | None:
        return await self.embedder.get_embedding(text)


embedding_service = EmbeddingService()
