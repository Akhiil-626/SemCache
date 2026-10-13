from pathlib import Path
from typing import Any, Literal
import yaml
from pydantic import BaseModel, Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ShadowPriceConfig(BaseModel):
    """Input and output pricing per 1M tokens for shadow cost tracking."""

    input: float = 0.15
    output: float = 0.60


class ModelTierConfig(BaseModel):
    """Configuration schema for an individual LLM model tier."""

    name: str
    model: str
    api_base: str = "http://localhost:11434"
    price_per_1m: float = 0.0
    shadow_price_per_1m: ShadowPriceConfig = Field(
        default_factory=lambda: ShadowPriceConfig(input=0.15, output=0.60)
    )

    @property
    def input_price(self) -> float:
        if isinstance(self.shadow_price_per_1m, ShadowPriceConfig):
            return self.shadow_price_per_1m.input
        if isinstance(self.shadow_price_per_1m, dict):
            return float(self.shadow_price_per_1m.get("input", 0.15))
        return float(self.shadow_price_per_1m)

    @property
    def output_price(self) -> float:
        if isinstance(self.shadow_price_per_1m, ShadowPriceConfig):
            return self.shadow_price_per_1m.output
        if isinstance(self.shadow_price_per_1m, dict):
            return float(self.shadow_price_per_1m.get("output", 0.60))
        return float(self.shadow_price_per_1m)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # General
    PROJECT_NAME: str = "semcache"
    ENVIRONMENT: Literal["development", "staging", "production", "test"] = "development"
    LOG_LEVEL: str = "INFO"
    DEBUG: bool = False

    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Database (Postgres with pgvector)
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "semcache_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str | None = None

    # Ollama Provider & LiteLLM
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_REQUEST_TIMEOUT: float = 30.0
    OLLAMA_RETRY_COUNT: int = 1
    TIERS_CONFIG_PATH: str = "config/tiers.yaml"

    # Embeddings & Vector Search
    EMBEDDING_MODEL: str = "all-minilm"
    EMBEDDING_DIM: int = 384
    SIMILARITY_THRESHOLD: float = 0.88

    # Caching & Redis
    EXACT_CACHE_ENABLED: bool = True
    SEMANTIC_CACHE_ENABLED: bool = True
    SEMANTIC_INFO_ONLY: bool = True
    ANTONYMS_CONFIG_PATH: str = "config/antonyms.yaml"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_VERSION: str = "v1"
    REDIS_URL: str = "redis://localhost:6379/0"

    # Routing Budget & Rules
    MAX_QUERY_BUDGET: float = 0.05
    ROUTING_CONFIG_PATH: str = "config/routing.yaml"

    @computed_field
    def async_database_url(self) -> str:
        if self.DATABASE_URL:
            if self.DATABASE_URL.startswith("postgresql://"):
                return self.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
            return self.DATABASE_URL
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    def load_tiers(self, path: str | None = None) -> list[ModelTierConfig]:
        """Loads and parses the ordered N-tier list from tiers.yaml."""
        target_path = Path(path or self.TIERS_CONFIG_PATH)
        if not target_path.exists():
            return [
                ModelTierConfig(
                    name="small",
                    model="ollama_chat/llama3.2:3b",
                    api_base=self.OLLAMA_BASE_URL,
                    price_per_1m=0.0,
                    shadow_price_per_1m=ShadowPriceConfig(input=0.15, output=0.60),
                ),
                ModelTierConfig(
                    name="large",
                    model="ollama_chat/llama3.1:8b",
                    api_base=self.OLLAMA_BASE_URL,
                    price_per_1m=0.0,
                    shadow_price_per_1m=ShadowPriceConfig(input=2.00, output=8.00),
                ),
            ]

        with open(target_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        raw_tiers = data.get("tiers", [])
        loaded_tiers: list[ModelTierConfig] = []
        for raw in raw_tiers:
            api_base = raw.get("api_base") or self.OLLAMA_BASE_URL
            raw_shadow: Any = raw.get("shadow_price_per_1m", {})

            if isinstance(raw_shadow, dict):
                shadow_cfg = ShadowPriceConfig(
                    input=float(raw_shadow.get("input", 0.15)),
                    output=float(raw_shadow.get("output", 0.60)),
                )
            elif isinstance(raw_shadow, (int, float)):
                shadow_cfg = ShadowPriceConfig(
                    input=float(raw_shadow),
                    output=float(raw_shadow) * 4.0,
                )
            else:
                shadow_cfg = ShadowPriceConfig(input=0.15, output=0.60)

            tier = ModelTierConfig(
                name=raw["name"],
                model=raw["model"],
                api_base=api_base,
                price_per_1m=float(raw.get("price_per_1m", 0.0)),
                shadow_price_per_1m=shadow_cfg,
            )
            loaded_tiers.append(tier)
        return loaded_tiers

    @property
    def tiers(self) -> list[ModelTierConfig]:
        return self.load_tiers()

    def get_tier(self, name: str) -> ModelTierConfig:
        """Finds a tier by name, defaulting to the lowest tier if not found."""
        current_tiers = self.tiers
        for tier in current_tiers:
            if tier.name == name:
                return tier
        return current_tiers[0]


settings = Settings()
