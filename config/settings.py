import os
import logging
from pydantic_settings import BaseSettings
from typing import List, Optional
from functools import lru_cache
from pydantic import field_validator, ConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    Hardcoded defaults are REMOVED for secrets — they MUST come from env/.env.
    """

    # database settings — MUST be set via .env or environment
    DATABASE_URL: str = ""

    # Application settings
    APP_NAME: str = "Vector Database API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # Index settings - Optimized for 10K vector dataset (best configuration)
    DEFAULT_M: int = 32
    DEFAULT_M0: int = 64
    DEFAULT_EF_CONSTRUCTION: int = 300
    DEFAULT_EF_SEARCH: int = 50
    DEFAULT_DISTANCE_METRIC: str = "cosine"
    DEFAULT_N_CLUSTERS: int = 100
    DEFAULT_N_PROBES: int = 10

    # pgvector toggle
    USE_PGVECTOR: bool = False

    # Text embedding
    DEFAULT_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    DEFAULT_TEXT_DIMENSION: int = 384
    EMBEDDING_DEVICE: str = "cpu"

    # Image embedding (CLIP ViT-B/32, 512-dim, lazy-loaded via sentence-transformers)
    DEFAULT_IMAGE_MODEL: str = "clip-ViT-B-32"
    DEFAULT_IMAGE_DIMENSION: int = 512

    # Audio embedding (CPU-friendly librosa MFCC mean-pool; see embedding_service docstring)
    DEFAULT_AUDIO_MODEL: str = "librosa-mfcc-128"
    DEFAULT_AUDIO_DIMENSION: int = 128
    AUDIO_SAMPLE_RATE: int = 22050
    AUDIO_MAX_DURATION_SEC: float = 30.0

    MEDIA_STORAGE_PATH: str = "media_storage"
    STORAGE_PROVIDER: str = "local"

    # Security settings — no hardcoded defaults!
    API_KEY: str = ""
    ALLOWED_HOSTS: List[str] = ["*"]
    CORS_ORIGINS: str = ""

    # Rate limiting — use "redis" backend for shared rate limit across replicas
    RATE_LIMIT_REQUESTS: int = 100
    RATE_LIMIT_TIME: int = 60
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_BACKEND: str = "memory"

    # Redis Cache and Celery Settings
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"
    celery_result_backend: str = "redis://localhost:6379/2"
    celery_task_always_eager: bool = False

    # Request limits
    MAX_REQUEST_SIZE_MB: int = 10

    # LLM Provider API Keys (read via os.getenv in services, declared here to allow .env loading)
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    TOGETHER_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    HF_API_KEY: str = ""

    # Search API Keys
    TAVILY_API_KEY: str = ""
    EXA_API_KEY: str = ""
    SERPAPI_KEY: str = ""
    SERPER_API_KEY: str = ""
    BRAVE_SEARCH_API_KEY: str = ""

    # Local LLM
    OLLAMA_BASE_URL: str = ""

    @field_validator("API_KEY", "DATABASE_URL", mode="before")
    @classmethod
    def validate_required_secrets(cls, v):
        """Warn if critical secrets are using empty/default values in production."""
        if not v or v == "your-api-key-here":
            if not os.environ.get("PYTEST_CURRENT_TEST"):
                logger.warning(
                    "⚠️  %s is not set! Set it via .env or environment variable. "
                    "Using empty string — this will likely cause errors.",
                    cls.__name__,
                )
        return v

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


@lru_cache()
def get_settings() -> Settings:
    """Get application settings with caching."""
    return Settings()

