from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """App settings, read from environment variables prefixed with RAG_ (or a .env file)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    ollama_base_url: str = "http://localhost:11434"
    default_llm_model: str = "qwen3:8b"
    default_embed_model: str = "nomic-embed-text"
    # JSON list in the environment, e.g. RAG_CORS_ORIGINS='["http://localhost:5173"]'
    cors_origins: list[str] = ["http://localhost:5173"]
    max_upload_mb: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
