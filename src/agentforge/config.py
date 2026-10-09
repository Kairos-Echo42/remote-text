from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AGENTFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    env: Literal["development", "test", "production"] = "development"
    secret_key: str = "change-me-in-production-at-least-32-bytes"
    master_key: str = "ZGV2ZWxvcG1lbnQtbWFzdGVyLWtleS0zMi1ieXRlcyE="
    database_url: str = "postgresql+asyncpg://agentforge:agentforge@localhost:5432/agentforge"
    redis_url: str = "redis://localhost:6379/0"
    gateway_url: str = "http://localhost:8000"
    artifact_root: Path = Path("./data/artifacts")
    workspace_root: Path = Path("./data/workspaces")
    host_workspace_root: str | None = None
    training_root: Path = Path("./data/training")
    host_training_root: str | None = None
    trainer_image: str = "agentforge-trainer:latest"
    trainer_gpu_available: bool = True
    git_commit: str | None = None
    training_gpu_concurrency: int = Field(default=1, ge=1, le=8)
    training_cpu_concurrency: int = Field(default=2, ge=1, le=32)
    training_max_dataset_bytes: int = Field(default=2 * 1024**3, ge=1)
    web_dir: Path = Path("./web")
    example_dir: Path = Path("./examples")
    default_model: str = "fake"
    docker_exe: str | None = None
    docker_image: str = "python:3.12-slim"
    docker_network_enabled: bool = False
    run_max_concurrency: int = Field(default=4, ge=1, le=64)
    worker_claim_idle_ms: int = Field(default=60_000, ge=5_000)
    worker_lease_seconds: int = Field(default=120, ge=1)
    sse_heartbeat_seconds: int = Field(default=15, ge=5)
    expose_docs: bool = True
    log_level: str = "INFO"
    admin_email: str = "admin@example.com"
    admin_password: str = "change-me-now"
    openai_api_key: str | None = None
    openai_base_url: str = "https://api.openai.com/v1"
    deepseek_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DEEPSEEK_API_KEY", "AGENTFORGE_DEEPSEEK_API_KEY"),
    )
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_chat_model: str = "deepseek-chat"
    dashscope_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices("DASHSCOPE_API_KEY", "AGENTFORGE_DASHSCOPE_API_KEY"),
    )
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    dashscope_embedding_model: str = "text-embedding-v4"
    search_provider: str = "fixture"
    tavily_api_key: str | None = None

    @property
    def is_test(self) -> bool:
        return self.env == "test"

    @property
    def is_production(self) -> bool:
        return self.env == "production"

    def ensure_directories(self) -> None:
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self.workspace_root.mkdir(parents=True, exist_ok=True)
        self.training_root.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings


def reset_settings_cache() -> None:
    get_settings.cache_clear()
