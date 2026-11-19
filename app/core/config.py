"""Конфигурация сервиса. Все параметры задаются переменными окружения или файлом .env."""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    host: str = "0.0.0.0"
    port: int = Field(9000, ge=1, le=65535)

    vllm_url: str = "http://localhost:8400"
    vllm_api_key: str = ""
    model_name: str = "PaddlePaddle/PaddleOCR-VL"

    pdf_render_dpi: int = Field(300, ge=72)
    vllm_request_timeout: float = Field(60.0, gt=0)
    vllm_max_retries: int = Field(3, ge=1)
    vllm_request_delay: float = Field(3.0, ge=0)
    max_upload_size_mb: int = Field(100, ge=1)

    log_level: str = "INFO"
    log_file: str = ""

    @property
    def vllm_base_url(self) -> str:
        return self.vllm_url.rstrip("/")

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()
