"""Схемы служебных эндпоинтов."""

from pydantic import BaseModel


class LiveResponse(BaseModel):
    status: str


class HealthResponse(BaseModel):
    status: str
    vllm: str
    model_available: bool | None = None
    error: str | None = None
