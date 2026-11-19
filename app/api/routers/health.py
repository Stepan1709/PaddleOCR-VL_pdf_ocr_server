"""Служебные эндпоинты: информация о сервисе, liveness и health."""

import logging

import httpx
from fastapi import APIRouter

from app.api.deps import SettingsDep, VllmClientDep
from app.schemas.health import HealthResponse, LiveResponse

logger = logging.getLogger(__name__)

router = APIRouter(tags=["service"])


@router.get("/")
async def root(settings: SettingsDep) -> dict[str, object]:
    return {
        "service": "PDF OCR Server",
        "version": "4.0.0",
        "endpoints": {
            "ocr": "POST /ocr - Отправить PDF файл для OCR",
            "live": "GET /live - Liveness-проба",
            "health": "GET /health - Состояние сервера и vLLM",
        },
        "model": settings.model_name,
    }


@router.get("/live", response_model=LiveResponse)
async def liveness() -> LiveResponse:
    """Проверка, что процесс жив (без обращения к vLLM)."""
    return LiveResponse(status="alive")


@router.get("/health", response_model=HealthResponse)
async def health(vllm: VllmClientDep, settings: SettingsDep) -> HealthResponse:
    """Проверка доступности vLLM и нужной модели."""
    try:
        models = await vllm.list_models()
    except httpx.HTTPStatusError as exc:
        return HealthResponse(status="degraded", vllm=f"error_{exc.response.status_code}")
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("vLLM недоступен: %r", exc)
        return HealthResponse(status="degraded", vllm="disconnected", error=str(exc))
    return HealthResponse(
        status="healthy",
        vllm="connected",
        model_available=any(settings.model_name in m for m in models),
    )
