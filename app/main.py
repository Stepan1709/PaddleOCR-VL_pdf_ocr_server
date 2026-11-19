"""Точка входа: сборка FastAPI-приложения."""

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI

from app.api.routers import health, ocr
from app.clients.vllm import VllmClient
from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import setup_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    timeout = httpx.Timeout(settings.vllm_request_timeout, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as http_client:
        app.state.vllm_client = VllmClient(
            http_client, settings.vllm_base_url, settings.model_name, settings.vllm_api_key
        )
        logger.info("Сервис запущен: vLLM %s, модель %s", settings.vllm_base_url, settings.model_name)
        yield
    logger.info("Сервис остановлен")


def create_app() -> FastAPI:
    settings = get_settings()
    setup_logging(settings.log_level, settings.log_file)
    app = FastAPI(
        title="PDF OCR Server",
        description="Сервис OCR для PDF с помощью PaddlePaddle/PaddleOCR-VL",
        version="4.0.0",
        lifespan=lifespan,
    )
    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(ocr.router)
    return app


app = create_app()
