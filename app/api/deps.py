"""Зависимости FastAPI: доступ к объектам, созданным в lifespan."""

from typing import Annotated

from fastapi import Depends, Request

from app.clients.vllm import VllmClient
from app.core.config import Settings, get_settings
from app.services.ocr import OcrService


def get_vllm_client(request: Request) -> VllmClient:
    client: VllmClient = request.app.state.vllm_client
    return client


SettingsDep = Annotated[Settings, Depends(get_settings)]
VllmClientDep = Annotated[VllmClient, Depends(get_vllm_client)]


def get_ocr_service(vllm: VllmClientDep, settings: SettingsDep) -> OcrService:
    return OcrService(
        vllm,
        dpi=settings.pdf_render_dpi,
        max_retries=settings.vllm_max_retries,
        request_delay=settings.vllm_request_delay,
    )


OcrServiceDep = Annotated[OcrService, Depends(get_ocr_service)]
