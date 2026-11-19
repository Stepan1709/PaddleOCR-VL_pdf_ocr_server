from collections.abc import AsyncIterator, Iterator

import fitz
import httpx
import pytest
import respx
from fastapi import FastAPI

from app.clients.vllm import VllmClient
from app.core.config import Settings, get_settings
from app.main import create_app

BASE_URL = "http://vllm.test"


def make_pdf(pages: int = 1) -> bytes:
    doc = fitz.open()
    for i in range(pages):
        doc.new_page().insert_text((72, 72), f"page {i + 1}")
    data = bytes(doc.tobytes())
    doc.close()
    return data


@pytest.fixture
def mock_vllm() -> Iterator[respx.MockRouter]:
    with respx.mock(base_url=BASE_URL, assert_all_called=False) as router:
        yield router


@pytest.fixture(autouse=True)
def no_retry_pause(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.services.ocr.retry_delay", lambda attempt: 0.0)


@pytest.fixture
async def app(mock_vllm: respx.MockRouter) -> AsyncIterator[FastAPI]:
    application = create_app()
    application.dependency_overrides[get_settings] = lambda: Settings(
        vllm_url=BASE_URL, vllm_request_delay=0, vllm_max_retries=2, pdf_render_dpi=72, max_upload_size_mb=1
    )
    async with httpx.AsyncClient(timeout=5) as http_client:
        application.state.vllm_client = VllmClient(http_client, BASE_URL, "paddle-model")
        yield application


@pytest.fixture
async def client(app: FastAPI) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c
