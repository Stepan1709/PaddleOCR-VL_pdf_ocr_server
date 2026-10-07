"""
Сервер для OCR-обработки PDF через модель PaddlePaddle/PaddleOCR-VL (vLLM).

Принимает PDF по API, постранично рендерит его в изображения, отправляет каждую
страницу в vLLM и возвращает текст с маркерами страниц.
"""
import asyncio
import base64
import logging
import sys
from contextlib import asynccontextmanager
from typing import Optional

import aiohttp
import fitz  # PyMuPDF
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from config import (
    HOST, PORT, VLLM_URL, VLLM_API_KEY, MODEL_NAME, PDF_RENDER_DPI,
    VLLM_REQUEST_TIMEOUT, VLLM_MAX_RETRIES, VLLM_REQUEST_DELAY, LOG_LEVEL, LOG_FILE,
)

VERSION = "3.1.0"
MIN_TEXT_LENGTH = 5  # ответ короче считается нераспознанной страницей

_handlers = [logging.StreamHandler(sys.stdout)]
if LOG_FILE:
    _handlers.append(logging.FileHandler(LOG_FILE, encoding="utf-8"))
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=_handlers,
)
logger = logging.getLogger(__name__)

session: Optional[aiohttp.ClientSession] = None


def vllm_headers() -> dict:
    headers = {"Content-Type": "application/json"}
    if VLLM_API_KEY:
        headers["Authorization"] = f"Bearer {VLLM_API_KEY}"
    return headers


@asynccontextmanager
async def lifespan(app: FastAPI):
    global session
    session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=VLLM_REQUEST_TIMEOUT, connect=30))
    logger.info(f"Сервер запущен на http://{HOST}:{PORT}")
    logger.info(f"vLLM: {VLLM_URL}, модель: {MODEL_NAME}")
    yield
    await session.close()
    logger.info("Сервер остановлен")


app = FastAPI(
    title="PDF OCR Server",
    description="Сервер для OCR-обработки PDF с помощью PaddlePaddle/PaddleOCR-VL",
    version=VERSION,
    lifespan=lifespan,
)


def page_marker(page_num: int, body: str) -> str:
    return f"\nСТРАНИЦА {page_num}\n{body}\n"


def render_page(doc: fitz.Document, page_index: int) -> bytes:
    """Рендерит страницу PDF (индекс с нуля) в PNG."""
    pixmap = doc[page_index].get_pixmap(dpi=PDF_RENDER_DPI)
    return pixmap.tobytes("png")


async def ocr_page(image_png: bytes, page_num: int) -> str:
    """Отправляет изображение страницы в vLLM и возвращает распознанный текст с маркером страницы."""
    image_url = "data:image/png;base64," + base64.b64encode(image_png).decode("utf-8")
    payload = {
        "model": MODEL_NAME,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": "OCR:"},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "max_tokens": 4096,
        "temperature": 0.1,
        "top_p": 0.95,
    }

    for attempt in range(1, VLLM_MAX_RETRIES + 1):
        try:
            await asyncio.sleep(VLLM_REQUEST_DELAY)
            async with session.post(f"{VLLM_URL}/v1/chat/completions",
                                    json=payload, headers=vllm_headers()) as response:
                if response.status != 200:
                    raise RuntimeError(f"vLLM вернул ошибку {response.status}: {await response.text()}")
                result = await response.json()

            text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            if len(text) < MIN_TEXT_LENGTH:
                logger.warning(f"Страница {page_num}: пустой или слишком короткий текст")
                return page_marker(page_num, "[Пустая страница или не удалось распознать текст]")

            logger.info(f"Страница {page_num}: распознано {len(text)} символов (попытка {attempt})")
            return page_marker(page_num, text)

        except (aiohttp.ClientError, asyncio.TimeoutError, RuntimeError) as e:
            logger.warning(f"Страница {page_num}: ошибка (попытка {attempt}/{VLLM_MAX_RETRIES}): {e!r}")
            if attempt == VLLM_MAX_RETRIES:
                logger.error(f"Страница {page_num}: все попытки не удались")
                return page_marker(page_num, f"[Ошибка OCR: {e}]")
            await asyncio.sleep(3 + 2 ** (attempt - 1))


async def process_pdf(filename: str, pdf_bytes: bytes) -> str:
    """Рендерит страницы PDF, распознаёт их последовательно и собирает результат."""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Не удалось открыть PDF: {e}") from e

    with doc:
        total_pages = doc.page_count
        logger.info(f"Получен файл: {filename}, страниц: {total_pages}")

        parts = []
        for page_index in range(total_pages):
            page_num = page_index + 1
            try:
                image = await asyncio.to_thread(render_page, doc, page_index)
                parts.append(await ocr_page(image, page_num))
            except Exception as e:
                logger.error(f"Страница {page_num}: ошибка обработки: {e!r}")
                parts.append(page_marker(page_num, f"[Ошибка: {e}]"))

    logger.info(f'Файл "{filename}" обработан, страниц: {total_pages}')
    return "".join(parts)


@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_pdf(file: UploadFile = File(...)) -> str:
    """
    Принимает PDF, возвращает текст с нумерацией страниц.

    Пример: curl -X POST -F "file=@document.pdf" http://localhost:9000/ocr
    """
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате PDF")

    pdf_bytes = await file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Файл пуст")

    try:
        return await process_pdf(file.filename, pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception(f"Ошибка при обработке {file.filename}")
        raise HTTPException(status_code=500, detail=f"Ошибка при обработке файла: {e}")


@app.get("/live")
async def liveness():
    """Проверка, что процесс жив (без обращения к vLLM)."""
    return {"status": "alive"}


@app.get("/health")
async def health_check():
    """Проверка доступности vLLM и нужной модели."""
    headers = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}
    try:
        async with session.get(f"{VLLM_URL}/v1/models", headers=headers) as response:
            if response.status != 200:
                return {"status": "degraded", "vllm": f"error_{response.status}"}
            models = await response.json()
            model_available = any(MODEL_NAME in m.get("id", "") for m in models.get("data", []))
            return {"status": "healthy", "vllm": "connected", "model_available": model_available}
    except Exception as e:
        return {"status": "degraded", "vllm": "disconnected", "error": str(e)}


@app.get("/")
async def root():
    return {
        "service": "PDF OCR Server",
        "version": VERSION,
        "endpoints": {
            "ocr": "POST /ocr - Отправить PDF файл для OCR",
            "live": "GET /live - Liveness-проба",
            "health": "GET /health - Состояние сервера и vLLM",
        },
        "model": MODEL_NAME,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT, log_level=LOG_LEVEL.lower(), access_log=True)
