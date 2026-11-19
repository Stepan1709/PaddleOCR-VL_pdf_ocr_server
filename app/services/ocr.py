"""Сценарий OCR: постраничный рендер PDF и распознавание через vLLM."""

import asyncio
import logging

from app.clients.vllm import VllmClient
from app.core.exceptions import VllmError
from app.services.pdf import open_pdf, render_page

logger = logging.getLogger(__name__)

MIN_TEXT_LENGTH = 5  # ответ короче считается нераспознанной страницей


def page_marker(page_num: int, body: str) -> str:
    return f"\nСТРАНИЦА {page_num}\n{body}\n"


def retry_delay(attempt: int) -> float:
    """Пауза перед повторной попыткой: 4, 5, 7, 11... секунд."""
    return float(3 + 2 ** (attempt - 1))


class OcrService:
    def __init__(self, vllm: VllmClient, dpi: int, max_retries: int, request_delay: float) -> None:
        self._vllm = vllm
        self._dpi = dpi
        self._max_retries = max_retries
        self._request_delay = request_delay

    async def process_pdf(self, filename: str, pdf_bytes: bytes) -> str:
        """Распознаёт страницы последовательно, чтобы не перегружать vLLM."""
        doc = open_pdf(pdf_bytes)
        with doc:
            total = doc.page_count
            logger.info("Получен файл %s, страниц: %d", filename, total)

            parts: list[str] = []
            for index in range(total):
                page_num = index + 1
                try:
                    image = await asyncio.to_thread(render_page, doc, index, self._dpi)
                    parts.append(await self._ocr_page(image, page_num))
                except Exception as exc:
                    logger.exception("Страница %d: ошибка обработки", page_num)
                    parts.append(page_marker(page_num, f"[Ошибка: {exc}]"))

        logger.info("Файл %s обработан, страниц: %d", filename, total)
        return "".join(parts)

    async def _ocr_page(self, image: bytes, page_num: int) -> str:
        for attempt in range(1, self._max_retries + 1):
            await asyncio.sleep(self._request_delay)
            try:
                text = await self._vllm.recognize(image)
            except VllmError as exc:
                logger.warning(
                    "Страница %d: ошибка (попытка %d/%d): %s", page_num, attempt, self._max_retries, exc
                )
                if attempt == self._max_retries:
                    logger.error("Страница %d: все попытки не удались", page_num)
                    return page_marker(page_num, f"[Ошибка OCR: {exc}]")
                await asyncio.sleep(retry_delay(attempt))
                continue

            if len(text) < MIN_TEXT_LENGTH:
                logger.warning("Страница %d: пустой или слишком короткий текст", page_num)
                return page_marker(page_num, "[Пустая страница или не удалось распознать текст]")
            logger.info("Страница %d: распознано %d символов (попытка %d)", page_num, len(text), attempt)
            return page_marker(page_num, text)
        return page_marker(page_num, "[Ошибка OCR]")  # недостижимо при max_retries >= 1
