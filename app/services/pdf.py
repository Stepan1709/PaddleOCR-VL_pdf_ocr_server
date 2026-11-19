"""Работа с PDF: открытие и рендер страниц в изображения."""

import fitz  # PyMuPDF

from app.core.exceptions import InvalidPdfError


def open_pdf(data: bytes) -> fitz.Document:
    try:
        return fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        raise InvalidPdfError(f"Не удалось открыть PDF: {exc}") from exc


def render_page(doc: fitz.Document, page_index: int, dpi: int) -> bytes:
    """Рендерит страницу (индекс с нуля) в PNG."""
    pixmap = doc[page_index].get_pixmap(dpi=dpi)
    return bytes(pixmap.tobytes("png"))
