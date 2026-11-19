"""Эндпоинт OCR для PDF."""

from typing import Annotated

from fastapi import APIRouter, File, UploadFile
from fastapi.responses import PlainTextResponse

from app.api.deps import OcrServiceDep, SettingsDep
from app.core.exceptions import EmptyFileError, FileTooLargeError, UnsupportedFileTypeError

router = APIRouter(tags=["ocr"])


@router.post("/ocr", response_class=PlainTextResponse)
async def ocr_pdf(
    service: OcrServiceDep, settings: SettingsDep, file: Annotated[UploadFile, File()]
) -> PlainTextResponse:
    """
    Принимает PDF, возвращает текст с маркерами страниц.

    Пример: curl -X POST -F "file=@document.pdf" http://localhost:9000/ocr
    """
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise UnsupportedFileTypeError()

    limit = settings.max_upload_size_bytes
    pdf_bytes = await file.read(limit + 1)
    if not pdf_bytes:
        raise EmptyFileError()
    if len(pdf_bytes) > limit:
        raise FileTooLargeError(f"Файл больше {settings.max_upload_size_mb} МБ")

    return PlainTextResponse(await service.process_pdf(filename, pdf_bytes))
