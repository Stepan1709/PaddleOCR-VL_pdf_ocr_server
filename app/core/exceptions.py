"""Доменные исключения и их преобразование в HTTP-ответы."""

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


class AppError(Exception):
    """Базовое исключение сервиса."""

    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    detail = "Внутренняя ошибка сервиса"

    def __init__(self, detail: str | None = None) -> None:
        self.detail = detail or self.detail
        super().__init__(self.detail)


class UnsupportedFileTypeError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Файл должен быть в формате PDF"


class EmptyFileError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    detail = "Файл пуст"


class InvalidPdfError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST


class FileTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE


class VllmError(Exception):
    """Ошибка обращения к vLLM. Наружу не уходит: сервис подставляет текст ошибки вместо страницы."""


async def app_error_handler(_: Request, exc: Exception) -> JSONResponse:
    if not isinstance(exc, AppError):
        raise exc
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, app_error_handler)
