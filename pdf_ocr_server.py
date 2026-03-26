#!/usr/bin/env python3
"""
Сервер для обработки PDF через OCR модель PaddlePaddle/PaddleOCR-VL
Принимает файлы по API, разбивает на страницы, отправляет в vLLM,
возвращает текст с нумерацией страниц.
"""

import os
import sys
import io
import base64
import json
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio
import aiohttp
from aiohttp import ClientTimeout, ClientError
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse
import PyPDF2
from pdf2image import convert_from_bytes
import fitz  # PyMuPDF - для более надежной работы с PDF
from tqdm import tqdm
import logging
from contextlib import asynccontextmanager

# Импортируем настройки
from config import HOST, PORT, VLLM_URL, VLLM_API_KEY, MODEL_NAME, TEMP_DIR, LOG_FILE

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Глобальная переменная для сессии aiohttp
session = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global session
    # Запуск: создаем сессию с увеличенными таймаутами
    timeout = ClientTimeout(total=60, connect=30, sock_read=120)
    session = aiohttp.ClientSession(timeout=timeout)
    logger.info(f"🚀 Сервер запущен на http://{HOST}:{PORT}")
    logger.info(f"📡 Подключен к vLLM: {VLLM_URL}")
    logger.info(f"🤖 Модель: {MODEL_NAME}")

    yield

    # Завершение: закрываем сессию
    if session:
        await session.close()
    logger.info("👋 Сервер остановлен")


# Создаем приложение FastAPI
app = FastAPI(
    title="PDF OCR Server",
    description="Сервер для OCR обработки PDF с помощью PaddlePaddle/PaddleOCR-VL",
    version="3.0.0",
    lifespan=lifespan
)


def log_error(filename: str, error: Exception):
    """Запись ошибки в лог-файл"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    error_msg = f"{timestamp} | Файл: {filename} | Ошибка: {str(error)}\n"

    # Записываем в файл
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(error_msg)

    # Также выводим в консоль
    logger.error(f"❌ Ошибка при обработке {filename}: {str(error)}")


def get_pdf_page_count(pdf_bytes: bytes) -> int:
    """Получение количества страниц в PDF"""
    try:
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        return len(pdf_reader.pages)
    except Exception as e:
        logger.warning(f"PyPDF2 не смог прочитать PDF, пробуем PyMuPDF: {e}")
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        return pdf_document.page_count


async def process_page_with_vllm(page_image_bytes: bytes, page_num: int, retry_count: int = 3) -> str:
    """
    Отправка изображения страницы в vLLM для OCR
    Возвращает распознанный текст
    """
    for attempt in range(retry_count):
        try:
            await asyncio.sleep(3)
            # Кодируем изображение в base64
            image_base64 = base64.b64encode(page_image_bytes).decode('utf-8')
            image_url = f"data:image/png;base64,{image_base64}"

            # Формируем запрос к vLLM (OpenAI-compatible API)
            payload = {
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "OCR:"
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_url
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 4096,
                "temperature": 0.1,
                "top_p": 0.95
            }

            # Заголовки для аутентификации
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {VLLM_API_KEY}"
            }

            # Отправляем запрос с увеличенным таймаутом
            async with session.post(f"{VLLM_URL}/v1/chat/completions",
                                    json=payload,
                                    headers=headers) as response:

                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"vLLM вернул ошибку {response.status}: {error_text}")

                result = await response.json()

                # Извлекаем текст из ответа
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

                # Если текст всё еще пустой или содержит только мусор
                if not text or len(text) < 5:
                    logger.warning(f"Страница {page_num}: получен пустой или слишком короткий текст")
                    return f"\nСТРАНИЦА {page_num}\n[Пустая страница или не удалось распознать текст]\n"

                logger.info(f"Страница {page_num}: распознано {len(text)} символов (попытка {attempt + 1})")

                # Добавляем строку с номером страницы
                return f"\nСТРАНИЦА {page_num}\n{text}\n"

        except (ClientError, asyncio.TimeoutError, Exception) as e:
            logger.warning(f"Ошибка при обработке страницы {page_num} (попытка {attempt + 1}/{retry_count}): {e}")
            if attempt < retry_count - 1:
                # Ждем перед повторной попыткой (экспоненциальная задержка)
                wait_time = 3 + (2 ** attempt)
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"Страница {page_num}: все попытки ({retry_count}) не удались")
                return f"\nСТРАНИЦА {page_num}\n[Ошибка OCR: {str(e)}]\n"


async def convert_pdf_page_to_image(pdf_bytes: bytes, page_num: int) -> bytes:
    """
    Конвертирует конкретную страницу PDF в изображение
    Использует pdf2image для конвертации
    """
    try:
        # Используем pdf2image для конвертации страницы
        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_num,
            last_page=page_num,
            dpi=300,  # Высокое разрешение для лучшего распознавания
            fmt='png'
        )

        if not images:
            raise Exception(f"Не удалось конвертировать страницу {page_num}")

        # Конвертируем изображение в байты
        img_byte_arr = io.BytesIO()
        images[0].save(img_byte_arr, format='PNG', optimize=False)
        img_byte_arr.seek(0)

        return img_byte_arr.getvalue()

    except Exception as e:
        logger.error(f"Ошибка конвертации страницы {page_num}: {e}")
        raise


async def process_pdf(filename: str, pdf_bytes: bytes) -> str:
    """
    Основная функция обработки PDF
    Разбивает на страницы, отправляет в vLLM, собирает результат
    """
    # Получаем количество страниц
    total_pages = get_pdf_page_count(pdf_bytes)
    logger.info(f"📄 Получен файл: {filename}")
    logger.info(f"📑 Количество страниц в файле: {total_pages}")

    all_text = []

    # Создаем прогресс-бар для обработки страниц
    with tqdm(total=total_pages, desc="Обработка страниц", unit="стр") as pbar:
        for page_num in range(1, total_pages + 1):
            try:
                # Конвертируем страницу в изображение
                page_image = await convert_pdf_page_to_image(pdf_bytes, page_num)

                # Отправляем в vLLM с повторными попытками
                page_text = await process_page_with_vllm(page_image, page_num, retry_count=3)
                all_text.append(page_text)

                # Обновляем прогресс-бар
                pbar.update(1)
                pbar.set_postfix({"Текущая страница": page_num})

                # Увеличиваем задержку между запросами для стабильности
                await asyncio.sleep(0.5)  # Увеличил с 0.1 до 0.5 секунд

            except Exception as e:
                error_msg = f"Ошибка при обработке страницы {page_num}: {str(e)}"
                logger.error(error_msg)
                all_text.append(f"\nСТРАНИЦА {page_num}\n[Ошибка: {str(e)}]\n")
                pbar.update(1)
                # При ошибке делаем дополнительную паузу
                await asyncio.sleep(1)
                continue

    # Собираем весь текст
    full_text = "".join(all_text)

    logger.info(f"✅ Файл \"{filename}\" успешно обработан. Всего страниц: {total_pages}")

    return full_text


@app.post("/ocr", response_class=PlainTextResponse)
async def ocr_pdf(file: UploadFile = File(...)) -> str:
    """
    Основной эндпоинт для обработки PDF

    Принимает PDF файл, возвращает текст с нумерацией страниц

    Пример использования:
    curl -X POST -F "file=@document.pdf" http://localhost:9000/ocr
    """
    # Проверяем тип файла
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Файл должен быть в формате PDF"
        )

    # Читаем содержимое файла
    try:
        pdf_bytes = await file.read()

        # Проверяем, что файл не пустой
        if len(pdf_bytes) == 0:
            raise HTTPException(
                status_code=400,
                detail="Файл пуст"
            )

        # Обрабатываем PDF
        result_text = await process_pdf(file.filename, pdf_bytes)

        return result_text

    except Exception as e:
        # Логируем ошибку
        log_error(file.filename, e)

        # Возвращаем ошибку клиенту
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при обработке файла: {str(e)}"
        )
    finally:
        # Принудительно освобождаем память
        if 'pdf_bytes' in locals():
            del pdf_bytes


@app.get("/health")
async def health_check():
    """Проверка работоспособности сервера"""
    # Проверяем доступность vLLM
    try:
        headers = {"Authorization": f"Bearer {VLLM_API_KEY}"}
        async with session.get(f"{VLLM_URL}/v1/models", headers=headers) as response:
            if response.status == 200:
                models = await response.json()
                model_available = any(MODEL_NAME in m.get("id", "") for m in models.get("data", []))
                return {
                    "status": "healthy",
                    "vllm": "connected",
                    "model_available": model_available
                }
            else:
                return {
                    "status": "degraded",
                    "vllm": f"error_{response.status}"
                }
    except Exception as e:
        return {
            "status": "degraded",
            "vllm": "disconnected",
            "error": str(e)
        }


@app.get("/")
async def root():
    """Корневой эндпоинт с информацией о сервере"""
    return {
        "service": "PDF OCR Server",
        "version": "3.0.0",
        "endpoints": {
            "ocr": "POST /ocr - Отправить PDF файл для OCR",
            "health": "GET /health - Проверка состояния сервера"
        },
        "model": MODEL_NAME,
        "vllm_url": VLLM_URL
    }


if __name__ == "__main__":
    import uvicorn

    # Запускаем сервер
    uvicorn.run(
        app,
        host=HOST,
        port=PORT,
        log_level="info",
        access_log=True
    )