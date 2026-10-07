"""Конфигурация сервиса. Все параметры задаются переменными окружения."""
import os

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "9000"))

VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8400").rstrip("/")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "").strip()
MODEL_NAME = os.getenv("MODEL_NAME", "PaddlePaddle/PaddleOCR-VL")

PDF_RENDER_DPI = int(os.getenv("PDF_RENDER_DPI", "300"))
VLLM_REQUEST_TIMEOUT = float(os.getenv("VLLM_REQUEST_TIMEOUT", "60"))
VLLM_MAX_RETRIES = int(os.getenv("VLLM_MAX_RETRIES", "3"))
VLLM_REQUEST_DELAY = float(os.getenv("VLLM_REQUEST_DELAY", "3"))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_FILE = os.getenv("LOG_FILE", "").strip()
