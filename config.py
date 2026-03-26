import os

HOST = "0.0.0.0"
PORT = 9000

try:
    from secrets import VLLM_URL, VLLM_API_KEY
except ImportError:
    VLLM_URL = "http://192.168.28.244:8400"
    VLLM_API_KEY = " " #write you API key here
    print(f"⚠️  Файл secrets.py не найден! Использую значения по умолчанию")

MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"

TEMP_DIR = "/tmp/pdf_ocr_server"

LOG_FILE = "log.txt"

os.makedirs(TEMP_DIR, exist_ok=True)