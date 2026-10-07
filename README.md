# PaddleOCR-VL_pdf_ocr_server

Сервис OCR для PDF-сканов. Рендерит каждую страницу PDF в изображение, отправляет её в модель
[PaddlePaddle/PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL), запущенную в vLLM (OpenAI-совместимый API),
и возвращает текст с маркерами страниц.

Часть RAG-пайплайна: его вызывает [file_to_text_converter_server](https://github.com/Stepan1709/file_to_text_converter_server)
для PDF без текстового слоя.

```
s3_to_qdrant_synchronizer ──> file_to_text_converter_server ──> PaddleOCR-VL_pdf_ocr_server (этот сервис) ──> vLLM
```

## Как это работает

1. PDF открывается через PyMuPDF, рендерится постранично (по умолчанию 300 DPI).
2. Каждая страница отправляется в `POST {VLLM_URL}/v1/chat/completions` с промптом `OCR:`.
3. При ошибке запрос повторяется (по умолчанию 3 попытки с нарастающей паузой). Если все попытки неудачны,
   вместо текста страницы в ответ подставляется строка `[Ошибка OCR: ...]`, обработка остальных страниц продолжается.
4. Страницы обрабатываются последовательно, чтобы не перегружать vLLM; перед каждым запросом выдерживается пауза
   `VLLM_REQUEST_DELAY`.

## Конфигурация

Все параметры — переменные окружения (шаблон — `.env.example`). Файл `.env` с ключами в git не попадает.

| Переменная             | По умолчанию               | Описание                                                     |
|------------------------|----------------------------|--------------------------------------------------------------|
| `VLLM_URL`             | `http://localhost:8400`    | URL vLLM-сервера                                             |
| `VLLM_API_KEY`         | пусто                      | API-ключ vLLM (если сервер запущен с `--api-key`)            |
| `MODEL_NAME`           | `PaddlePaddle/PaddleOCR-VL`| Имя модели в vLLM                                            |
| `PDF_RENDER_DPI`       | `300`                      | Разрешение рендера страниц                                   |
| `VLLM_REQUEST_TIMEOUT` | `60`                       | Таймаут одного запроса к vLLM, сек                           |
| `VLLM_MAX_RETRIES`     | `3`                        | Число попыток на страницу                                    |
| `VLLM_REQUEST_DELAY`   | `3`                        | Пауза перед каждым запросом к vLLM, сек                      |
| `HOST` / `PORT`        | `0.0.0.0` / `9000`         | Адрес и порт сервиса (в Docker-образе проброшен `9000`)      |
| `LOG_LEVEL`            | `INFO`                     | Уровень логирования                                          |
| `LOG_FILE`             | пусто                      | Путь к файлу логов; по умолчанию логи только в stdout        |

## Запуск

### Docker Compose

```bash
git clone https://github.com/Stepan1709/PaddleOCR-VL_pdf_ocr_server
cd PaddleOCR-VL_pdf_ocr_server
cp .env.example .env      # укажите VLLM_URL и VLLM_API_KEY
docker compose up -d --build
docker compose logs -f
```

### Docker

```bash
docker build -t paddle-pdf-ocr-server .
docker run -d --name paddle-pdf-ocr-server -p 9000:9000 --env-file .env --restart unless-stopped paddle-pdf-ocr-server
docker logs -f --tail 100 paddle-pdf-ocr-server
```

### Локально

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export VLLM_URL=http://<host>:8400 VLLM_API_KEY=<key>
python pdf_ocr_server.py
```

## API

| Метод  | Путь      | Описание                                                                    |
|--------|-----------|-----------------------------------------------------------------------------|
| `POST` | `/ocr`    | OCR PDF-файла (multipart, поле `file`), ответ — `text/plain`                |
| `GET`  | `/live`   | Liveness-проба (используется в `HEALTHCHECK`, vLLM не проверяет)            |
| `GET`  | `/health` | Доступность vLLM и модели (`healthy` / `degraded`)                          |
| `GET`  | `/`       | Информация о сервисе                                                        |
| `GET`  | `/docs`   | Swagger UI                                                                  |

```bash
curl -X POST "http://localhost:9000/ocr" -F "file=@/path/to/document.pdf"
curl http://localhost:9000/health
```

Формат ответа — страницы с маркерами `СТРАНИЦА N`:

```
СТРАНИЦА 1
Текст первой страницы...

СТРАНИЦА 2
Текст второй страницы...
```

Ошибки: `400` — не PDF, пустой или повреждённый файл; `500` — прочие ошибки обработки.
