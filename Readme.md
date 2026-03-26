
# PDF OCR Server with PaddleOCR-VL

### Сервер для OCR обработки PDF файлов с использованием модели PaddlePaddle/PaddleOCR-VL через vLLM на удаленном сервере.

## Настройка окружения

1. Клонируйте репозиторий:
```bash
git clone https://github.com/Stepan1709/PaddleOCR-VL_pdf_ocr_server
```

2. Настройте параметры подключения к vLLM в файле `secrets.py`:
```python
VLLM_URL = "http://ip:port"
VLLM_API_KEY = " "
```

## Сборка Docker образа
1. Соберите образ в папке склонированного репозиторя:
```bash
docker build -t paddle-pdf-ocr-server:latest .
```

## Запуск контейнера

```bash
docker run -d --name paddle-pdf-ocr-server -p 9000:9000 --restart unless-stopped paddle-pdf-ocr-server:latest
```

## Получение логов

```bash
docker logs -f --tail 100 paddle-pdf-ocr-server
```

## Использование API

### Отправить PDF файл на OCR

```bash
curl -X POST "http://localhost:9000/ocr" -F "file=@/path/to/document.pdf"
```


### Проверка статуса сервера

```bash
curl -X GET "http://localhost:9000/health"
```

### Получить информацию о сервисе

```bash
curl -X GET "http://localhost:9000/"
```

## Формат ответа

Сервер возвращает текст в формате plain text, где каждая страница начинается с маркера `СТРАНИЦА X`:

```
СТРАНИЦА 1
Текст первой страницы...

СТРАНИЦА 2
Текст второй страницы...
```

## Параметры конфигурации

В файле `config.py` можно настроить:
- `HOST` - адрес для запуска сервера
- `PORT` - порт для запуска сервера
- `MODEL_NAME` - имя модели в vLLM
- `VLLM_URL` - URL vLLM сервера
- `VLLM_API_KEY` - API ключ для vLLM

## Логирование

Логи сохраняются в файл `log.txt` и выводятся в консоль.
