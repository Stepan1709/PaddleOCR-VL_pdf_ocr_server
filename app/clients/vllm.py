"""Клиент OpenAI-совместимого vLLM-сервера с моделью PaddleOCR-VL."""

import base64
import logging
from typing import Any

import httpx

from app.core.exceptions import VllmError

logger = logging.getLogger(__name__)

OCR_PROMPT = "OCR:"


class VllmClient:
    def __init__(self, http_client: httpx.AsyncClient, base_url: str, model: str, api_key: str = "") -> None:
        self._http = http_client
        self._base_url = base_url
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    async def recognize(self, image_png: bytes) -> str:
        """Одна попытка распознавания изображения. При любой ошибке бросает VllmError."""
        image_url = "data:image/png;base64," + base64.b64encode(image_png).decode("ascii")
        payload = {
            "model": self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": OCR_PROMPT},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            "max_tokens": 4096,
            "temperature": 0.1,
            "top_p": 0.95,
        }
        try:
            response = await self._http.post(
                f"{self._base_url}/v1/chat/completions", json=payload, headers=self._headers
            )
            response.raise_for_status()
            content = response.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            raise VllmError(
                f"vLLM вернул ошибку {exc.response.status_code}: {exc.response.text[:200]}"
            ) from exc
        except httpx.HTTPError as exc:
            raise VllmError(f"ошибка запроса к vLLM: {exc!r}") from exc
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            raise VllmError(f"некорректный ответ vLLM: {exc!r}") from exc
        return str(content or "").strip()

    async def list_models(self) -> list[str]:
        """Возвращает идентификаторы моделей, доступных в vLLM."""
        response = await self._http.get(f"{self._base_url}/v1/models", headers=self._headers, timeout=10.0)
        response.raise_for_status()
        data: list[dict[str, Any]] = response.json().get("data", [])
        return [str(m.get("id", "")) for m in data]
