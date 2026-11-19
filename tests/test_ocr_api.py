import httpx
import respx

from tests.conftest import make_pdf


def vllm_reply(text: str) -> dict[str, object]:
    return {"choices": [{"message": {"content": text}}]}


async def post_pdf(client: httpx.AsyncClient, data: bytes, name: str = "doc.pdf") -> httpx.Response:
    return await client.post("/ocr", files={"file": (name, data, "application/pdf")})


async def test_ocr_returns_text_with_page_markers(
    client: httpx.AsyncClient, mock_vllm: respx.MockRouter
) -> None:
    mock_vllm.post("/v1/chat/completions").respond(json=vllm_reply("распознанный текст"))

    response = await post_pdf(client, make_pdf(pages=2))

    assert response.status_code == 200
    assert response.text == ("\nСТРАНИЦА 1\nраспознанный текст\n\nСТРАНИЦА 2\nраспознанный текст\n")


async def test_ocr_retries_then_succeeds(client: httpx.AsyncClient, mock_vllm: respx.MockRouter) -> None:
    route = mock_vllm.post("/v1/chat/completions")
    route.side_effect = [httpx.Response(500), httpx.Response(200, json=vllm_reply("после повтора"))]

    response = await post_pdf(client, make_pdf())

    assert route.call_count == 2
    assert "после повтора" in response.text


async def test_ocr_failed_page_contains_error_marker(
    client: httpx.AsyncClient, mock_vllm: respx.MockRouter
) -> None:
    mock_vllm.post("/v1/chat/completions").respond(status_code=500)

    response = await post_pdf(client, make_pdf())

    assert response.status_code == 200
    assert "[Ошибка OCR:" in response.text


async def test_ocr_short_answer_is_marked_empty(
    client: httpx.AsyncClient, mock_vllm: respx.MockRouter
) -> None:
    mock_vllm.post("/v1/chat/completions").respond(json=vllm_reply("ab"))

    response = await post_pdf(client, make_pdf())

    assert "[Пустая страница" in response.text


async def test_non_pdf_extension_returns_400(client: httpx.AsyncClient) -> None:
    response = await post_pdf(client, b"data", name="doc.txt")
    assert response.status_code == 400


async def test_empty_file_returns_400(client: httpx.AsyncClient) -> None:
    response = await post_pdf(client, b"")
    assert response.status_code == 400


async def test_broken_pdf_returns_400(client: httpx.AsyncClient) -> None:
    response = await post_pdf(client, b"not a pdf at all")
    assert response.status_code == 400


async def test_too_large_file_returns_413(client: httpx.AsyncClient) -> None:
    response = await post_pdf(client, b"0" * (1024 * 1024 + 1))
    assert response.status_code == 413
