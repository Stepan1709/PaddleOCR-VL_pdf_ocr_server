import httpx
import respx


async def test_live(client: httpx.AsyncClient) -> None:
    response = await client.get("/live")
    assert response.json() == {"status": "alive"}


async def test_health_healthy(client: httpx.AsyncClient, mock_vllm: respx.MockRouter) -> None:
    mock_vllm.get("/v1/models").respond(json={"data": [{"id": "paddle-model"}]})

    body = (await client.get("/health")).json()

    assert body["status"] == "healthy"
    assert body["vllm"] == "connected"


async def test_health_degraded_when_vllm_down(client: httpx.AsyncClient, mock_vllm: respx.MockRouter) -> None:
    mock_vllm.get("/v1/models").mock(side_effect=httpx.ConnectError("refused"))

    body = (await client.get("/health")).json()

    assert body["status"] == "degraded"
    assert body["vllm"] == "disconnected"


async def test_health_degraded_on_vllm_error_status(
    client: httpx.AsyncClient, mock_vllm: respx.MockRouter
) -> None:
    mock_vllm.get("/v1/models").respond(status_code=503)

    body = (await client.get("/health")).json()

    assert body == {"status": "degraded", "vllm": "error_503", "model_available": None, "error": None}
