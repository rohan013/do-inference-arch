"""Tests for the FastAPI request router."""

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from classifier import Classification, ModelClass, RoutePool, UserTier
from main import _routing_headers


def _sample_classification(**overrides):
    defaults = dict(
        prompt_len=2,
        model_class=ModelClass.DENSE_LONG_CONTEXT,
        user_tier=UserTier.REGULAR,
        route_pool=RoutePool.DECODE,
        upstream_url="http://vllm-decode:8000",
        route_reason="default",
    )
    defaults.update(overrides)
    return Classification(**defaults)


def _json_upstream_response(payload=None, status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.json.return_value = payload or {"choices": [{"message": {"content": "Hi"}}]}
    response.headers = {"content-type": "application/json"}
    return response


def _stream_upstream_response(content=b"data: {}\n\n", status_code=200):
    response = MagicMock()
    response.status_code = status_code
    response.headers = {"content-type": "text/event-stream"}

    async def aiter_bytes():
        yield content

    response.aiter_bytes = aiter_bytes
    response.aread = AsyncMock(return_value=content)
    response.aclose = AsyncMock()
    return response


@pytest.fixture
def mock_http(router_client):
    client = router_client.app.state.http
    client.post = AsyncMock(return_value=_json_upstream_response())
    client.send = AsyncMock(return_value=_stream_upstream_response())
    return router_client


class TestRoutingHeaders:
    def test_routing_headers_include_tier(self):
        classification = _sample_classification(user_tier=UserTier.PREMIUM)
        headers = _routing_headers(classification, "req-123")
        assert headers["X-Request-Id"] == "req-123"
        assert headers["X-Router-User-Tier"] == "premium"
        assert headers["X-Router-Pool"] == "decode"


class TestHealthAndMetrics:
    def test_healthz(self, router_client):
        response = router_client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_metrics_endpoint(self, router_client):
        response = router_client.get("/metrics")
        assert response.status_code == 200
        assert "router_requests_total" in response.text


class TestChatCompletions:
    def test_missing_messages_returns_400(self, router_client):
        response = router_client.post("/v1/chat/completions", json={"model": "llama-3.1-8b"})
        assert response.status_code == 400
        assert response.json()["detail"] == "messages is required"

    def test_proxies_json_response_to_decode(self, mock_http, chat_payload):
        response = mock_http.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 200
        mock_http.app.state.http.post.assert_awaited_once()
        assert response.headers["X-Router-Pool"] == "decode"
        assert response.headers["X-Router-Route-Reason"] == "default"

    def test_uses_default_model_when_omitted(self, mock_http):
        mock_http.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hi"}]},
        )
        body = mock_http.app.state.http.post.await_args.kwargs["json"]
        assert body["model"] == "llama-3.1-8b"

    def test_premium_tier_header_propagates(self, mock_http, chat_payload):
        response = mock_http.post(
            "/v1/chat/completions",
            json=chat_payload,
            headers={"X-User-Tier": "PREMIUM", "X-Request-Id": "tier-test"},
        )
        assert response.headers["X-Router-User-Tier"] == "premium"
        assert response.headers["X-Request-Id"] == "tier-test"

    def test_long_prompt_routes_to_prefill(self, mock_http):
        payload = {
            "model": "llama-3.1-8b",
            "messages": [{"role": "user", "content": "word " * 3000}],
        }
        response = mock_http.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        url = mock_http.app.state.http.post.await_args.args[0]
        assert url.startswith("http://vllm-prefill:8000")
        assert response.headers["X-Router-Pool"] == "prefill"

    def test_upstream_http_error_returns_502(self, mock_http, chat_payload):
        mock_http.app.state.http.post = AsyncMock(
            side_effect=httpx.ConnectError("connection refused")
        )
        response = mock_http.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 502
        assert "Upstream unavailable" in response.json()["detail"]

    def test_upstream_4xx_json_error(self, mock_http, chat_payload):
        mock_http.app.state.http.post = AsyncMock(
            return_value=_json_upstream_response({"error": "rate limited"}, status_code=429)
        )
        response = mock_http.post("/v1/chat/completions", json=chat_payload)
        assert response.status_code == 429
        assert response.json()["error"] == "rate limited"

    def test_streaming_success(self, mock_http, chat_payload):
        stream_payload = {**chat_payload, "stream": True}
        response = mock_http.post("/v1/chat/completions", json=stream_payload)
        assert response.status_code == 200
        assert b"data:" in response.content
        mock_http.app.state.http.send.assert_awaited_once()

    def test_streaming_upstream_error_with_json_body(self, mock_http, chat_payload):
        mock_http.app.state.http.send = AsyncMock(
            return_value=_stream_upstream_response(
                content=b'{"error":"busy"}',
                status_code=503,
            )
        )
        stream_payload = {**chat_payload, "stream": True}
        response = mock_http.post("/v1/chat/completions", json=stream_payload)
        assert response.status_code == 503
        assert response.json()["error"] == "busy"

    def test_streaming_upstream_error_with_non_json_body(self, mock_http, chat_payload):
        mock_http.app.state.http.send = AsyncMock(
            return_value=_stream_upstream_response(content=b"plain error", status_code=500)
        )
        stream_payload = {**chat_payload, "stream": True}
        response = mock_http.post("/v1/chat/completions", json=stream_payload)
        assert response.status_code == 500
        assert response.json()["detail"] == "plain error"

    def test_routing_headers_forwarded_to_upstream(self, mock_http, chat_payload):
        mock_http.post("/v1/chat/completions", json=chat_payload)
        headers = mock_http.app.state.http.post.await_args.kwargs["headers"]
        assert headers["X-Router-Pool"] == "decode"
        assert headers["X-Router-Model-Class"] == "dense_long_context"
