"""
FastAPI request router for the DO Agentic Inference Cloud.

Classifies requests by prompt length and model class, then proxies to the
appropriate vLLM pool (prefill or decode).
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

from classifier import Classification, classify_request

LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger("request-router")

ROUTER_REQUESTS = Counter(
    "router_requests_total",
    "Total routed requests",
    ["route_pool", "model_class"],
)
ROUTER_LATENCY = Histogram(
    "router_upstream_latency_seconds",
    "Upstream vLLM round-trip latency",
    ["route_pool"],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60],
)
ROUTER_ERRORS = Counter(
    "router_upstream_errors_total",
    "Upstream proxy failures",
    ["route_pool", "status"],
)

DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "llama-3.1-8b")
HTTP_PORT = int(os.getenv("HTTP_PORT", "8080"))
METRICS_PORT = int(os.getenv("METRICS_PORT", "9090"))
UPSTREAM_TIMEOUT = float(os.getenv("UPSTREAM_TIMEOUT_SECONDS", "300"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    from prometheus_client import start_http_server

    start_http_server(METRICS_PORT)
    logger.info("Metrics server listening on :%s", METRICS_PORT)
    app.state.http = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)
    try:
        yield
    finally:
        await app.state.http.aclose()


app = FastAPI(title="DO Inference Router", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _routing_headers(classification: Classification, request_id: str) -> dict[str, str]:
    return {
        "X-Request-Id": request_id,
        "X-Router-Prompt-Len": str(classification.prompt_len),
        "X-Router-Model-Class": classification.model_class.value,
        "X-Router-Pool": classification.route_pool.value,
    }


async def _proxy_upstream(
    client: httpx.AsyncClient,
    classification: Classification,
    body: dict[str, Any],
    request_id: str,
    *,
    stream: bool = False,
) -> httpx.Response:
    url = f"{classification.upstream_url}/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        **_routing_headers(classification, request_id),
    }
    start = time.perf_counter()
    try:
        response = await client.post(url, json=body, headers=headers, stream=stream)
    except httpx.HTTPError as exc:
        ROUTER_ERRORS.labels(
            classification.route_pool.value, "upstream_error"
        ).inc()
        logger.exception("Upstream request failed request_id=%s url=%s", request_id, url)
        raise HTTPException(status_code=502, detail=f"Upstream unavailable: {exc}") from exc
    finally:
        ROUTER_LATENCY.labels(classification.route_pool.value).observe(
            time.perf_counter() - start
        )

    if not stream and response.status_code >= 400:
        ROUTER_ERRORS.labels(
            classification.route_pool.value, str(response.status_code)
        ).inc()

    return response


@app.post("/v1/chat/completions")
async def chat_completions(request: Request) -> Response:
    body = await request.json()
    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="messages is required")

    model = body.get("model") or DEFAULT_MODEL
    body["model"] = model

    classification = classify_request(messages, model)
    request_id = request.headers.get("X-Request-Id", str(uuid.uuid4()))

    ROUTER_REQUESTS.labels(
        classification.route_pool.value,
        classification.model_class.value,
    ).inc()

    logger.info(
        "route request_id=%s model=%s prompt_len=%d pool=%s class=%s",
        request_id,
        model,
        classification.prompt_len,
        classification.route_pool.value,
        classification.model_class.value,
    )

    client: httpx.AsyncClient = request.app.state.http

    if body.get("stream"):
        upstream = await _proxy_upstream(
            client, classification, body, request_id, stream=True
        )
        if upstream.status_code >= 400:
            content = await upstream.aread()
            await upstream.aclose()
            ROUTER_ERRORS.labels(
                classification.route_pool.value, str(upstream.status_code)
            ).inc()
            try:
                payload = json.loads(content)
            except json.JSONDecodeError:
                payload = {"detail": content.decode(errors="replace")}
            return JSONResponse(
                status_code=upstream.status_code,
                content=payload,
                headers=_routing_headers(classification, request_id),
            )

        async def stream_body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                await upstream.aclose()

        return StreamingResponse(
            stream_body(),
            status_code=upstream.status_code,
            media_type=upstream.headers.get("content-type", "text/event-stream"),
            headers=_routing_headers(classification, request_id),
        )

    upstream = await _proxy_upstream(client, classification, body, request_id)
    return JSONResponse(
        status_code=upstream.status_code,
        content=upstream.json(),
        headers=_routing_headers(classification, request_id),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=HTTP_PORT, log_level=LOG_LEVEL.lower())
