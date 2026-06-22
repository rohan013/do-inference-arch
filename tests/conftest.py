"""Shared pytest fixtures for router and worker tests."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient


SAMPLE_VLLM_METRICS = """
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama-3.1-8b"} 5.0
vllm:num_requests_waiting{model_name="other"} 3.0
"""


async def _wait_for_stop_poll_loop(
    self,
    client,
    prefill_service_url,
    decode_service_url,
    interval,
    stop,
):
    await stop.wait()


@pytest.fixture
def router_client():
    """FastAPI TestClient with metrics server and poll loop disabled."""
    with (
        patch("prometheus_client.start_http_server"),
        patch(
            "pool_metrics.PoolLoadTracker.poll_loop",
            new=_wait_for_stop_poll_loop,
        ),
    ):
        from main import app

        with TestClient(app) as client:
            yield client


@pytest.fixture
def chat_payload():
    return {
        "model": "llama-3.1-8b",
        "messages": [{"role": "user", "content": "Hello"}],
    }
