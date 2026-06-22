"""Tests for vLLM pool metrics polling."""

import asyncio

import httpx
import pytest
import respx

from pool_metrics import PoolLoadTracker, metrics_url_from_service_url, parse_vllm_queue_depth

from tests.conftest import SAMPLE_VLLM_METRICS


class TestMetricsParsing:
    def test_parse_vllm_queue_depth(self):
        assert parse_vllm_queue_depth(SAMPLE_VLLM_METRICS) == 8

    def test_parse_empty_metrics(self):
        assert parse_vllm_queue_depth("") == 0

    def test_metrics_url_from_service_url(self):
        assert (
            metrics_url_from_service_url("http://vllm-decode:8000")
            == "http://vllm-decode:8001/metrics"
        )

    def test_metrics_url_strips_trailing_slash(self):
        assert (
            metrics_url_from_service_url("http://vllm-prefill:8000/")
            == "http://vllm-prefill:8001/metrics"
        )


class TestPoolLoadTracker:
    @pytest.mark.asyncio
    async def test_poll_once_success(self):
        tracker = PoolLoadTracker()
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.get("http://decode:8001/metrics").respond(text=SAMPLE_VLLM_METRICS)
                respx.get("http://prefill:8001/metrics").respond(text=SAMPLE_VLLM_METRICS)
                await tracker.poll_once(
                    client,
                    "http://prefill:8001/metrics",
                    "http://decode:8001/metrics",
                )
        assert tracker.decode_waiting == 8
        assert tracker.prefill_waiting == 8
        assert tracker.stale is False

    @pytest.mark.asyncio
    async def test_poll_once_decode_failure_marks_stale(self):
        tracker = PoolLoadTracker()
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.get("http://decode:8001/metrics").respond(status_code=503)
                respx.get("http://prefill:8001/metrics").respond(text=SAMPLE_VLLM_METRICS)
                await tracker.poll_once(
                    client,
                    "http://prefill:8001/metrics",
                    "http://decode:8001/metrics",
                )
        assert tracker.prefill_waiting == 8
        assert tracker.stale is True

    @pytest.mark.asyncio
    async def test_poll_once_prefill_failure_marks_stale(self):
        tracker = PoolLoadTracker()
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.get("http://decode:8001/metrics").respond(text=SAMPLE_VLLM_METRICS)
                respx.get("http://prefill:8001/metrics").respond(status_code=503)
                await tracker.poll_once(
                    client,
                    "http://prefill:8001/metrics",
                    "http://decode:8001/metrics",
                )
        assert tracker.decode_waiting == 8
        assert tracker.stale is True

    @pytest.mark.asyncio
    async def test_poll_loop_exits_when_stopped(self):
        tracker = PoolLoadTracker()
        stop = asyncio.Event()
        async with httpx.AsyncClient() as client:
            with respx.mock:
                respx.get(url__regex=r".*/metrics").respond(text=SAMPLE_VLLM_METRICS)
                task = asyncio.create_task(
                    tracker.poll_loop(
                        client,
                        "http://prefill:8000",
                        "http://decode:8000",
                        interval=0.01,
                        stop=stop,
                    )
                )
                await asyncio.sleep(0.05)
                stop.set()
                await asyncio.wait_for(task, timeout=1.0)
        assert tracker.decode_waiting == 8
