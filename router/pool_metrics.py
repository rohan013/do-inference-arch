"""Background polling of vLLM queue-depth metrics for load-aware routing."""

from __future__ import annotations

import asyncio
import logging
import os
import re
from urllib.parse import urlparse, urlunparse

import httpx

logger = logging.getLogger("request-router.pool-metrics")

_QUEUE_METRIC = re.compile(
    r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.eE+-]+)$",
    re.MULTILINE,
)


def metrics_url_from_service_url(service_url: str) -> str:
    """Map OpenAI service URL (port 8000) to Prometheus scrape URL (port 8001)."""
    parsed = urlparse(service_url.rstrip("/"))
    host = parsed.hostname or parsed.netloc.split(":")[0]
    return urlunparse((parsed.scheme, f"{host}:8001", "/metrics", "", "", ""))


def parse_vllm_queue_depth(metrics_text: str) -> int:
    """Sum vllm:num_requests_waiting across all labeled series."""
    total = 0.0
    for match in _QUEUE_METRIC.finditer(metrics_text):
        total += float(match.group(1))
    return int(total)


class PoolLoadTracker:
    """Caches per-pool queue depth from periodic /metrics polls."""

    def __init__(self) -> None:
        self.decode_waiting: int = 0
        self.prefill_waiting: int = 0
        self.stale: bool = True

    async def poll_once(
        self,
        client: httpx.AsyncClient,
        prefill_metrics_url: str,
        decode_metrics_url: str,
    ) -> None:
        decode_ok = prefill_ok = False
        try:
            response = await client.get(decode_metrics_url)
            response.raise_for_status()
            self.decode_waiting = parse_vllm_queue_depth(response.text)
            decode_ok = True
        except httpx.HTTPError:
            logger.warning("Failed to fetch decode metrics from %s", decode_metrics_url)

        try:
            response = await client.get(prefill_metrics_url)
            response.raise_for_status()
            self.prefill_waiting = parse_vllm_queue_depth(response.text)
            prefill_ok = True
        except httpx.HTTPError:
            logger.warning("Failed to fetch prefill metrics from %s", prefill_metrics_url)

        self.stale = not (decode_ok and prefill_ok)

    async def poll_loop(
        self,
        client: httpx.AsyncClient,
        prefill_service_url: str,
        decode_service_url: str,
        interval: float,
        stop: asyncio.Event,
    ) -> None:
        prefill_metrics_url = metrics_url_from_service_url(prefill_service_url)
        decode_metrics_url = metrics_url_from_service_url(decode_service_url)
        logger.info(
            "Pool metrics poller started interval=%ss prefill=%s decode=%s",
            interval,
            prefill_metrics_url,
            decode_metrics_url,
        )
        while not stop.is_set():
            await self.poll_once(client, prefill_metrics_url, decode_metrics_url)
            try:
                await asyncio.wait_for(stop.wait(), timeout=interval)
            except asyncio.TimeoutError:
                continue
