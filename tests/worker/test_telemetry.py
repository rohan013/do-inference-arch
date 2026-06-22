"""Tests for inference telemetry and request tracing."""

import pytest

from worker.telemetry import RequestTrace


class TestRequestTrace:
    def _trace(self, **overrides) -> RequestTrace:
        defaults = dict(
            request_id="req-1",
            model_id="llama-3.1-8b",
            prompt_tokens=128,
            max_new_tokens=64,
            node_id="decode-0",
            t_received=100.0,
            t_prefill_end=100.2,
            t_decode_end=100.5,
        )
        defaults.update(overrides)
        return RequestTrace(**defaults)

    def test_finalize_computes_latencies(self):
        trace = self._trace()
        trace.finalize(output_tokens=10)
        assert trace.ttft_ms == pytest.approx(200.0)
        assert trace.tpot_ms == pytest.approx(33.333, rel=0.01)
        assert trace.tokens_per_sec == pytest.approx(20.0, rel=0.01)

    def test_finalize_single_output_token(self):
        trace = self._trace(t_decode_end=100.2)
        trace.finalize(output_tokens=1)
        assert trace.tpot_ms == pytest.approx(0.0, abs=0.01)

    def test_finalize_zero_duration(self):
        trace = self._trace(t_received=100.0, t_prefill_end=100.0, t_decode_end=100.0)
        trace.finalize(output_tokens=5)
        assert trace.tokens_per_sec == 0.0

    def test_finalize_with_itl_samples(self):
        trace = self._trace()
        trace.finalize(output_tokens=5, itl_samples=[10.0, 12.0, 11.0])

    def test_to_json_rounds_fields(self):
        trace = self._trace(kv_cache_hit=True, prefix_tokens_reused=64, expert_routing_entropy=0.42)
        trace.finalize(output_tokens=8)
        payload = trace.to_json()
        assert payload["request_id"] == "req-1"
        assert payload["kv_cache_hit"] is True
        assert payload["prefix_tokens_reused"] == 64
        assert payload["ttft_ms"] == 200.0
        assert payload["expert_routing_entropy"] == 0.42
