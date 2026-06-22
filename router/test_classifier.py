"""Unit tests for request classification and pool metrics parsing."""

import os
import unittest

from classifier import ModelClass, RoutePool, classify_model, classify_request, estimate_prompt_tokens
from pool_metrics import metrics_url_from_service_url, parse_vllm_queue_depth


SAMPLE_METRICS = """
# HELP vllm:num_requests_waiting Number of requests waiting to be processed.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{model_name="llama-3.1-8b"} 5.0
vllm:num_requests_waiting{model_name="other"} 3.0
"""


class ClassifierTests(unittest.TestCase):
    def test_short_prompt_routes_to_decode(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            threshold=2048,
            prefill_url="http://prefill:8000",
            decode_url="http://decode:8000",
        )
        self.assertEqual(result.route_pool, RoutePool.DECODE)
        self.assertEqual(result.upstream_url, "http://decode:8000")
        self.assertEqual(result.model_class, ModelClass.DENSE_LONG_CONTEXT)
        self.assertEqual(result.route_reason, "default")

    def test_long_prompt_routes_to_prefill(self):
        long_content = "word " * 3000
        messages = [{"role": "user", "content": long_content}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            threshold=2048,
            prefill_url="http://prefill:8000",
            decode_url="http://decode:8000",
        )
        self.assertGreater(result.prompt_len, 2048)
        self.assertEqual(result.route_pool, RoutePool.PREFILL)
        self.assertEqual(result.upstream_url, "http://prefill:8000")
        self.assertEqual(result.route_reason, "disaggregate")

    def test_spillover_when_decode_saturated(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            threshold=2048,
            prefill_url="http://prefill:8000",
            decode_url="http://decode:8000",
            decode_queue=32,
            prefill_queue=4,
            decode_saturation_threshold=16,
        )
        self.assertEqual(result.route_pool, RoutePool.PREFILL)
        self.assertEqual(result.route_reason, "spillover")

    def test_no_spillover_when_prefill_also_loaded(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            decode_queue=32,
            prefill_queue=40,
            decode_saturation_threshold=16,
        )
        self.assertEqual(result.route_pool, RoutePool.DECODE)
        self.assertEqual(result.route_reason, "default")

    def test_long_prompt_still_prefill_when_decode_saturated(self):
        long_content = "word " * 3000
        messages = [{"role": "user", "content": long_content}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            threshold=2048,
            decode_queue=100,
            prefill_queue=0,
        )
        self.assertEqual(result.route_pool, RoutePool.PREFILL)
        self.assertEqual(result.route_reason, "disaggregate")

    def test_mixtral_classified_as_moe(self):
        self.assertEqual(classify_model("mixtral-8x7b"), ModelClass.MOE)

    def test_draft_model_class(self):
        self.assertEqual(classify_model("llama-draft-7b"), ModelClass.DRAFT)

    def test_estimate_prompt_tokens_multimodal_text_parts(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "abcd" * 10}],
            }
        ]
        self.assertGreaterEqual(estimate_prompt_tokens(messages), 10)

    def test_env_threshold(self):
        os.environ["PROMPT_LEN_DISAGGREGATE_THRESHOLD"] = "10"
        try:
            messages = [{"role": "user", "content": "hello world " * 20}]
            result = classify_request(messages, "llama-3.1-8b")
            self.assertEqual(result.route_pool, RoutePool.PREFILL)
        finally:
            os.environ.pop("PROMPT_LEN_DISAGGREGATE_THRESHOLD", None)


class PoolMetricsTests(unittest.TestCase):
    def test_parse_vllm_queue_depth(self):
        self.assertEqual(parse_vllm_queue_depth(SAMPLE_METRICS), 8)

    def test_metrics_url_from_service_url(self):
        self.assertEqual(
            metrics_url_from_service_url("http://vllm-decode:8000"),
            "http://vllm-decode:8001/metrics",
        )


if __name__ == "__main__":
    unittest.main()
