"""Unit tests for request classification."""

import os
import unittest

from classifier import ModelClass, RoutePool, classify_model, classify_request, estimate_prompt_tokens


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


if __name__ == "__main__":
    unittest.main()
