"""Tests for router request classification."""

import os

import pytest

from classifier import (
    ModelClass,
    RoutePool,
    UserTier,
    classify_model,
    classify_request,
    estimate_prompt_tokens,
    parse_user_tier,
)


class TestEstimatePromptTokens:
    def test_empty_messages_minimum_one_token(self):
        assert estimate_prompt_tokens([{"role": "user", "content": ""}]) >= 1

    def test_string_content(self):
        messages = [{"role": "user", "content": "Hello world"}]
        assert estimate_prompt_tokens(messages) == max(1, len("Hello world") // 4 + 2)

    def test_multimodal_text_parts(self):
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "abcd" * 10}],
            }
        ]
        assert estimate_prompt_tokens(messages) >= 10


class TestClassifyModel:
    def test_mixtral_is_moe(self):
        assert classify_model("mixtral-8x7b") == ModelClass.MOE

    def test_moe_pattern_variants(self):
        assert classify_model("custom-moe-model") == ModelClass.MOE
        assert classify_model("model-8x22b") == ModelClass.MOE

    def test_draft_model_class(self):
        assert classify_model("llama-draft-7b") == ModelClass.DRAFT
        assert classify_model("speculative-helper") == ModelClass.DRAFT

    def test_dense_long_context_default(self):
        assert classify_model("llama-3.1-8b") == ModelClass.DENSE_LONG_CONTEXT


class TestParseUserTier:
    def test_missing_defaults_regular(self):
        assert parse_user_tier(None) == UserTier.REGULAR

    def test_premium(self):
        assert parse_user_tier("PREMIUM") == UserTier.PREMIUM
        assert parse_user_tier(" premium ") == UserTier.PREMIUM

    def test_regular(self):
        assert parse_user_tier("REGULAR") == UserTier.REGULAR

    def test_invalid_defaults_regular(self):
        assert parse_user_tier("enterprise") == UserTier.REGULAR


class TestClassifyRequest:
    def test_short_prompt_routes_to_decode(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            threshold=2048,
            prefill_url="http://prefill:8000",
            decode_url="http://decode:8000",
        )
        assert result.route_pool == RoutePool.DECODE
        assert result.upstream_url == "http://decode:8000"
        assert result.model_class == ModelClass.DENSE_LONG_CONTEXT
        assert result.route_reason == "default"
        assert result.user_tier == UserTier.REGULAR

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
        assert result.prompt_len > 2048
        assert result.route_pool == RoutePool.PREFILL
        assert result.route_reason == "disaggregate"

    def test_spillover_when_decode_saturated(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            decode_queue=32,
            prefill_queue=4,
            decode_saturation_threshold=16,
        )
        assert result.route_pool == RoutePool.PREFILL
        assert result.route_reason == "spillover"

    def test_no_spillover_when_prefill_also_loaded(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            decode_queue=32,
            prefill_queue=40,
            decode_saturation_threshold=16,
        )
        assert result.route_pool == RoutePool.DECODE

    def test_spillover_disabled(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            decode_queue=100,
            prefill_queue=0,
            decode_saturation_threshold=16,
            prefill_spillover_enabled=False,
        )
        assert result.route_pool == RoutePool.DECODE

    def test_premium_stays_on_decode_under_moderate_load(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            user_tier=UserTier.PREMIUM,
            decode_queue=20,
            prefill_queue=4,
        )
        assert result.route_pool == RoutePool.DECODE

    def test_regular_spills_under_moderate_load(self):
        messages = [{"role": "user", "content": "Hello"}]
        result = classify_request(
            messages,
            "llama-3.1-8b",
            user_tier=UserTier.REGULAR,
            decode_queue=20,
            prefill_queue=4,
        )
        assert result.route_pool == RoutePool.PREFILL
        assert result.route_reason == "spillover"

    def test_long_prompt_for_both_tiers(self):
        long_content = "word " * 3000
        messages = [{"role": "user", "content": long_content}]
        for tier in (UserTier.PREMIUM, UserTier.REGULAR):
            result = classify_request(
                messages,
                "llama-3.1-8b",
                user_tier=tier,
                threshold=2048,
                decode_queue=100,
            )
            assert result.route_pool == RoutePool.PREFILL

    def test_env_threshold(self):
        os.environ["PROMPT_LEN_DISAGGREGATE_THRESHOLD"] = "10"
        try:
            messages = [{"role": "user", "content": "hello world " * 20}]
            result = classify_request(messages, "llama-3.1-8b")
            assert result.route_pool == RoutePool.PREFILL
        finally:
            os.environ.pop("PROMPT_LEN_DISAGGREGATE_THRESHOLD", None)

    def test_env_tier_thresholds(self):
        os.environ["DECODE_QUEUE_SATURATION_THRESHOLD_PREMIUM"] = "5"
        os.environ["DECODE_QUEUE_SATURATION_THRESHOLD_REGULAR"] = "50"
        try:
            messages = [{"role": "user", "content": "Hi"}]
            premium = classify_request(
                messages, "llama-3.1-8b", user_tier=UserTier.PREMIUM, decode_queue=10, prefill_queue=0
            )
            regular = classify_request(
                messages, "llama-3.1-8b", user_tier=UserTier.REGULAR, decode_queue=10, prefill_queue=0
            )
            assert premium.route_pool == RoutePool.PREFILL
            assert regular.route_pool == RoutePool.DECODE
        finally:
            os.environ.pop("DECODE_QUEUE_SATURATION_THRESHOLD_PREMIUM", None)
            os.environ.pop("DECODE_QUEUE_SATURATION_THRESHOLD_REGULAR", None)

    def test_env_spillover_bool(self):
        os.environ["PREFILL_SPILLOVER_ENABLED"] = "false"
        try:
            messages = [{"role": "user", "content": "Hi"}]
            result = classify_request(
                messages,
                "llama-3.1-8b",
                decode_queue=100,
                prefill_queue=0,
            )
            assert result.route_pool == RoutePool.DECODE
        finally:
            os.environ.pop("PREFILL_SPILLOVER_ENABLED", None)

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
    def test_env_spillover_enabled_truthy(self, value):
        os.environ["PREFILL_SPILLOVER_ENABLED"] = value
        try:
            messages = [{"role": "user", "content": "Hi"}]
            result = classify_request(
                messages,
                "llama-3.1-8b",
                decode_queue=100,
                prefill_queue=0,
                decode_saturation_threshold=16,
            )
            assert result.route_pool == RoutePool.PREFILL
        finally:
            os.environ.pop("PREFILL_SPILLOVER_ENABLED", None)
