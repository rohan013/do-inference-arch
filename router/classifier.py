"""
Request classification for the inference router.

Determines prompt length, model class, and upstream pool based on the
architecture spec disaggregation threshold (prompt_len > 2K → prefill pool).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
import re
from typing import Optional


class ModelClass(str, Enum):
    MOE = "moe"
    DENSE_LONG_CONTEXT = "dense_long_context"
    DRAFT = "draft"


class RoutePool(str, Enum):
    PREFILL = "prefill"
    DECODE = "decode"


class UserTier(str, Enum):
    PREMIUM = "premium"
    REGULAR = "regular"


@dataclass(frozen=True)
class Classification:
    prompt_len: int
    model_class: ModelClass
    user_tier: UserTier
    route_pool: RoutePool
    upstream_url: str
    route_reason: str


# Heuristic token estimate: ~4 chars per token for English text.
_CHARS_PER_TOKEN = 4

_MOE_PATTERN = re.compile(r"mixtral|moe|8x7b|8x22b", re.I)
_DRAFT_PATTERN = re.compile(r"draft|7b-instruct|speculative", re.I)


def _config_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _config_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def estimate_prompt_tokens(messages: list[dict]) -> int:
    """Estimate prompt token count from OpenAI-style message list."""
    total_chars = 0
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            total_chars += len(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
        # Role/name overhead per message.
        total_chars += 8
    return max(1, total_chars // _CHARS_PER_TOKEN)


def parse_user_tier(header_value: str | None) -> UserTier:
    """Parse X-User-Tier header value; default to REGULAR when missing or invalid."""
    if header_value is None:
        return UserTier.REGULAR
    normalized = header_value.strip().upper()
    if normalized == "PREMIUM":
        return UserTier.PREMIUM
    return UserTier.REGULAR


def _tier_saturation_threshold(tier: UserTier) -> int:
    if tier == UserTier.PREMIUM:
        return _config_int("DECODE_QUEUE_SATURATION_THRESHOLD_PREMIUM", 32)
    return _config_int("DECODE_QUEUE_SATURATION_THRESHOLD_REGULAR", 8)


def classify_model(model: str) -> ModelClass:
    """Map served model name to hardware/parallelism class from the spec."""
    name = model.lower()
    if _DRAFT_PATTERN.search(name):
        return ModelClass.DRAFT
    if _MOE_PATTERN.search(name):
        return ModelClass.MOE
    return ModelClass.DENSE_LONG_CONTEXT


def classify_request(
    messages: list[dict],
    model: str,
    *,
    user_tier: UserTier = UserTier.REGULAR,
    threshold: Optional[int] = None,
    prefill_url: Optional[str] = None,
    decode_url: Optional[str] = None,
    decode_queue: int = 0,
    prefill_queue: int = 0,
    decode_saturation_threshold: Optional[int] = None,
    prefill_spillover_enabled: Optional[bool] = None,
) -> Classification:
    """
    Classify an incoming request and select the upstream vLLM pool.

    Long prompts (> threshold) always use disaggregated prefill.
    Short prompts default to decode; spill to prefill when decode is saturated
    for the caller's tier (PREMIUM spills later than REGULAR).
    """
    threshold = threshold if threshold is not None else _config_int(
        "PROMPT_LEN_DISAGGREGATE_THRESHOLD", 2048
    )
    prefill_url = prefill_url or os.getenv(
        "PREFILL_SERVICE_URL", "http://vllm-prefill:8000"
    )
    decode_url = decode_url or os.getenv(
        "DECODE_SERVICE_URL", "http://vllm-decode:8000"
    )
    if decode_saturation_threshold is None:
        decode_saturation_threshold = _tier_saturation_threshold(user_tier)
    if prefill_spillover_enabled is None:
        prefill_spillover_enabled = _config_bool("PREFILL_SPILLOVER_ENABLED", True)

    prompt_len = estimate_prompt_tokens(messages)
    model_class = classify_model(model)

    if prompt_len > threshold:
        return Classification(
            prompt_len=prompt_len,
            model_class=model_class,
            user_tier=user_tier,
            route_pool=RoutePool.PREFILL,
            upstream_url=prefill_url.rstrip("/"),
            route_reason="disaggregate",
        )

    if (
        prefill_spillover_enabled
        and decode_queue >= decode_saturation_threshold
        and prefill_queue < decode_queue
    ):
        return Classification(
            prompt_len=prompt_len,
            model_class=model_class,
            user_tier=user_tier,
            route_pool=RoutePool.PREFILL,
            upstream_url=prefill_url.rstrip("/"),
            route_reason="spillover",
        )

    return Classification(
        prompt_len=prompt_len,
        model_class=model_class,
        user_tier=user_tier,
        route_pool=RoutePool.DECODE,
        upstream_url=decode_url.rstrip("/"),
        route_reason="default",
    )
