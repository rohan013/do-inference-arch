"""
Request tracing and Prometheus metrics for the inference stack.

Traces are emitted as structured JSON to a Unix socket (<1µs overhead).
A Fluent Bit sidecar aggregates and ships to Prometheus/Thanos.
GPU hardware metrics (SM utilization, HBM bandwidth, NVLink throughput)
are collected separately via DCGM Exporter (NVIDIA) or ROCm SMI (AMD).
"""

import time
from dataclasses import dataclass, field
from typing import Optional

from prometheus_client import Counter, Gauge, Histogram, Summary


# ---------------------------------------------------------------------------
# Application-level latency metrics
# ---------------------------------------------------------------------------

# Bucket edges tuned to our SLO thresholds (ms)
TTFT_HIST = Histogram(
    "inference_ttft_ms",
    "Time to first token in milliseconds",
    ["model_id", "precision", "kv_cache_hit"],
    buckets=[50, 100, 150, 200, 300, 500, 800, 1200, 2000, 5000],
)

TPOT_HIST = Histogram(
    "inference_tpot_ms",
    "Time per output token in milliseconds",
    ["model_id", "precision"],
    buckets=[5, 10, 15, 20, 30, 50, 80, 120, 200],
)

# Summary exposes P50 and P99 automatically
ITL_SUMMARY = Summary(
    "inference_itl_ms",
    "Inter-token latency in milliseconds",
    ["model_id"],
)

# ---------------------------------------------------------------------------
# Throughput and efficiency
# ---------------------------------------------------------------------------

TOKENS_PER_SEC = Gauge(
    "inference_tokens_per_second",
    "Decode throughput per node",
    ["node_id", "model_id"],
)

# phase = "prefill" | "decode"
# Decode MFU is inherently low (<20%) — do not compare to prefill MFU target (>55%)
GPU_MFU = Gauge(
    "inference_gpu_mfu",
    "Model FLOP utilization [0,1]",
    ["node_id", "phase"],
)

KV_CACHE_HIT_RATE = Gauge(
    "inference_kv_cache_hit_rate",
    "Prefix cache hit rate (prefix_tokens_reused / prompt_tokens)",
    ["node_id", "model_id"],
)

# ---------------------------------------------------------------------------
# Agentic workflow metrics
# ---------------------------------------------------------------------------

TOOL_WAIT_HIST = Histogram(
    "agentic_tool_wait_seconds",
    "Time a request spent blocked waiting for a tool result",
    ["model_id"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30, 60],
)

KV_SWAP_TOTAL = Counter(
    "agentic_kv_swap_total",
    "Total KV cache swaps to CPU (tool preemption events)",
    ["node_id"],
)

# ---------------------------------------------------------------------------
# MoE health
# ---------------------------------------------------------------------------

# High CV (>0.6) means one expert is absorbing most tokens — routing collapse
EXPERT_LOAD_CV = Gauge(
    "moe_expert_load_imbalance_cv",
    "Coefficient of variation of per-expert token counts (std/mean)",
    ["model_id", "layer_idx"],
)

# ---------------------------------------------------------------------------
# Request trace
# ---------------------------------------------------------------------------

@dataclass
class RequestTrace:
    request_id: str
    model_id: str
    prompt_tokens: int
    max_new_tokens: int
    node_id: str
    precision: str = "bf16"

    # Unix epoch seconds — set by the serving framework at each milestone
    t_received: float = 0.0
    t_queue_exit: float = 0.0
    t_prefill_start: float = 0.0
    t_prefill_end: float = 0.0     # = first token generated
    t_decode_end: float = 0.0      # = last token generated

    # Cache attribution
    kv_cache_hit: bool = False
    prefix_tokens_reused: int = 0

    # MoE-specific (0.0 for dense models)
    expert_routing_entropy: float = 0.0

    # Derived — populated by finalize()
    ttft_ms: float = field(init=False, default=0.0)
    tpot_ms: float = field(init=False, default=0.0)
    tokens_per_sec: float = field(init=False, default=0.0)

    def finalize(self, output_tokens: int, itl_samples: Optional[list[float]] = None) -> None:
        self.ttft_ms = (self.t_prefill_end - self.t_received) * 1000
        decode_duration = self.t_decode_end - self.t_prefill_end
        self.tpot_ms = (decode_duration / max(output_tokens - 1, 1)) * 1000
        total = self.t_decode_end - self.t_received
        self.tokens_per_sec = output_tokens / total if total > 0 else 0.0

        # Push to Prometheus
        hit = str(self.kv_cache_hit).lower()
        TTFT_HIST.labels(self.model_id, self.precision, hit).observe(self.ttft_ms)
        TPOT_HIST.labels(self.model_id, self.precision).observe(self.tpot_ms)
        TOKENS_PER_SEC.labels(self.node_id, self.model_id).set(self.tokens_per_sec)

        if itl_samples:
            for sample in itl_samples:
                ITL_SUMMARY.labels(self.model_id).observe(sample)

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "model_id": self.model_id,
            "node_id": self.node_id,
            "precision": self.precision,
            "prompt_tokens": self.prompt_tokens,
            "kv_cache_hit": self.kv_cache_hit,
            "prefix_tokens_reused": self.prefix_tokens_reused,
            "ttft_ms": round(self.ttft_ms, 2),
            "tpot_ms": round(self.tpot_ms, 2),
            "tokens_per_sec": round(self.tokens_per_sec, 1),
            "expert_routing_entropy": self.expert_routing_entropy,
        }
