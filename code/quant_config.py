"""
Per-layer quantization configuration for heterogeneous model serving.

Supports FP8 (E4M3/E5M2), INT8 SmoothQuant, and BF16 fallback.
Designed to run on both NVIDIA (H200/B200) and AMD (MI300X/MI325X) via
Triton-compiled kernels — no CUDA-specific code in this config layer.
"""

from dataclasses import dataclass, field
from typing import Literal

QuantMode = Literal["fp8_e4m3", "fp8_e5m2", "int8_sq", "bf16"]


@dataclass
class LayerQuantConfig:
    mode: QuantMode
    # Dynamic: compute scale per-tensor at runtime (safer, ~2% slower)
    # Static: use pre-calibrated scale (faster, requires calibration dataset)
    dynamic_scaling: bool = True
    # SmoothQuant: migrate outlier magnitude from activations to weights
    # 0.0 = no migration, 0.85 = recommended for most dense models
    smooth_alpha: float = 0.0
    # Clip activations to this percentile before quantization
    clip_ratio: float = 1.0


@dataclass
class ModelQuantConfig:
    # Applied to all layers not in layer_overrides
    default: LayerQuantConfig = field(
        default_factory=lambda: LayerQuantConfig("fp8_e4m3")
    )

    # Per-layer overrides — keys match HuggingFace module name suffixes
    layer_overrides: dict[str, LayerQuantConfig] = field(default_factory=dict)

    # Accuracy gate: refuse to serve if regression exceeds this threshold
    max_accuracy_regression_pct: float = 0.5
    calibration_dataset: str = "mmlu_5shot_1k"

    def get(self, layer_name: str) -> LayerQuantConfig:
        for suffix, cfg in self.layer_overrides.items():
            if layer_name.endswith(suffix):
                return cfg
        return self.default


# Layers that must always run in BF16 regardless of model or config.
# lm_head: logit resolution determines reasoning chain quality — never quantize.
# embed_tokens: embedding lookup, quantization causes vocab distribution shift.
ALWAYS_BF16 = {"embed_tokens", "lm_head", "embed_out"}

_BF16 = LayerQuantConfig("bf16")
_FP8_ATTN = LayerQuantConfig("fp8_e5m2", clip_ratio=0.99)   # wider dynamic range for attention
_INT8_FFN = LayerQuantConfig("int8_sq", smooth_alpha=0.85)


# Production config: Mixtral-style MoE 200B+
# FP8 E4M3 for expert FFN layers (post-ReLU activations are bounded).
# FP8 E5M2 for attention (wider dynamic range handles long-context outliers).
# BF16 for routing gate (probability distribution must be precise).
MIXTRAL_QUANT = ModelQuantConfig(
    default=LayerQuantConfig("fp8_e4m3"),
    layer_overrides={
        "embed_tokens":  _BF16,
        "lm_head":       _BF16,
        "self_attn":     _FP8_ATTN,
        "gate":          _BF16,           # MoE routing gate — keep precise
    },
)

# Production config: dense long-context reasoning model (Llama-class 400B+)
# INT8 SmoothQuant throughout — FP8 degrades chain-of-thought on math benchmarks.
# Validate: MATH-500 regression must be <0.5% before promoting to production.
DENSE_LONGCTX_QUANT = ModelQuantConfig(
    default=_INT8_FFN,
    layer_overrides={
        "embed_tokens": _BF16,
        "lm_head":      _BF16,
        "self_attn":    LayerQuantConfig("int8_sq", smooth_alpha=0.75),
    },
    max_accuracy_regression_pct=0.5,
    calibration_dataset="math500_1k",
)

# Production config: small draft model (<7B) for speculative decoding
# BF16 throughout — model is already small, memory saving from quantization is minimal,
# accuracy preservation matters more for draft acceptance rate.
DRAFT_MODEL_QUANT = ModelQuantConfig(
    default=LayerQuantConfig("bf16"),
    layer_overrides={},
    max_accuracy_regression_pct=1.0,
)
