"""Tests for worker-side quantization configuration."""

from worker.quant_config import (
    ALWAYS_BF16,
    DENSE_LONGCTX_QUANT,
    DRAFT_MODEL_QUANT,
    MIXTRAL_QUANT,
    LayerQuantConfig,
    ModelQuantConfig,
)


class TestLayerQuantConfig:
    def test_defaults(self):
        cfg = LayerQuantConfig("fp8_e4m3")
        assert cfg.mode == "fp8_e4m3"
        assert cfg.dynamic_scaling is True
        assert cfg.smooth_alpha == 0.0


class TestModelQuantConfig:
    def test_get_returns_default_for_unknown_layer(self):
        cfg = ModelQuantConfig(default=LayerQuantConfig("bf16"))
        assert cfg.get("model.layers.0.mlp").mode == "bf16"

    def test_get_matches_layer_suffix(self):
        cfg = ModelQuantConfig(
            default=LayerQuantConfig("fp8_e4m3"),
            layer_overrides={
                "self_attn": LayerQuantConfig("fp8_e5m2"),
                "lm_head": LayerQuantConfig("bf16"),
            },
        )
        assert cfg.get("model.layers.31.self_attn").mode == "fp8_e5m2"
        assert cfg.get("model.lm_head").mode == "bf16"

    def test_always_bf16_layers(self):
        assert "lm_head" in ALWAYS_BF16
        assert "embed_tokens" in ALWAYS_BF16


class TestProductionConfigs:
    def test_mixtral_quant_layers(self):
        assert MIXTRAL_QUANT.get("model.embed_tokens").mode == "bf16"
        assert MIXTRAL_QUANT.get("model.layers.0.self_attn").mode == "fp8_e5m2"
        assert MIXTRAL_QUANT.get("model.layers.0.block_sparse_moe.gate").mode == "bf16"
        assert MIXTRAL_QUANT.get("model.layers.0.mlp.experts.0.w1").mode == "fp8_e4m3"

    def test_dense_longctx_quant(self):
        assert DENSE_LONGCTX_QUANT.get("model.layers.0.self_attn").smooth_alpha == 0.75
        assert DENSE_LONGCTX_QUANT.get("model.layers.0.mlp").mode == "int8_sq"
        assert DENSE_LONGCTX_QUANT.calibration_dataset == "math500_1k"

    def test_draft_model_quant(self):
        assert DRAFT_MODEL_QUANT.default.mode == "bf16"
        assert DRAFT_MODEL_QUANT.max_accuracy_regression_pct == 1.0
