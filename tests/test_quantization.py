"""Tests for quantization."""

from specinferkit.quantization.fp8 import FP8Quantizer, QuantizationConfig


class TestFP8Quantizer:
    def test_init_default(self):
        q = FP8Quantizer()
        assert q.config.dtype == "float8_e4m3fn"

    def test_init_with_config(self):
        config = QuantizationConfig(dtype="float8_e5m2")
        q = FP8Quantizer(config)
        assert q.config.dtype == "float8_e5m2"

    def test_init_invalid_dtype(self):
        config = QuantizationConfig(dtype="float16")
        with pytest.raises(ValueError):
            FP8Quantizer(config)
