"""Tests for quantization registry."""

import pytest
import torch
import torch.nn as nn

from tinylm.quant import (
    QUANT_REGISTRY,
    NoneQuantMethod,
    QuantConfig,
    QuantMethod,
    QuantParams,
    QuantRegistry,
    TernaryQuantMethod,
    available_and_ready_methods,
    available_methods,
    make_linear,
)


class TestQuantRegistry:
    """Tests for QuantRegistry class."""

    def test_registry_has_methods(self):
        """Test that registry has expected methods registered."""
        methods = available_methods()
        assert "none" in methods
        assert "ternary" in methods

    def test_none_method_always_available(self):
        """Test that 'none' method is always available."""
        assert "none" in available_and_ready_methods()

    def test_get_method(self):
        """Test retrieving methods from registry."""
        none_cls = QUANT_REGISTRY.get("none")
        assert none_cls is NoneQuantMethod
        assert none_cls.is_available()

    def test_get_unknown_method_raises(self):
        """Test that unknown method raises ValueError."""
        with pytest.raises(ValueError, match="Unknown"):
            QUANT_REGISTRY.get("unknown_method")

    def test_contains(self):
        """Test __contains__ method."""
        assert "none" in QUANT_REGISTRY
        assert "ternary" in QUANT_REGISTRY
        assert "unknown" not in QUANT_REGISTRY


class TestQuantParams:
    """Tests for QuantParams dataclass."""

    def test_default_params(self):
        """Test default parameter values."""
        params = QuantParams()
        assert params.enabled is True
        assert params.threshold_factor == 0.05
        assert params.per_channel is True
        assert params.backend == "auto"
        assert params.bits == 8
        assert params.symmetric is True
        assert params.group_size == -1

    def test_custom_params(self):
        """Test custom parameter values."""
        params = QuantParams(
            enabled=False,
            threshold_factor=0.1,
            bits=4,
            group_size=128,
        )
        assert params.enabled is False
        assert params.threshold_factor == 0.1
        assert params.bits == 4
        assert params.group_size == 128


class TestQuantConfig:
    """Tests for QuantConfig dataclass."""

    def test_default_config(self):
        """Test default config values."""
        config = QuantConfig()
        assert config.enabled is False
        assert config.method == "none"
        assert config.quantize_attention is True
        assert config.quantize_mlp is True
        assert config.quantize_head is False

    def test_enabled_config(self):
        """Test enabled quantization config."""
        config = QuantConfig(enabled=True, method="ternary")
        assert config.enabled is True
        assert config.method == "ternary"

    def test_to_dict(self):
        """Test serialization to dict."""
        config = QuantConfig(enabled=True, method="ternary")
        d = config.to_dict()
        assert d["enabled"] is True
        assert d["method"] == "ternary"

    def test_from_dict(self):
        """Test deserialization from dict."""
        d = {"enabled": True, "method": "ternary", "threshold_factor": 0.1}
        config = QuantConfig.from_dict(d)
        assert config.enabled is True
        assert config.method == "ternary"
        assert config.threshold_factor == 0.1

    def test_from_dict_ignores_unknown_keys(self):
        """Test that unknown keys are ignored."""
        d = {"enabled": True, "unknown_key": "value"}
        config = QuantConfig.from_dict(d)
        assert config.enabled is True

    def test_repr_disabled(self):
        """Test repr for disabled config."""
        config = QuantConfig()
        assert "enabled=False" in repr(config)

    def test_repr_enabled(self):
        """Test repr for enabled config."""
        config = QuantConfig(enabled=True, method="ternary")
        assert "enabled=True" in repr(config)
        assert "ternary" in repr(config)


class TestNoneQuantMethod:
    """Tests for NoneQuantMethod."""

    def test_is_available(self):
        """Test that none method is always available."""
        assert NoneQuantMethod.is_available()

    def test_create_linear(self):
        """Test creating standard linear layer."""
        layer = NoneQuantMethod.create_linear(64, 128, bias=True)
        assert isinstance(layer, nn.Linear)
        assert layer.in_features == 64
        assert layer.out_features == 128
        assert layer.bias is not None

    def test_create_linear_no_bias(self):
        """Test creating linear without bias."""
        layer = NoneQuantMethod.create_linear(64, 128, bias=False)
        assert layer.bias is None


class TestTernaryQuantMethod:
    """Tests for TernaryQuantMethod."""

    def test_quantize_weights_per_channel(self):
        """Test per-channel weight quantization."""
        weight = torch.randn(32, 64)
        params = QuantParams(per_channel=True, threshold_factor=0.05)

        quantized, metadata = TernaryQuantMethod.quantize_weights(weight, params)

        # Should be ternary values
        unique_values = quantized.unique()
        assert len(unique_values) <= 3
        assert all(v in [-1.0, 0.0, 1.0] for v in unique_values)

        # Should have scale
        assert "scale" in metadata
        assert metadata["scale"].shape[0] == 32  # per output channel

    def test_quantize_weights_global(self):
        """Test global weight quantization."""
        weight = torch.randn(32, 64)
        params = QuantParams(per_channel=False, threshold_factor=0.05)

        quantized, metadata = TernaryQuantMethod.quantize_weights(weight, params)

        # Should be ternary values
        unique_values = quantized.unique()
        assert len(unique_values) <= 3

        # Should have scalar scale
        assert "scale" in metadata

    def test_dequantize_weights(self):
        """Test weight dequantization."""
        weight = torch.randn(32, 64)
        params = QuantParams(per_channel=True)

        quantized, metadata = TernaryQuantMethod.quantize_weights(weight, params)
        dequantized = TernaryQuantMethod.dequantize_weights(quantized, metadata)

        # Dequantized should have same shape
        assert dequantized.shape == weight.shape

    @pytest.mark.skipif(
        not TernaryQuantMethod.is_available(),
        reason="BitTorch not installed"
    )
    def test_create_linear_with_bittorch(self):
        """Test creating TernaryLinear when BitTorch is available."""
        layer = TernaryQuantMethod.create_linear(64, 128)
        # Should be TernaryLinear from bittorch
        assert layer.__class__.__name__ == "TernaryLinear"


class TestMakeLinear:
    """Tests for make_linear factory function."""

    def test_no_config_returns_linear(self):
        """Test that no config returns standard linear."""
        layer = make_linear(64, 128)
        assert isinstance(layer, nn.Linear)

    def test_disabled_config_returns_linear(self):
        """Test that disabled config returns standard linear."""
        config = QuantConfig(enabled=False, method="ternary")
        layer = make_linear(64, 128, quant_config=config)
        assert isinstance(layer, nn.Linear)

    def test_none_method_returns_linear(self):
        """Test that 'none' method returns standard linear."""
        config = QuantConfig(enabled=True, method="none")
        layer = make_linear(64, 128, quant_config=config)
        assert isinstance(layer, nn.Linear)

    def test_layer_type_attention(self):
        """Test layer_type='attention' respects quantize_attention."""
        config = QuantConfig(
            enabled=True,
            method="none",  # Use none to avoid BitTorch dependency
            quantize_attention=True,
        )
        layer = make_linear(64, 128, quant_config=config, layer_type="attention")
        assert isinstance(layer, nn.Linear)

    def test_layer_type_attention_disabled(self):
        """Test layer_type='attention' when quantize_attention=False."""
        config = QuantConfig(
            enabled=True,
            method="ternary",
            quantize_attention=False,
        )
        # Should return standard linear since attention is disabled
        layer = make_linear(64, 128, quant_config=config, layer_type="attention")
        assert isinstance(layer, nn.Linear)

    def test_layer_type_mlp(self):
        """Test layer_type='mlp' respects quantize_mlp."""
        config = QuantConfig(
            enabled=True,
            method="ternary",
            quantize_mlp=False,
        )
        # Should return standard linear since mlp is disabled
        layer = make_linear(64, 128, quant_config=config, layer_type="mlp")
        assert isinstance(layer, nn.Linear)

    def test_layer_type_head(self):
        """Test layer_type='head' respects quantize_head."""
        config = QuantConfig(
            enabled=True,
            method="ternary",
            quantize_head=False,  # Default
        )
        # Should return standard linear since head is disabled by default
        layer = make_linear(64, 128, quant_config=config, layer_type="head")
        assert isinstance(layer, nn.Linear)

    @pytest.mark.skipif(
        TernaryQuantMethod.is_available(),
        reason="BitTorch is installed - cannot test unavailable method"
    )
    def test_unavailable_method_raises(self):
        """Test that unavailable method raises RuntimeError."""
        config = QuantConfig(enabled=True, method="ternary")
        with pytest.raises(RuntimeError, match="not available"):
            make_linear(64, 128, quant_config=config)

    @pytest.mark.skipif(
        not TernaryQuantMethod.is_available(),
        reason="BitTorch not installed"
    )
    def test_ternary_with_bittorch(self):
        """Test ternary quantization when BitTorch is available."""
        config = QuantConfig(enabled=True, method="ternary")
        layer = make_linear(64, 128, quant_config=config)
        assert layer.__class__.__name__ == "TernaryLinear"


class TestQuantMethodBaseClass:
    """Tests for QuantMethod base class defaults."""

    def test_default_quantize_weights(self):
        """Test default quantize_weights returns unchanged tensor."""
        weight = torch.randn(32, 64)
        # NoneQuantMethod inherits default behavior
        result, metadata = NoneQuantMethod.quantize_weights(weight)
        assert torch.equal(result, weight)
        assert metadata == {}

    def test_default_dequantize_weights(self):
        """Test default dequantize_weights returns unchanged tensor."""
        weight = torch.randn(32, 64)
        result = NoneQuantMethod.dequantize_weights(weight, {})
        assert torch.equal(result, weight)

    def test_default_quantize_activations(self):
        """Test default quantize_activations returns unchanged tensor."""
        x = torch.randn(8, 32, 64)
        result, metadata = NoneQuantMethod.quantize_activations(x)
        assert torch.equal(result, x)
        assert metadata == {}


class TestCustomRegistry:
    """Tests for creating custom registries."""

    def test_create_custom_registry(self):
        """Test creating a custom registry."""
        registry = QuantRegistry("custom_quant")
        assert registry.name == "custom_quant"
        assert registry.available() == []

    def test_register_method(self):
        """Test registering a method to custom registry."""
        registry = QuantRegistry("custom")

        @registry.register("test_method")
        class TestMethod(QuantMethod):
            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def create_linear(cls, in_features, out_features, bias=False, params=None):
                return nn.Linear(in_features, out_features, bias=bias)

        assert "test_method" in registry
        assert TestMethod.name == "test_method"

    def test_duplicate_registration_raises(self):
        """Test that duplicate registration raises ValueError."""
        registry = QuantRegistry("custom")

        @registry.register("method")
        class Method1(QuantMethod):
            @classmethod
            def is_available(cls):
                return True

            @classmethod
            def create_linear(cls, in_features, out_features, bias=False, params=None):
                return nn.Linear(in_features, out_features, bias=bias)

        with pytest.raises(ValueError, match="already registered"):
            @registry.register("method")
            class Method2(QuantMethod):
                @classmethod
                def is_available(cls):
                    return True

                @classmethod
                def create_linear(cls, in_features, out_features, bias=False, params=None):
                    return nn.Linear(in_features, out_features, bias=bias)
