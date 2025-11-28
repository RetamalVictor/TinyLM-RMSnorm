"""Integration tests for TinyLM + BitTorch ternary quantization."""

import sys
import pytest
import torch
import torch.nn as nn

# Add parent directory to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from tinylm import QuantConfig, make_linear, TinyLM, MHA, Block, build_sincos


# Skip all tests if bittorch is not available
pytestmark = pytest.mark.skipif(
    not pytest.importorskip("bittorch", reason="BitTorch not installed"),
    reason="BitTorch not installed"
)


class TestQuantConfig:
    """Test QuantConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        cfg = QuantConfig()
        assert cfg.enabled is False
        assert cfg.method == "none"
        assert cfg.threshold_factor == 0.05
        assert cfg.per_channel is True
        assert cfg.backend == "auto"
        assert cfg.quantize_attention is True
        assert cfg.quantize_mlp is True
        assert cfg.quantize_head is False

    def test_enabled_config(self):
        """Test enabled ternary configuration."""
        cfg = QuantConfig(enabled=True, method="ternary")
        assert cfg.enabled is True
        assert cfg.method == "ternary"

    def test_to_dict(self):
        """Test serialization to dictionary."""
        cfg = QuantConfig(enabled=True, threshold_factor=0.1)
        d = cfg.to_dict()
        assert d["enabled"] is True
        assert d["threshold_factor"] == 0.1
        assert "quantize_head" in d

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        d = {"enabled": True, "method": "ternary", "threshold_factor": 0.1}
        cfg = QuantConfig.from_dict(d)
        assert cfg.enabled is True
        assert cfg.method == "ternary"
        assert cfg.threshold_factor == 0.1
        # Defaults should be used for missing keys
        assert cfg.per_channel is True

    def test_from_dict_ignores_unknown_keys(self):
        """Test that unknown keys are ignored."""
        d = {"enabled": True, "unknown_key": "value"}
        cfg = QuantConfig.from_dict(d)
        assert cfg.enabled is True
        assert not hasattr(cfg, "unknown_key")


class TestMakeLinear:
    """Test make_linear factory function."""

    def test_disabled_returns_linear(self):
        """Test that disabled config returns nn.Linear."""
        cfg = QuantConfig(enabled=False)
        layer = make_linear(64, 32, quant_config=cfg)
        assert isinstance(layer, nn.Linear)

    def test_none_config_returns_linear(self):
        """Test that None config returns nn.Linear."""
        layer = make_linear(64, 32, quant_config=None)
        assert isinstance(layer, nn.Linear)

    def test_enabled_returns_ternary_linear(self):
        """Test that enabled config returns TernaryLinear."""
        from bittorch.nn import TernaryLinear
        cfg = QuantConfig(enabled=True, method="ternary")
        layer = make_linear(64, 32, quant_config=cfg)
        assert isinstance(layer, TernaryLinear)

    def test_layer_type_attention(self):
        """Test attention layer type respects quantize_attention flag."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, quantize_attention=True)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="attention")
        assert isinstance(layer, TernaryLinear)

        cfg = QuantConfig(enabled=True, quantize_attention=False)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="attention")
        assert isinstance(layer, nn.Linear)

    def test_layer_type_mlp(self):
        """Test MLP layer type respects quantize_mlp flag."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, quantize_mlp=True)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="mlp")
        assert isinstance(layer, TernaryLinear)

        cfg = QuantConfig(enabled=True, quantize_mlp=False)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="mlp")
        assert isinstance(layer, nn.Linear)

    def test_layer_type_head(self):
        """Test head layer type respects quantize_head flag."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, quantize_head=True)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="head")
        assert isinstance(layer, TernaryLinear)

        # By default, head should NOT be quantized
        cfg = QuantConfig(enabled=True)
        layer = make_linear(64, 32, quant_config=cfg, layer_type="head")
        assert isinstance(layer, nn.Linear)


class TestMHAWithQuant:
    """Test Multi-Head Attention with ternary quantization."""

    def test_mha_without_quant(self):
        """Test MHA works without quantization."""
        mha = MHA(dim=64, n_heads=4)
        x = torch.randn(2, 8, 64)
        sin, cos = build_sincos(16, 16, x.device)
        out = mha(x, sin, cos)
        assert out.shape == (2, 8, 64)

    def test_mha_with_quant(self):
        """Test MHA works with ternary quantization."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, method="ternary")
        mha = MHA(dim=64, n_heads=4, quant_config=cfg)

        # Verify layers are TernaryLinear
        assert isinstance(mha.qkv, TernaryLinear)
        assert isinstance(mha.proj, TernaryLinear)

        x = torch.randn(2, 8, 64)
        sin, cos = build_sincos(16, 16, x.device)
        out = mha(x, sin, cos)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()


class TestBlockWithQuant:
    """Test Transformer Block with ternary quantization."""

    def test_block_with_quant(self):
        """Test Block works with ternary quantization."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, method="ternary")
        block = Block(dim=64, n_heads=4, quant_config=cfg)

        # Verify attention layers are TernaryLinear
        assert isinstance(block.attn.qkv, TernaryLinear)
        assert isinstance(block.attn.proj, TernaryLinear)

        # Verify MLP layers are TernaryLinear
        assert isinstance(block.mlp[0], TernaryLinear)  # fc1
        assert isinstance(block.mlp[3], TernaryLinear)  # fc2

        x = torch.randn(2, 8, 64)
        sin, cos = build_sincos(16, 16, x.device)
        out = block(x, sin, cos)
        assert out.shape == (2, 8, 64)
        assert not torch.isnan(out).any()


class TestTinyLMWithQuant:
    """Test TinyLM model with ternary quantization."""

    def test_model_without_quant(self):
        """Test TinyLM works without quantization."""
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4)
        idx = torch.randint(0, 100, (2, 8))
        sin, cos = build_sincos(16, 16, idx.device)
        logits = model(idx, sin, cos)
        assert logits.shape == (2, 8, 100)

    def test_model_with_quant(self):
        """Test TinyLM works with ternary quantization."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, method="ternary")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg)

        # Verify quant_config is stored
        assert model.quant_config is cfg

        # Verify head is nn.Linear (not quantized by default)
        assert isinstance(model.head, nn.Linear)

        # Verify attention layers are quantized
        assert isinstance(model.blocks[0].attn.qkv, TernaryLinear)

        idx = torch.randint(0, 100, (2, 8))
        sin, cos = build_sincos(16, 16, idx.device)
        logits = model(idx, sin, cos)
        assert logits.shape == (2, 8, 100)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_model_with_quantized_head(self):
        """Test TinyLM with quantized output head."""
        from bittorch.nn import TernaryLinear

        cfg = QuantConfig(enabled=True, method="ternary", quantize_head=True)
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg)

        # Verify head is TernaryLinear
        assert isinstance(model.head, TernaryLinear)


class TestGradientFlow:
    """Test gradient flow through quantized layers."""

    def test_gradient_flow_mha(self):
        """Test gradients flow through quantized MHA."""
        cfg = QuantConfig(enabled=True, method="ternary")
        mha = MHA(dim=64, n_heads=4, quant_config=cfg)

        x = torch.randn(2, 8, 64, requires_grad=True)
        sin, cos = build_sincos(16, 16, x.device)
        out = mha(x, sin, cos)
        loss = out.sum()
        loss.backward()

        # Check gradients exist
        assert x.grad is not None
        assert mha.qkv.weight.grad is not None
        assert mha.proj.weight.grad is not None

    def test_gradient_flow_model(self):
        """Test gradients flow through quantized TinyLM."""
        cfg = QuantConfig(enabled=True, method="ternary")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg)

        idx = torch.randint(0, 100, (2, 8))
        targets = torch.randint(0, 100, (2, 8))
        sin, cos = build_sincos(16, 16, idx.device)

        logits = model(idx, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, 100), targets.view(-1))
        loss.backward()

        # Check gradients exist for key layers
        assert model.tok.weight.grad is not None
        assert model.blocks[0].attn.qkv.weight.grad is not None
        assert model.head.weight.grad is not None


class TestTrainingStep:
    """Test a single training step with quantized model."""

    def test_single_training_step(self):
        """Test model can complete a training step."""
        cfg = QuantConfig(enabled=True, method="ternary")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        idx = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))
        sin, cos = build_sincos(32, 16, idx.device)

        # Training step
        optimizer.zero_grad()
        logits = model(idx, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, 100), targets.view(-1))
        loss.backward()
        optimizer.step()

        # Loss should be finite
        assert torch.isfinite(loss)

    def test_loss_decreases(self):
        """Test loss decreases over multiple training steps."""
        cfg = QuantConfig(enabled=True, method="ternary")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Fixed batch for consistent comparison
        torch.manual_seed(42)
        idx = torch.randint(0, 100, (4, 16))
        targets = torch.randint(0, 100, (4, 16))
        sin, cos = build_sincos(32, 16, idx.device)

        losses = []
        for _ in range(10):
            optimizer.zero_grad()
            logits = model(idx, sin, cos)
            loss = nn.functional.cross_entropy(logits.view(-1, 100), targets.view(-1))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should generally decrease (allowing some fluctuation)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDA:
    """Test ternary quantization on CUDA."""

    def test_model_on_cuda(self):
        """Test quantized model works on CUDA."""
        cfg = QuantConfig(enabled=True, method="ternary", backend="auto")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg).cuda()

        idx = torch.randint(0, 100, (2, 8)).cuda()
        sin, cos = build_sincos(16, 16, idx.device)

        logits = model(idx, sin, cos)
        assert logits.device.type == "cuda"
        assert not torch.isnan(logits).any()

    def test_training_on_cuda(self):
        """Test training step on CUDA."""
        cfg = QuantConfig(enabled=True, method="ternary", backend="auto")
        model = TinyLM(vocab_size=100, dim=64, n_layers=2, n_heads=4, quant_config=cfg).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        idx = torch.randint(0, 100, (4, 16)).cuda()
        targets = torch.randint(0, 100, (4, 16)).cuda()
        sin, cos = build_sincos(32, 16, idx.device)

        optimizer.zero_grad()
        logits = model(idx, sin, cos)
        loss = nn.functional.cross_entropy(logits.view(-1, 100), targets.view(-1))
        loss.backward()
        optimizer.step()

        assert torch.isfinite(loss)
