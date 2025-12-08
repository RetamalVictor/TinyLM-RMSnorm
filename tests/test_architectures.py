"""Tests for multi-architecture support in TinyLM."""

import pytest
import torch

from tinylm import TinyLM
from tinylm.architectures import (
    ArchitectureConfig,
    get_architecture,
    list_architectures,
    register_architecture,
)
from tinylm.components import (
    PositionalContext,
    build_activation,
    build_attention,
    build_mlp,
    build_norm,
    build_pos_emb,
)
from tinylm.inference.cache_manager import StandardCache
from tinylm.model.blocks import PostNormBlock, PreNormBlock, build_block


# Test fixtures
@pytest.fixture
def small_config():
    return {
        "vocab_size": 1000,
        "dim": 128,
        "n_layers": 2,
        "n_heads": 4,
        "max_seq_len": 256,
    }


@pytest.fixture
def batch_input():
    return torch.randint(0, 1000, (2, 32))


class TestArchitectureRegistry:
    """Tests for architecture configuration and registry."""

    def test_list_architectures(self):
        archs = list_architectures()
        assert "llama" in archs
        assert "gpt" in archs

    def test_get_llama_architecture(self):
        cfg = get_architecture("llama")
        assert cfg.name == "llama"
        assert cfg.norm_type == "rmsnorm"
        assert cfg.norm_position == "pre"
        assert cfg.pos_emb_type == "rope"
        assert cfg.mlp_type == "gated"
        assert cfg.activation == "silu"
        assert cfg.use_bias is False

    def test_get_gpt_architecture(self):
        cfg = get_architecture("gpt")
        assert cfg.name == "gpt"
        assert cfg.norm_type == "layernorm"
        assert cfg.norm_position == "post"
        assert cfg.pos_emb_type == "learned"
        assert cfg.mlp_type == "standard"
        assert cfg.activation == "gelu"
        assert cfg.use_bias is True

    def test_unknown_architecture_raises(self):
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_architecture("unknown")

    def test_architecture_config_serialization(self):
        cfg = get_architecture("llama")
        d = cfg.to_dict()
        restored = ArchitectureConfig.from_dict(d)
        assert restored.name == cfg.name
        assert restored.norm_type == cfg.norm_type
        assert restored.pos_emb_type == cfg.pos_emb_type

    def test_register_custom_architecture(self):
        custom = ArchitectureConfig(
            name="custom",
            norm_type="layernorm",
            norm_position="pre",
            pos_emb_type="rope",
            mlp_type="standard",
            activation="gelu",
        )
        register_architecture("custom", custom)
        assert "custom" in list_architectures()
        retrieved = get_architecture("custom")
        assert retrieved.name == "custom"


class TestComponentFactories:
    """Tests for component factory functions."""

    def test_build_rmsnorm(self):
        norm = build_norm("rmsnorm", dim=128)
        x = torch.randn(2, 32, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_build_layernorm(self):
        norm = build_norm("layernorm", dim=128)
        x = torch.randn(2, 32, 128)
        out = norm(x)
        assert out.shape == x.shape

    def test_build_rope(self):
        rope = build_pos_emb("rope", dim=32, max_seq_len=256)
        cache = rope.precompute(256, torch.device("cpu"))
        assert "sin" in cache
        assert "cos" in cache
        # Shape is (1, 1, seq_len, head_dim) for broadcasting
        assert cache["sin"].shape[-2:] == (256, 32)

    def test_build_learned_pos_emb(self):
        pos_emb = build_pos_emb("learned", dim=128, max_seq_len=256)
        cache = pos_emb.precompute(256, torch.device("cpu"))
        assert "positions" in cache

        ctx = PositionalContext(seq_len=32, start_pos=0)
        ctx.positions = cache["positions"]
        x = torch.randn(2, 32, 128)
        out = pos_emb.apply(x, ctx)
        # Learned embeddings return (1, seq, dim) for broadcasting
        assert out.shape[-2:] == (32, 128)

    def test_build_mha(self):
        attn = build_attention("mha", dim=128, n_heads=4)
        x = torch.randn(2, 32, 128)
        ctx = PositionalContext(seq_len=32, start_pos=0)
        out = attn(x, ctx)
        assert out.shape == x.shape

    def test_build_gated_mlp(self):
        mlp = build_mlp("gated", dim=128, hidden_ratio=4.0, activation="silu")
        x = torch.randn(2, 32, 128)
        out = mlp(x)
        assert out.shape == x.shape

    def test_build_standard_mlp(self):
        mlp = build_mlp("standard", dim=128, hidden_ratio=4.0, activation="gelu")
        x = torch.randn(2, 32, 128)
        out = mlp(x)
        assert out.shape == x.shape

    def test_build_activations(self):
        for act_name in ["silu", "gelu", "relu"]:
            act = build_activation(act_name)
            x = torch.randn(2, 32, 128)
            out = act(x)
            assert out.shape == x.shape


class TestTransformerBlocks:
    """Tests for transformer block implementations."""

    def test_prenorm_block_shape(self):
        block = PreNormBlock(dim=128, n_heads=4)
        x = torch.randn(2, 32, 128)
        ctx = PositionalContext(seq_len=32, start_pos=0)
        out = block(x, ctx)
        assert out.shape == x.shape

    def test_postnorm_block_shape(self):
        block = PostNormBlock(dim=128, n_heads=4)
        x = torch.randn(2, 32, 128)
        ctx = PositionalContext(seq_len=32, start_pos=0)
        out = block(x, ctx)
        assert out.shape == x.shape

    def test_build_block_pre(self):
        block = build_block("pre", dim=128, n_heads=4)
        assert isinstance(block, PreNormBlock)

    def test_build_block_post(self):
        block = build_block("post", dim=128, n_heads=4)
        assert isinstance(block, PostNormBlock)

    def test_block_with_kv_cache(self):
        block = PreNormBlock(dim=128, n_heads=4)
        x = torch.randn(2, 32, 128)
        ctx = PositionalContext(seq_len=32, start_pos=0)

        # Create cache using CacheManager
        cache = StandardCache(n_layers=1, n_heads=4, head_dim=32)
        cache.allocate(batch_size=2, max_seq_len=64)

        out = block(x, ctx, cache=cache, layer_idx=0, start_pos=0)
        assert out.shape == x.shape


class TestTinyLMModel:
    """Tests for the main TinyLM model class."""

    def test_create_llama_model(self, small_config):
        model = TinyLM(**small_config, architecture="llama")
        assert model.arch.name == "llama"
        assert model.arch.norm_type == "rmsnorm"

    def test_create_gpt_model(self, small_config):
        model = TinyLM(**small_config, architecture="gpt")
        assert model.arch.name == "gpt"
        assert model.arch.norm_type == "layernorm"

    def test_llama_forward_shape(self, small_config, batch_input):
        model = TinyLM(**small_config, architecture="llama")
        out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])

    def test_gpt_forward_shape(self, small_config, batch_input):
        model = TinyLM(**small_config, architecture="gpt")
        out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])

    def test_forward_with_custom_arch_config(self, small_config, batch_input):
        cfg = ArchitectureConfig(
            name="custom",
            norm_type="layernorm",
            norm_position="pre",
            pos_emb_type="rope",
            mlp_type="gated",
            activation="gelu",
        )
        model = TinyLM(**small_config, arch_config=cfg)
        out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])

    def test_create_kv_cache(self, small_config):
        model = TinyLM(**small_config, architecture="llama")
        cache = model.create_kv_cache(batch_size=2, max_seq_len=64)

        assert isinstance(cache, StandardCache)
        assert cache.n_layers == small_config["n_layers"]
        assert cache.is_allocated
        assert cache.batch_size == 2
        assert cache.max_seq_len == 64

    def test_forward_with_kv_cache(self, small_config):
        model = TinyLM(**small_config, architecture="llama")
        cache = model.create_kv_cache(batch_size=2, max_seq_len=64)

        # First forward (prompt)
        x = torch.randint(0, 1000, (2, 16))
        out1 = model(x, cache=cache, start_pos=0)
        assert out1.shape == (2, 16, small_config["vocab_size"])

        # Second forward (single token generation)
        x2 = torch.randint(0, 1000, (2, 1))
        out2 = model(x2, cache=cache, start_pos=16)
        assert out2.shape == (2, 1, small_config["vocab_size"])

    def test_get_num_params(self, small_config):
        llama = TinyLM(**small_config, architecture="llama")
        gpt = TinyLM(**small_config, architecture="gpt")

        llama_params = llama.get_num_params()
        gpt_params = gpt.get_num_params()

        assert llama_params > 0
        assert gpt_params > 0
        # LLaMA has more params due to gated MLP (3 projections vs 2)
        # GPT has learned pos embeddings and biases but fewer MLP params
        assert llama_params != gpt_params  # They should differ


class TestModelShapeConsistency:
    """Tests to ensure shape consistency across different configurations."""

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_various_batch_sizes(self, arch):
        model = TinyLM(vocab_size=1000, dim=128, n_layers=2, n_heads=4, architecture=arch)
        for batch_size in [1, 2, 4, 8]:
            x = torch.randint(0, 1000, (batch_size, 32))
            out = model(x)
            assert out.shape == (batch_size, 32, 1000)

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_various_seq_lengths(self, arch):
        model = TinyLM(vocab_size=1000, dim=128, n_layers=2, n_heads=4, architecture=arch)
        for seq_len in [1, 16, 32, 64, 128]:
            x = torch.randint(0, 1000, (2, seq_len))
            out = model(x)
            assert out.shape == (2, seq_len, 1000)

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_various_dims(self, arch):
        for dim, n_heads in [(64, 2), (128, 4), (256, 8)]:
            model = TinyLM(
                vocab_size=1000, dim=dim, n_layers=2, n_heads=n_heads, architecture=arch
            )
            x = torch.randint(0, 1000, (2, 32))
            out = model(x)
            assert out.shape == (2, 32, 1000)

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_various_layer_counts(self, arch):
        for n_layers in [1, 2, 4, 6]:
            model = TinyLM(
                vocab_size=1000, dim=128, n_layers=n_layers, n_heads=4, architecture=arch
            )
            x = torch.randint(0, 1000, (2, 32))
            out = model(x)
            assert out.shape == (2, 32, 1000)


class TestGradientFlow:
    """Tests to ensure gradients flow properly."""

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_backward_pass(self, arch):
        model = TinyLM(vocab_size=1000, dim=128, n_layers=2, n_heads=4, architecture=arch)
        x = torch.randint(0, 1000, (2, 32))
        y = torch.randint(0, 1000, (2, 32))

        out = model(x)
        loss = torch.nn.functional.cross_entropy(
            out.view(-1, 1000), y.view(-1)
        )
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_no_nan_in_forward(self, arch):
        model = TinyLM(vocab_size=1000, dim=128, n_layers=2, n_heads=4, architecture=arch)
        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        assert not torch.isnan(out).any(), "NaN in forward pass"
        assert not torch.isinf(out).any(), "Inf in forward pass"


class TestInference:
    """Tests for inference utilities."""

    def test_sample_top_p(self):
        from tinylm.inference import sample_top_p

        logits = torch.randn(2, 1000)
        tokens = sample_top_p(logits, top_p=0.9)
        assert tokens.shape == (2, 1)
        assert tokens.dtype == torch.long

    def test_generate_import(self):
        from tinylm.inference import generate
        assert callable(generate)


class TestGQAMQA:
    """Tests for Grouped-Query Attention (GQA) and Multi-Query Attention (MQA)."""

    def test_mha_n_kv_heads_equals_n_heads(self):
        """Standard MHA: n_kv_heads == n_heads."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=128, n_heads=8, n_kv_heads=8)
        assert mha.n_kv_heads == 8
        assert mha.n_rep == 1  # No repetition needed

        x = torch.randn(2, 16, 128)
        pos_ctx = PositionalContext(seq_len=16, start_pos=0, device=x.device)
        out = mha(x, pos_ctx)
        assert out.shape == (2, 16, 128)

    def test_gqa_n_kv_heads_less_than_n_heads(self):
        """GQA: n_kv_heads < n_heads (grouped)."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=128, n_heads=8, n_kv_heads=2)
        assert mha.n_kv_heads == 2
        assert mha.n_rep == 4  # Each KV head serves 4 Q heads

        x = torch.randn(2, 16, 128)
        pos_ctx = PositionalContext(seq_len=16, start_pos=0, device=x.device)
        out = mha(x, pos_ctx)
        assert out.shape == (2, 16, 128)

    def test_mqa_n_kv_heads_equals_one(self):
        """MQA: n_kv_heads == 1 (single KV head for all Q heads)."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=128, n_heads=8, n_kv_heads=1)
        assert mha.n_kv_heads == 1
        assert mha.n_rep == 8  # Single KV head serves all 8 Q heads

        x = torch.randn(2, 16, 128)
        pos_ctx = PositionalContext(seq_len=16, start_pos=0, device=x.device)
        out = mha(x, pos_ctx)
        assert out.shape == (2, 16, 128)

    def test_gqa_projection_sizes(self):
        """Verify GQA uses smaller KV projections."""
        from tinylm.components.attention import MHA

        mha_full = MHA(dim=128, n_heads=8, n_kv_heads=8)
        mha_gqa = MHA(dim=128, n_heads=8, n_kv_heads=2)

        # Q projection same size
        assert mha_full.q_proj.weight.shape == mha_gqa.q_proj.weight.shape

        # KV projection smaller for GQA (2 heads vs 8 heads)
        # kv_proj is [2 * n_kv_heads * head_dim, dim]
        assert mha_full.kv_proj.weight.shape[0] == 2 * 8 * 16  # 256
        assert mha_gqa.kv_proj.weight.shape[0] == 2 * 2 * 16   # 64

    def test_gqa_model_forward(self):
        """Test TinyLM with GQA architecture."""
        cfg = ArchitectureConfig(
            name="gqa_test",
            attention_type="gqa",
            n_kv_heads=2,  # 2 KV heads for 4 Q heads
        )
        model = TinyLM(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4,
            arch_config=cfg
        )
        assert model.n_kv_heads == 2

        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 1000)

    def test_mqa_model_forward(self):
        """Test TinyLM with MQA architecture."""
        cfg = ArchitectureConfig(
            name="mqa_test",
            attention_type="mqa",
            n_kv_heads=1,  # Single KV head
        )
        model = TinyLM(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4,
            arch_config=cfg
        )
        assert model.n_kv_heads == 1

        x = torch.randint(0, 1000, (2, 32))
        out = model(x)
        assert out.shape == (2, 32, 1000)

    def test_gqa_with_kv_cache(self):
        """Test GQA with KV cache for generation."""
        cfg = ArchitectureConfig(
            name="gqa_cache_test",
            attention_type="gqa",
            n_kv_heads=2,
        )
        model = TinyLM(
            vocab_size=1000, dim=128, n_layers=2, n_heads=4,
            arch_config=cfg
        )
        cache = model.create_kv_cache(batch_size=2, max_seq_len=64)

        # Verify cache uses n_kv_heads (memory savings)
        assert cache._n_kv_heads == 2

        # Prefill
        x1 = torch.randint(0, 1000, (2, 16))
        out1 = model(x1, cache=cache, start_pos=0)
        assert out1.shape == (2, 16, 1000)

        # Generate
        x2 = torch.randint(0, 1000, (2, 1))
        out2 = model(x2, cache=cache, start_pos=16)
        assert out2.shape == (2, 1, 1000)

    def test_gqa_gradient_flow(self):
        """Test gradients flow through GQA."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=128, n_heads=8, n_kv_heads=2)
        x = torch.randn(2, 16, 128, requires_grad=True)
        pos_ctx = PositionalContext(seq_len=16, start_pos=0, device=x.device)

        out = mha(x, pos_ctx)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None
        assert mha.q_proj.weight.grad is not None
        assert mha.kv_proj.weight.grad is not None
        assert mha.proj.weight.grad is not None

    def test_gqa_param_count_reduction(self):
        """Verify GQA reduces parameter count."""
        from tinylm.components.attention import MHA

        mha_full = MHA(dim=256, n_heads=8, n_kv_heads=8)
        mha_gqa = MHA(dim=256, n_heads=8, n_kv_heads=2)
        mha_mqa = MHA(dim=256, n_heads=8, n_kv_heads=1)

        params_full = sum(p.numel() for p in mha_full.parameters())
        params_gqa = sum(p.numel() for p in mha_gqa.parameters())
        params_mqa = sum(p.numel() for p in mha_mqa.parameters())

        # GQA and MQA should have fewer parameters
        assert params_gqa < params_full
        assert params_mqa < params_gqa


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
