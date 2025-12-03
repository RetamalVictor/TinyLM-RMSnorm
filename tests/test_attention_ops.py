"""Tests for AttentionOp implementations."""

import pytest
import torch
import torch.nn as nn

from tinylm.components.attention.ops import (
    AttentionOp,
    StandardAttentionOp,
    FlashAttentionOp,
    MemoryEfficientAttentionOp,
    build_attention_op,
    available_attention_ops,
)
from tinylm.components.registry import ATTENTION_OP_REGISTRY


class TestAttentionOpRegistry:
    """Tests for attention op registry."""

    def test_registry_has_standard(self):
        """Test that standard attention is registered."""
        assert "standard" in ATTENTION_OP_REGISTRY
        assert "standard" in available_attention_ops()

    def test_registry_has_flash(self):
        """Test that flash attention is registered."""
        assert "flash" in ATTENTION_OP_REGISTRY
        assert "flash" in available_attention_ops()

    def test_registry_has_memory_efficient(self):
        """Test that memory_efficient attention is registered."""
        assert "memory_efficient" in ATTENTION_OP_REGISTRY
        assert "memory_efficient" in available_attention_ops()

    def test_build_standard(self):
        """Test building standard attention op."""
        op = build_attention_op("standard")
        assert isinstance(op, StandardAttentionOp)

    def test_build_flash(self):
        """Test building flash attention op."""
        op = build_attention_op("flash")
        assert isinstance(op, FlashAttentionOp)

    def test_build_memory_efficient(self):
        """Test building memory_efficient attention op."""
        op = build_attention_op("memory_efficient")
        assert isinstance(op, MemoryEfficientAttentionOp)

    def test_build_unknown_raises(self):
        """Test that unknown op type raises error."""
        with pytest.raises(ValueError, match="Unknown"):
            build_attention_op("unknown_op")

    def test_build_with_dropout(self):
        """Test building with custom dropout."""
        op = build_attention_op("standard", dropout=0.1)
        assert op.dropout == 0.1

    def test_build_with_scale(self):
        """Test building with custom scale."""
        op = build_attention_op("standard", scale=0.5)
        assert op.scale == 0.5


class TestStandardAttentionOp:
    """Tests for StandardAttentionOp."""

    def test_is_available(self):
        """Test that standard attention is always available."""
        assert StandardAttentionOp.is_available()

    def test_forward_basic(self):
        """Test basic forward pass."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = op(q, k, v)

        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_forward_causal(self):
        """Test causal attention."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = op(q, k, v, is_causal=True)

        assert out.shape == (B, H, T, D)

    def test_forward_non_causal(self):
        """Test non-causal attention."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = op(q, k, v, is_causal=False)

        assert out.shape == (B, H, T, D)

    def test_forward_with_kv_cache(self):
        """Test attention with longer K/V (simulating KV cache)."""
        op = StandardAttentionOp()
        B, H, T_q, T_kv, D = 2, 4, 1, 16, 16
        q = torch.randn(B, H, T_q, D)
        k = torch.randn(B, H, T_kv, D)
        v = torch.randn(B, H, T_kv, D)

        # Single query, multiple K/V (generation with cache)
        out = op(q, k, v, is_causal=False)

        assert out.shape == (B, H, T_q, D)

    def test_forward_with_attn_mask(self):
        """Test attention with explicit mask."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        # Additive attention bias
        attn_mask = torch.zeros(H, T, T)

        out = op(q, k, v, attn_mask=attn_mask, is_causal=False)

        assert out.shape == (B, H, T, D)

    def test_forward_different_dtypes(self):
        """Test attention with different data types."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16

        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            q = torch.randn(B, H, T, D, dtype=dtype)
            k = torch.randn(B, H, T, D, dtype=dtype)
            v = torch.randn(B, H, T, D, dtype=dtype)

            out = op(q, k, v)

            assert out.dtype == dtype
            assert out.shape == (B, H, T, D)

    def test_forward_with_dropout_training(self):
        """Test attention with dropout in training mode."""
        op = StandardAttentionOp(dropout=0.5)
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        # With training=True, dropout should be applied
        out = op(q, k, v, training=True)
        assert out.shape == (B, H, T, D)

        # With training=False, no dropout
        out = op(q, k, v, training=False)
        assert out.shape == (B, H, T, D)


class TestFlashAttentionOp:
    """Tests for FlashAttentionOp."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        op = FlashAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = op(q, k, v)

        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()

    def test_forward_with_mask_fallback(self):
        """Test that flash attention falls back with explicit mask."""
        op = FlashAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)
        attn_mask = torch.zeros(H, T, T)

        # Should fallback gracefully with mask
        out = op(q, k, v, attn_mask=attn_mask)

        assert out.shape == (B, H, T, D)


class TestMemoryEfficientAttentionOp:
    """Tests for MemoryEfficientAttentionOp."""

    def test_forward_basic(self):
        """Test basic forward pass."""
        op = MemoryEfficientAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out = op(q, k, v)

        assert out.shape == (B, H, T, D)
        assert not torch.isnan(out).any()


class TestAttentionOpEquivalence:
    """Test that different attention ops produce equivalent results."""

    def test_standard_vs_flash_equivalence(self):
        """Test that standard and flash produce same results (when flash falls back)."""
        standard_op = StandardAttentionOp()
        flash_op = FlashAttentionOp()

        B, H, T, D = 2, 4, 8, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_standard = standard_op(q, k, v, is_causal=True)
        out_flash = flash_op(q, k, v, is_causal=True)

        # Should be close (exact match depends on kernel)
        assert torch.allclose(out_standard, out_flash, atol=1e-4)

    def test_standard_vs_memory_efficient_equivalence(self):
        """Test that standard and memory_efficient produce same results."""
        standard_op = StandardAttentionOp()
        mem_op = MemoryEfficientAttentionOp()

        B, H, T, D = 2, 4, 8, 16
        torch.manual_seed(42)
        q = torch.randn(B, H, T, D)
        k = torch.randn(B, H, T, D)
        v = torch.randn(B, H, T, D)

        out_standard = standard_op(q, k, v, is_causal=True)
        out_mem = mem_op(q, k, v, is_causal=True)

        # Should be close (exact match depends on kernel)
        assert torch.allclose(out_standard, out_mem, atol=1e-4)


class TestMHAWithAttentionOp:
    """Test MHA with different attention ops."""

    def test_mha_default_attention_op(self):
        """Test MHA uses standard attention by default."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=64, n_heads=4)
        assert isinstance(mha.attention_op, StandardAttentionOp)

    def test_mha_with_flash_attention(self):
        """Test MHA with flash attention op."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=64, n_heads=4, attention_op="flash")
        assert isinstance(mha.attention_op, FlashAttentionOp)

    def test_mha_with_memory_efficient_attention(self):
        """Test MHA with memory_efficient attention op."""
        from tinylm.components.attention import MHA

        mha = MHA(dim=64, n_heads=4, attention_op="memory_efficient")
        assert isinstance(mha.attention_op, MemoryEfficientAttentionOp)

    def test_mha_forward_with_different_ops(self):
        """Test MHA forward works with all attention ops."""
        from tinylm.components.attention import MHA
        from tinylm.components.positional import RoPE
        from tinylm.components.positional.base import PositionalContext

        B, T, D = 2, 8, 64
        n_heads = 4
        head_dim = D // n_heads

        for op_type in ["standard", "flash", "memory_efficient"]:
            mha = MHA(dim=D, n_heads=n_heads, attention_op=op_type)

            # Setup positional embedding
            rope = RoPE(dim=head_dim, max_seq_len=32)
            cache = rope.precompute(32, torch.device("cpu"))
            mha.set_pos_emb(rope)

            pos_ctx = PositionalContext(
                seq_len=T,
                start_pos=0,
                sin=cache["sin"],
                cos=cache["cos"],
                device=torch.device("cpu"),
            )

            x = torch.randn(B, T, D)
            out = mha(x, pos_ctx)

            assert out.shape == (B, T, D)
            assert not torch.isnan(out).any()

    def test_mha_swap_attention_op_runtime(self):
        """Test swapping attention op at runtime."""
        from tinylm.components.attention import MHA, build_attention_op

        mha = MHA(dim=64, n_heads=4, attention_op="standard")
        assert isinstance(mha.attention_op, StandardAttentionOp)

        # Swap at runtime
        mha.attention_op = build_attention_op("flash")
        assert isinstance(mha.attention_op, FlashAttentionOp)


class TestGradientFlow:
    """Test gradient flow through attention ops."""

    def test_gradient_flow_standard(self):
        """Test gradients flow through standard attention."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)

        out = op(q, k, v)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None

    def test_gradient_flow_flash(self):
        """Test gradients flow through flash attention."""
        op = FlashAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D, requires_grad=True)
        k = torch.randn(B, H, T, D, requires_grad=True)
        v = torch.randn(B, H, T, D, requires_grad=True)

        out = op(q, k, v)
        loss = out.sum()
        loss.backward()

        assert q.grad is not None
        assert k.grad is not None
        assert v.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestCUDAAttentionOps:
    """Test attention ops on CUDA."""

    def test_standard_on_cuda(self):
        """Test standard attention on CUDA."""
        op = StandardAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D, device="cuda")
        k = torch.randn(B, H, T, D, device="cuda")
        v = torch.randn(B, H, T, D, device="cuda")

        out = op(q, k, v)

        assert out.device.type == "cuda"
        assert out.shape == (B, H, T, D)

    def test_flash_on_cuda(self):
        """Test flash attention on CUDA."""
        op = FlashAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D, device="cuda")
        k = torch.randn(B, H, T, D, device="cuda")
        v = torch.randn(B, H, T, D, device="cuda")

        out = op(q, k, v)

        assert out.device.type == "cuda"
        assert out.shape == (B, H, T, D)

    def test_memory_efficient_on_cuda(self):
        """Test memory_efficient attention on CUDA."""
        op = MemoryEfficientAttentionOp()
        B, H, T, D = 2, 4, 8, 16
        q = torch.randn(B, H, T, D, device="cuda")
        k = torch.randn(B, H, T, D, device="cuda")
        v = torch.randn(B, H, T, D, device="cuda")

        out = op(q, k, v)

        assert out.device.type == "cuda"
        assert out.shape == (B, H, T, D)
