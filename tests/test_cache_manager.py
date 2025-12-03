"""Tests for KV Cache Manager implementations."""

import pytest
import torch

from tinylm.inference.cache_manager import (
    CacheManager,
    StandardCache,
)


class TestStandardCacheAllocation:
    """Tests for cache allocation behavior."""

    def test_init_not_allocated(self):
        """Cache should not be allocated after init."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        assert not cache.is_allocated

    def test_allocate_sets_allocated_flag(self):
        """Allocate should set is_allocated to True."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        cache.allocate(batch_size=2, max_seq_len=128)
        assert cache.is_allocated

    def test_allocate_stores_dimensions(self):
        """Allocate should store batch_size and max_seq_len."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        cache.allocate(batch_size=2, max_seq_len=128)
        assert cache.batch_size == 2
        assert cache.max_seq_len == 128

    def test_n_layers_property(self):
        """n_layers should return configured value."""
        cache = StandardCache(n_layers=6, n_heads=8, head_dim=64)
        assert cache.n_layers == 6

    def test_reset_deallocates(self):
        """Reset should deallocate cache."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        cache.allocate(batch_size=2, max_seq_len=128)
        assert cache.is_allocated
        cache.reset()
        assert not cache.is_allocated


class TestStandardCacheUpdateGet:
    """Tests for cache update and get operations."""

    @pytest.fixture
    def cache(self):
        """Create and allocate a standard cache."""
        c = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        c.allocate(batch_size=2, max_seq_len=128)
        return c

    def test_update_and_get_single_token(self, cache):
        """Update with single token and retrieve."""
        # Create KV for single token: [B=2, H=8, T=1, D=64]
        k = torch.randn(2, 8, 1, 64)
        v = torch.randn(2, 8, 1, 64)

        cache.update(layer_idx=0, k=k, v=v, start_pos=0)
        k_out, v_out = cache.get(layer_idx=0, end_pos=1)

        assert k_out.shape == (2, 8, 1, 64)
        assert v_out.shape == (2, 8, 1, 64)
        assert torch.allclose(k_out, k)
        assert torch.allclose(v_out, v)

    def test_update_and_get_sequence(self, cache):
        """Update with sequence and retrieve."""
        # Create KV for 10 tokens: [B=2, H=8, T=10, D=64]
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)

        cache.update(layer_idx=0, k=k, v=v, start_pos=0)
        k_out, v_out = cache.get(layer_idx=0, end_pos=10)

        assert k_out.shape == (2, 8, 10, 64)
        assert torch.allclose(k_out, k)
        assert torch.allclose(v_out, v)

    def test_incremental_update(self, cache):
        """Test incremental updates (autoregressive pattern)."""
        # First: prefill with 5 tokens
        k1 = torch.randn(2, 8, 5, 64)
        v1 = torch.randn(2, 8, 5, 64)
        cache.update(layer_idx=0, k=k1, v=v1, start_pos=0)

        # Second: add 1 token at position 5
        k2 = torch.randn(2, 8, 1, 64)
        v2 = torch.randn(2, 8, 1, 64)
        cache.update(layer_idx=0, k=k2, v=v2, start_pos=5)

        # Get all 6 tokens
        k_out, v_out = cache.get(layer_idx=0, end_pos=6)

        assert k_out.shape == (2, 8, 6, 64)
        assert torch.allclose(k_out[:, :, :5, :], k1)
        assert torch.allclose(k_out[:, :, 5:6, :], k2)

    def test_multiple_layers(self, cache):
        """Test updates across multiple layers."""
        for layer_idx in range(4):
            k = torch.full((2, 8, 3, 64), float(layer_idx))
            v = torch.full((2, 8, 3, 64), float(layer_idx) + 0.5)
            cache.update(layer_idx=layer_idx, k=k, v=v, start_pos=0)

        for layer_idx in range(4):
            k_out, v_out = cache.get(layer_idx=layer_idx, end_pos=3)
            assert torch.allclose(k_out, torch.full_like(k_out, float(layer_idx)))
            assert torch.allclose(v_out, torch.full_like(v_out, float(layer_idx) + 0.5))

    def test_partial_get(self, cache):
        """Get should return only up to end_pos."""
        k = torch.randn(2, 8, 10, 64)
        v = torch.randn(2, 8, 10, 64)
        cache.update(layer_idx=0, k=k, v=v, start_pos=0)

        # Get only first 5 positions
        k_out, v_out = cache.get(layer_idx=0, end_pos=5)
        assert k_out.shape == (2, 8, 5, 64)
        assert torch.allclose(k_out, k[:, :, :5, :])


class TestStandardCacheErrors:
    """Tests for error handling."""

    def test_update_before_allocate_raises(self):
        """Update before allocate should raise RuntimeError."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        k = torch.randn(2, 8, 1, 64)
        v = torch.randn(2, 8, 1, 64)

        with pytest.raises(RuntimeError, match="not allocated"):
            cache.update(layer_idx=0, k=k, v=v, start_pos=0)

    def test_get_before_allocate_raises(self):
        """Get before allocate should raise RuntimeError."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)

        with pytest.raises(RuntimeError, match="not allocated"):
            cache.get(layer_idx=0, end_pos=1)

    def test_invalid_layer_idx_raises(self):
        """Invalid layer_idx should raise IndexError."""
        cache = StandardCache(n_layers=4, n_heads=8, head_dim=64)
        cache.allocate(batch_size=2, max_seq_len=128)

        k = torch.randn(2, 8, 1, 64)
        v = torch.randn(2, 8, 1, 64)

        with pytest.raises(IndexError):
            cache.update(layer_idx=10, k=k, v=v, start_pos=0)

        with pytest.raises(IndexError):
            cache.get(layer_idx=10, end_pos=1)


class TestStandardCacheDevice:
    """Tests for device and dtype handling."""

    def test_cpu_cache(self):
        """Cache on CPU."""
        cache = StandardCache(
            n_layers=2, n_heads=4, head_dim=32,
            device=torch.device('cpu'), dtype=torch.float32
        )
        cache.allocate(batch_size=1, max_seq_len=64)

        k = torch.randn(1, 4, 5, 32)
        v = torch.randn(1, 4, 5, 32)
        cache.update(layer_idx=0, k=k, v=v, start_pos=0)

        k_out, v_out = cache.get(layer_idx=0, end_pos=5)
        assert k_out.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_cache(self):
        """Cache on CUDA."""
        cache = StandardCache(
            n_layers=2, n_heads=4, head_dim=32,
            device=torch.device('cuda'), dtype=torch.float16
        )
        cache.allocate(batch_size=1, max_seq_len=64)

        k = torch.randn(1, 4, 5, 32, device='cuda', dtype=torch.float16)
        v = torch.randn(1, 4, 5, 32, device='cuda', dtype=torch.float16)
        cache.update(layer_idx=0, k=k, v=v, start_pos=0)

        k_out, v_out = cache.get(layer_idx=0, end_pos=5)
        assert k_out.device.type == 'cuda'
        assert k_out.dtype == torch.float16


class TestStandardCacheGenerationPattern:
    """Tests simulating actual generation patterns."""

    def test_autoregressive_generation_pattern(self):
        """Simulate typical autoregressive generation."""
        cache = StandardCache(n_layers=2, n_heads=4, head_dim=32)
        cache.allocate(batch_size=1, max_seq_len=100)

        # Step 1: Prefill with prompt (10 tokens)
        prompt_len = 10
        k_prompt = torch.randn(1, 4, prompt_len, 32)
        v_prompt = torch.randn(1, 4, prompt_len, 32)

        for layer in range(2):
            cache.update(layer, k_prompt, v_prompt, start_pos=0)

        # Step 2: Generate 5 tokens one at a time
        generated_k = []
        generated_v = []
        for step in range(5):
            pos = prompt_len + step
            k_new = torch.randn(1, 4, 1, 32)
            v_new = torch.randn(1, 4, 1, 32)
            generated_k.append(k_new)
            generated_v.append(v_new)

            for layer in range(2):
                cache.update(layer, k_new, v_new, start_pos=pos)

                # Verify we can get full history
                k_out, v_out = cache.get(layer, end_pos=pos + 1)
                assert k_out.shape == (1, 4, pos + 1, 32)

        # Final verification: all tokens present
        k_final, v_final = cache.get(layer_idx=0, end_pos=15)
        assert k_final.shape == (1, 4, 15, 32)
        assert torch.allclose(k_final[:, :, :prompt_len, :], k_prompt)
        for i, k_gen in enumerate(generated_k):
            assert torch.allclose(k_final[:, :, prompt_len + i:prompt_len + i + 1, :], k_gen)
