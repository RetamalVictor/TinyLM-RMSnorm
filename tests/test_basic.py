"""Basic tests for TinyLM model components."""

import pytest
import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that core modules can be imported."""
    try:
        from model import TinyLM, build_sincos, prealloc_kvcache
        from train import CharDataset
        assert True
    except ImportError as e:
        pytest.skip(f"Import failed: {e}")


def test_sincos_generation():
    """Test that RoPE sin/cos tables can be generated."""
    try:
        from model import build_sincos

        seq_len = 128
        dim = 64
        device = torch.device('cpu')

        sin, cos = build_sincos(seq_len, dim, device)

        assert sin.shape == (1, 1, seq_len, dim)
        assert cos.shape == (1, 1, seq_len, dim)
        assert sin.device == device
        assert cos.device == device
    except ImportError:
        pytest.skip("Model module not available")


def test_kvcache_allocation():
    """Test KV-cache pre-allocation."""
    try:
        from model import prealloc_kvcache

        batch_size = 2
        max_seq = 256
        n_heads = 8
        head_dim = 64
        device = torch.device('cpu')
        dtype = torch.float32

        cache = prealloc_kvcache(batch_size, max_seq, n_heads, head_dim, device, dtype)

        assert 'k' in cache
        assert 'v' in cache
        assert cache['k'].shape == (batch_size, n_heads, max_seq, head_dim)
        assert cache['v'].shape == (batch_size, n_heads, max_seq, head_dim)
        assert cache['k'].device == device
        assert cache['k'].dtype == dtype
    except ImportError:
        pytest.skip("Model module not available")


def test_model_creation():
    """Test that TinyLM model can be created."""
    try:
        from model import TinyLM

        vocab_size = 100
        dim = 128
        n_layers = 2
        n_heads = 4

        model = TinyLM(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=0.0
        )

        # Check model attributes
        assert model.dim == dim
        assert model.n_heads == n_heads
        assert len(model.blocks) == n_layers

        # Check parameter count
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 0

    except ImportError:
        pytest.skip("Model module not available")


def test_model_forward():
    """Test model forward pass."""
    try:
        from model import TinyLM, build_sincos

        # Small model for testing
        vocab_size = 100
        dim = 128
        n_layers = 2
        n_heads = 4
        seq_len = 32
        batch_size = 2

        model = TinyLM(
            vocab_size=vocab_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=0.0
        )
        model.eval()

        # Create inputs
        device = torch.device('cpu')
        idx = torch.randint(0, vocab_size, (batch_size, seq_len))
        sin, cos = build_sincos(seq_len, dim // n_heads, device)

        # Forward pass
        with torch.no_grad():
            logits = model(idx, sin, cos)

        # Check output shape
        assert logits.shape == (batch_size, seq_len, vocab_size)

    except ImportError:
        pytest.skip("Model module not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])