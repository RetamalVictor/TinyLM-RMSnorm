"""Tests for gradient checkpointing in TinyLM."""

import pytest
import torch

from tinylm import TinyLM


@pytest.fixture
def small_config():
    """Small model configuration for testing."""
    return {
        "vocab_size": 1000,
        "dim": 128,
        "n_layers": 4,
        "n_heads": 4,
        "max_seq_len": 256,
    }


@pytest.fixture
def batch_input():
    """Sample batch input."""
    return torch.randint(0, 1000, (2, 32))


class TestGradientCheckpointingConfig:
    """Tests for gradient checkpointing configuration."""

    def test_default_checkpointing_disabled(self, small_config):
        """Test that gradient checkpointing is disabled by default."""
        model = TinyLM(**small_config)
        assert model.gradient_checkpointing is False

    def test_enable_checkpointing(self, small_config):
        """Test that gradient checkpointing can be enabled."""
        model = TinyLM(**small_config, gradient_checkpointing=True)
        assert model.gradient_checkpointing is True

    def test_disable_checkpointing(self, small_config):
        """Test that gradient checkpointing can be explicitly disabled."""
        model = TinyLM(**small_config, gradient_checkpointing=False)
        assert model.gradient_checkpointing is False


class TestGradientCheckpointingForward:
    """Tests for forward pass with gradient checkpointing."""

    def test_forward_with_checkpointing_training(self, small_config, batch_input):
        """Test forward pass with checkpointing in training mode."""
        model = TinyLM(**small_config, gradient_checkpointing=True)
        model.train()
        out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])

    def test_forward_with_checkpointing_eval(self, small_config, batch_input):
        """Test forward pass with checkpointing in eval mode (should skip checkpointing)."""
        model = TinyLM(**small_config, gradient_checkpointing=True)
        model.eval()
        with torch.no_grad():
            out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])

    def test_forward_output_matches(self, small_config, batch_input):
        """Test that forward output is the same with and without checkpointing."""
        torch.manual_seed(42)
        model_no_ckpt = TinyLM(**small_config, gradient_checkpointing=False)

        torch.manual_seed(42)
        model_ckpt = TinyLM(**small_config, gradient_checkpointing=True)

        # Both models should have identical parameters
        model_no_ckpt.eval()
        model_ckpt.eval()

        with torch.no_grad():
            out_no_ckpt = model_no_ckpt(batch_input)
            out_ckpt = model_ckpt(batch_input)

        torch.testing.assert_close(out_no_ckpt, out_ckpt, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_forward_with_different_architectures(self, small_config, batch_input, arch):
        """Test checkpointing works with different architectures."""
        model = TinyLM(**small_config, architecture=arch, gradient_checkpointing=True)
        model.train()
        out = model(batch_input)
        assert out.shape == (2, 32, small_config["vocab_size"])
        assert not torch.isnan(out).any()


class TestGradientCheckpointingBackward:
    """Tests for backward pass with gradient checkpointing."""

    def test_backward_with_checkpointing(self, small_config, batch_input):
        """Test that backward pass works with checkpointing."""
        model = TinyLM(**small_config, gradient_checkpointing=True)
        model.train()

        y = torch.randint(0, small_config["vocab_size"], (2, 32))
        out = model(batch_input)
        loss = torch.nn.functional.cross_entropy(
            out.view(-1, small_config["vocab_size"]), y.view(-1)
        )
        loss.backward()

        # Check gradients exist and are valid
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"

    def test_gradients_match(self, small_config, batch_input):
        """Test that gradients are the same with and without checkpointing."""
        torch.manual_seed(42)
        model_no_ckpt = TinyLM(**small_config, gradient_checkpointing=False)

        torch.manual_seed(42)
        model_ckpt = TinyLM(**small_config, gradient_checkpointing=True)

        # Use same target
        y = torch.randint(0, small_config["vocab_size"], (2, 32))

        # Compute gradients without checkpointing
        model_no_ckpt.train()
        out_no_ckpt = model_no_ckpt(batch_input)
        loss_no_ckpt = torch.nn.functional.cross_entropy(
            out_no_ckpt.view(-1, small_config["vocab_size"]), y.view(-1)
        )
        loss_no_ckpt.backward()
        grads_no_ckpt = {
            name: param.grad.clone() for name, param in model_no_ckpt.named_parameters()
            if param.grad is not None
        }

        # Compute gradients with checkpointing
        model_ckpt.train()
        out_ckpt = model_ckpt(batch_input)
        loss_ckpt = torch.nn.functional.cross_entropy(
            out_ckpt.view(-1, small_config["vocab_size"]), y.view(-1)
        )
        loss_ckpt.backward()
        grads_ckpt = {
            name: param.grad.clone() for name, param in model_ckpt.named_parameters()
            if param.grad is not None
        }

        # Compare gradients
        for name in grads_no_ckpt:
            torch.testing.assert_close(
                grads_no_ckpt[name], grads_ckpt[name],
                rtol=1e-4, atol=1e-4,
                msg=f"Gradient mismatch for {name}"
            )

    @pytest.mark.parametrize("arch", ["llama", "gpt"])
    def test_backward_different_architectures(self, small_config, batch_input, arch):
        """Test backward pass works with different architectures."""
        model = TinyLM(**small_config, architecture=arch, gradient_checkpointing=True)
        model.train()

        y = torch.randint(0, small_config["vocab_size"], (2, 32))
        out = model(batch_input)
        loss = torch.nn.functional.cross_entropy(
            out.view(-1, small_config["vocab_size"]), y.view(-1)
        )
        loss.backward()

        # Check at least some gradients exist
        has_grads = any(
            param.grad is not None
            for param in model.parameters()
            if param.requires_grad
        )
        assert has_grads, "No gradients computed"


class TestGradientCheckpointingMemory:
    """Tests for memory behavior with gradient checkpointing."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
    def test_memory_reduction(self):
        """Test that gradient checkpointing reduces memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        config = {
            "vocab_size": 1000,
            "dim": 256,
            "n_layers": 8,
            "n_heads": 8,
            "max_seq_len": 512,
        }

        # Measure memory without checkpointing
        torch.manual_seed(42)
        model_no_ckpt = TinyLM(**config, gradient_checkpointing=False).cuda()
        model_no_ckpt.train()

        x = torch.randint(0, 1000, (4, 128)).cuda()
        y = torch.randint(0, 1000, (4, 128)).cuda()

        torch.cuda.reset_peak_memory_stats()
        out = model_no_ckpt(x)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 1000), y.view(-1))
        loss.backward()
        mem_no_ckpt = torch.cuda.max_memory_allocated()

        del model_no_ckpt, out, loss
        torch.cuda.empty_cache()

        # Measure memory with checkpointing
        torch.manual_seed(42)
        model_ckpt = TinyLM(**config, gradient_checkpointing=True).cuda()
        model_ckpt.train()

        torch.cuda.reset_peak_memory_stats()
        out = model_ckpt(x)
        loss = torch.nn.functional.cross_entropy(out.view(-1, 1000), y.view(-1))
        loss.backward()
        mem_ckpt = torch.cuda.max_memory_allocated()

        # Checkpointing should use less memory (allow some tolerance)
        assert mem_ckpt < mem_no_ckpt, (
            f"Checkpointing did not reduce memory: "
            f"{mem_ckpt / 1e6:.1f}MB vs {mem_no_ckpt / 1e6:.1f}MB"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
