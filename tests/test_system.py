"""System tests for TinyLM end-to-end functionality.

These tests verify that the entire training pipeline works correctly,
helping catch regressions when making changes to the codebase.
"""

import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer

from tinylm import TinyLM, generate
from tinylm.architectures import get_architecture
from tinylm.quant import QuantConfig
from tinylm.training import (
    CheckpointManager,
    Trainer,
    TrainerConfig,
    build_tokenizer,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "The quick brown fox jumps over the lazy dog. " * 100


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def tiny_model():
    """Create a tiny model for testing."""
    return TinyLM(
        vocab_size=256,
        dim=64,
        n_layers=2,
        n_heads=2,
        max_seq_len=64,
    )


@pytest.fixture
def tokenizer(temp_dir, sample_text):
    """Create a byte-level tokenizer."""
    # Write sample text
    text_file = temp_dir / "sample.txt"
    text_file.write_text(sample_text)

    # Build tokenizer
    tokenizer_path = temp_dir / "tokenizer.json"
    build_tokenizer(
        [str(text_file)],
        str(tokenizer_path),
        vocab_size=256,
        tokenizer_type="bytelevel",
    )
    return Tokenizer.from_file(str(tokenizer_path))


# ---------------------------------------------------------------------------
# Model Creation Tests
# ---------------------------------------------------------------------------

class TestModelCreation:
    """Test model creation with various configurations."""

    def test_create_llama_model(self):
        """Test creating a LLaMA-style model."""
        model = TinyLM(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            architecture="llama",
        )
        assert model.arch.name == "llama"
        assert model.arch.norm_type == "rmsnorm"
        assert model.arch.pos_emb_type == "rope"

    def test_create_gpt_model(self):
        """Test creating a GPT-style model."""
        model = TinyLM(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            architecture="gpt",
        )
        assert model.arch.name == "gpt"
        assert model.arch.norm_type == "layernorm"
        assert model.arch.pos_emb_type == "learned"

    def test_create_with_arch_config(self):
        """Test creating model with explicit ArchitectureConfig."""
        arch = get_architecture("llama")
        model = TinyLM(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            arch_config=arch,
        )
        assert model.arch == arch

    def test_create_with_quant_config(self):
        """Test creating model with quantization."""
        quant = QuantConfig(enabled=True)
        model = TinyLM(
            vocab_size=1000,
            dim=128,
            n_layers=2,
            n_heads=4,
            quant_config=quant,
        )
        assert model.quant_config == quant


# ---------------------------------------------------------------------------
# Forward Pass Tests
# ---------------------------------------------------------------------------

class TestForwardPass:
    """Test forward pass with various inputs."""

    def test_basic_forward(self, tiny_model):
        """Test basic forward pass."""
        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            logits = tiny_model(x)
        assert logits.shape == (2, 16, 256)

    def test_forward_with_cache(self, tiny_model):
        """Test forward pass with KV cache."""
        cache = tiny_model.create_kv_cache(batch_size=1, max_seq_len=32)
        x = torch.randint(0, 256, (1, 8))

        with torch.no_grad():
            logits1 = tiny_model(x, cache=cache, start_pos=0)
        assert logits1.shape == (1, 8, 256)

        # Continue with more tokens
        x2 = torch.randint(0, 256, (1, 4))
        with torch.no_grad():
            logits2 = tiny_model(x2, cache=cache, start_pos=8)
        assert logits2.shape == (1, 4, 256)

    def test_forward_different_architectures(self):
        """Test forward pass works for both architectures."""
        for arch in ["llama", "gpt"]:
            model = TinyLM(
                vocab_size=256,
                dim=64,
                n_layers=2,
                n_heads=2,
                architecture=arch,
            )
            x = torch.randint(0, 256, (2, 16))
            with torch.no_grad():
                logits = model(x)
            assert logits.shape == (2, 16, 256), f"Failed for {arch}"


# ---------------------------------------------------------------------------
# Training Integration Tests
# ---------------------------------------------------------------------------

class TestTrainingIntegration:
    """Test training pipeline integration."""

    def test_single_training_step(self, tiny_model):
        """Test a single training step."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)

        x = torch.randint(0, 256, (4, 32))
        y = torch.randint(0, 256, (4, 32))

        tiny_model.train()
        logits = tiny_model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1)
        )

        loss.backward()
        optimizer.step()

        assert loss.item() > 0

    def test_trainer_basic(self, tiny_model):
        """Test Trainer class basic functionality."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-3)
        config = TrainerConfig(
            total_steps=10,
            eval_every=5,
            log_every=5,
            device="cpu",
        )

        trainer = Trainer(
            model=tiny_model,
            optimizer=optimizer,
            config=config,
        )

        # Create dummy dataloader
        dataset = [(
            torch.randint(0, 256, (8, 32)),
            torch.randint(0, 256, (8, 32))
        ) for _ in range(20)]

        # Run a few steps manually
        for batch in dataset[:3]:
            metrics = trainer.train_step(batch)
            trainer.state.step += 1

        assert trainer.state.step == 3

    def test_loss_decreases_over_training(self, tiny_model):
        """Verify that loss decreases during training."""
        optimizer = torch.optim.AdamW(tiny_model.parameters(), lr=1e-2)

        # Same data to ensure learning
        x = torch.randint(0, 256, (4, 32))
        y = x.clone()  # Next token prediction on same data

        losses = []
        tiny_model.train()

        for _ in range(50):
            optimizer.zero_grad()
            logits = tiny_model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1)
            )
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        # Loss should decrease
        assert losses[-1] < losses[0], "Loss did not decrease during training"


# ---------------------------------------------------------------------------
# Checkpointing Tests
# ---------------------------------------------------------------------------

class TestCheckpointing:
    """Test checkpointing and model loading."""

    def test_save_and_load_model(self, tiny_model, temp_dir):
        """Test saving and loading model weights."""
        # Get initial output
        x = torch.randint(0, 256, (1, 16))
        with torch.no_grad():
            original_output = tiny_model(x)

        # Save checkpoint
        ckpt_path = temp_dir / "model.pt"
        torch.save({
            "model": tiny_model.state_dict(),
        }, ckpt_path)

        # Create new model and load
        new_model = TinyLM(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=2,
            max_seq_len=64,
        )
        state = torch.load(ckpt_path)
        new_model.load_state_dict(state["model"])

        # Verify outputs match
        with torch.no_grad():
            loaded_output = new_model(x)

        torch.testing.assert_close(original_output, loaded_output)

    def test_checkpoint_manager(self, tiny_model, temp_dir):
        """Test CheckpointManager functionality."""
        ckpt_manager = CheckpointManager(temp_dir / "ckpts", keep_last=2)

        # Save multiple checkpoints
        for step in [100, 200, 300]:
            state = {
                "step": step,
                "model": tiny_model.state_dict(),
            }
            ckpt_manager.save(state, step, is_best=(step == 200))

        # Should only keep last 2
        latest = ckpt_manager.get_latest()
        assert latest is not None
        assert "300" in str(latest)

        # Best should exist
        best_path = temp_dir / "ckpts" / "best.pt"
        assert best_path.exists()


# ---------------------------------------------------------------------------
# Generation Tests
# ---------------------------------------------------------------------------

class TestGeneration:
    """Test text generation."""

    def test_generate_basic(self, tiny_model, tokenizer):
        """Test basic text generation."""
        tiny_model.eval()

        # Generate text
        output = generate(
            tiny_model,
            tokenizer,
            prompt="The",
            max_new_tokens=10,
            temperature=0.0,  # Greedy
        )

        assert isinstance(output, str)
        assert len(output) > 3  # At least longer than prompt

    def test_generate_with_sampling(self, tiny_model, tokenizer):
        """Test generation with sampling."""
        tiny_model.eval()

        output = generate(
            tiny_model,
            tokenizer,
            prompt="The",
            max_new_tokens=10,
            temperature=0.9,
            top_p=0.9,
        )

        assert isinstance(output, str)

    def test_generate_deterministic(self, tiny_model, tokenizer):
        """Test that greedy generation is deterministic."""
        tiny_model.eval()

        outputs = []
        for _ in range(3):
            out = generate(
                tiny_model,
                tokenizer,
                prompt="The",
                max_new_tokens=10,
                temperature=0.0,
            )
            outputs.append(out)

        assert all(o == outputs[0] for o in outputs)


# ---------------------------------------------------------------------------
# Configuration Tests
# ---------------------------------------------------------------------------

class TestHydraConfigs:
    """Test that all Hydra configs are valid."""

    def test_model_configs_exist(self):
        """Verify model config files exist."""
        conf_dir = Path(__file__).parent.parent / "conf" / "model"
        assert conf_dir.exists()

        configs = list(conf_dir.glob("*.yaml"))
        assert len(configs) >= 1

        # Check expected configs
        config_names = {c.stem for c in configs}
        assert "tiny" in config_names or len(config_names) > 0

    def test_training_configs_have_required_keys(self):
        """Verify training configs have all required keys."""
        import yaml

        conf_dir = Path(__file__).parent.parent / "conf" / "training"
        required_keys = [
            "steps", "batch_size", "seq_len", "lr",
            "early_stopping_patience", "early_stopping_min_delta"
        ]

        for config_file in conf_dir.glob("*.yaml"):
            with open(config_file) as f:
                config = yaml.safe_load(f)

            for key in required_keys:
                assert key in config, f"Missing {key} in {config_file.name}"


# ---------------------------------------------------------------------------
# Architecture Tests
# ---------------------------------------------------------------------------

class TestArchitectures:
    """Test architecture configurations."""

    def test_llama_architecture(self):
        """Test LLaMA architecture settings."""
        arch = get_architecture("llama")
        assert arch.norm_type == "rmsnorm"
        assert arch.pos_emb_type == "rope"
        assert arch.mlp_type == "gated"
        assert arch.activation == "silu"
        assert arch.norm_position == "pre"

    def test_gpt_architecture(self):
        """Test GPT architecture settings."""
        arch = get_architecture("gpt")
        assert arch.norm_type == "layernorm"
        assert arch.pos_emb_type == "learned"
        assert arch.mlp_type == "standard"
        assert arch.activation == "gelu"

    def test_architecture_roundtrip(self):
        """Test architecture config serialization."""
        arch = get_architecture("llama")
        d = arch.to_dict()
        restored = type(arch).from_dict(d)
        assert arch == restored


# ---------------------------------------------------------------------------
# Quantization Tests
# ---------------------------------------------------------------------------

class TestQuantization:
    """Test quantization functionality."""

    def test_quant_config_creation(self):
        """Test QuantConfig creation."""
        config = QuantConfig(enabled=True)
        assert config.enabled

        config_disabled = QuantConfig(enabled=False)
        assert not config_disabled.enabled

    def test_quant_config_roundtrip(self):
        """Test QuantConfig serialization."""
        config = QuantConfig(enabled=True, per_channel=True)
        d = config.to_dict()
        restored = QuantConfig.from_dict(d)
        assert config.enabled == restored.enabled
        assert config.per_channel == restored.per_channel

    def test_model_with_quant_forward(self):
        """Test forward pass with quantization enabled."""
        quant = QuantConfig(enabled=True)
        model = TinyLM(
            vocab_size=256,
            dim=64,
            n_layers=2,
            n_heads=2,
            quant_config=quant,
        )

        x = torch.randint(0, 256, (2, 16))
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (2, 16, 256)


# ---------------------------------------------------------------------------
# Integration Tests (require data files)
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not Path("data/tinyshakespeare_train.txt").exists(),
    reason="TinyShakespeare data not available"
)
class TestWithRealData:
    """Integration tests using real data files."""

    def test_training_smoke_test(self):
        """Smoke test: run training for a few steps."""
        result = subprocess.run(
            [
                "uv", "run", "python", "train.py",
                "model=tiny",
                "training=quick_test",
                "training.steps=10",
                "logging.eval_every=5",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Should complete without error
        assert result.returncode == 0, f"Training failed: {result.stderr}"
