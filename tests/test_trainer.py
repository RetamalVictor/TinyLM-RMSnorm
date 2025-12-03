"""Tests for the Trainer class."""

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from tinylm.training import Trainer, TrainerConfig, TrainerState


class SimpleModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size: int = 100, dim: int = 32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.fc = nn.Linear(dim, vocab_size)

    def forward(self, x):
        return self.fc(self.embed(x))


def create_dummy_dataloader(
    num_samples: int = 100,
    seq_len: int = 16,
    vocab_size: int = 100,
    batch_size: int = 8,
):
    """Create a dummy dataloader for testing."""
    x = torch.randint(0, vocab_size, (num_samples, seq_len))
    y = torch.randint(0, vocab_size, (num_samples, seq_len))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TestTrainerConfig:
    """Tests for TrainerConfig."""

    def test_default_config(self):
        config = TrainerConfig()
        assert config.total_steps == 10000
        assert config.grad_accum_steps == 1
        assert config.grad_clip == 1.0
        assert config.mixed_precision is False

    def test_custom_config(self):
        config = TrainerConfig(
            total_steps=100,
            grad_accum_steps=4,
            grad_clip=0.5,
            mixed_precision=True,
        )
        assert config.total_steps == 100
        assert config.grad_accum_steps == 4
        assert config.grad_clip == 0.5
        assert config.mixed_precision is True


class TestTrainerState:
    """Tests for TrainerState."""

    def test_default_state(self):
        state = TrainerState()
        assert state.step == 0
        assert state.best_val_loss == float("inf")
        assert state.accum_loss == 0.0
        assert state.should_stop is False


class TestTrainer:
    """Tests for the Trainer class."""

    @pytest.fixture
    def model(self):
        return SimpleModel(vocab_size=100, dim=32)

    @pytest.fixture
    def optimizer(self, model):
        return torch.optim.AdamW(model.parameters(), lr=1e-3)

    @pytest.fixture
    def config(self):
        return TrainerConfig(
            total_steps=10,
            grad_accum_steps=1,
            grad_clip=1.0,
            mixed_precision=False,
            log_every=5,
            eval_every=5,
            device="cpu",
        )

    @pytest.fixture
    def trainer(self, model, optimizer, config):
        return Trainer(model, optimizer, config)

    @pytest.fixture
    def train_dl(self):
        return create_dummy_dataloader(num_samples=100, batch_size=8)

    @pytest.fixture
    def val_dl(self):
        return create_dummy_dataloader(num_samples=50, batch_size=8)

    def test_trainer_creation(self, trainer):
        """Test trainer can be created."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.config is not None
        assert trainer.state.step == 0

    def test_train_step(self, trainer, train_dl):
        """Test single training step."""
        batch = next(iter(train_dl))
        metrics = trainer.train_step(batch)
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_evaluate(self, trainer, val_dl):
        """Test evaluation."""
        metrics = trainer.evaluate(val_dl, max_batches=5)
        assert "val_loss" in metrics
        assert "val_perplexity" in metrics
        assert metrics["val_loss"] > 0
        assert metrics["val_perplexity"] > 0

    def test_train_loop(self, trainer, train_dl, val_dl):
        """Test full training loop."""
        final_metrics = trainer.train(train_dl, val_dl, start_step=0)
        assert "final_step" in final_metrics
        assert final_metrics["final_step"] == trainer.config.total_steps

    def test_hooks_called(self, trainer, train_dl, val_dl):
        """Test that hooks are called during training."""
        hook_calls = {"on_train_start": 0, "on_step_end": 0, "on_eval": 0, "on_train_end": 0}

        def make_counter(event):
            def counter(t, m):
                hook_calls[event] += 1
            return counter

        trainer.add_hook("on_train_start", make_counter("on_train_start"))
        trainer.add_hook("on_step_end", make_counter("on_step_end"))
        trainer.add_hook("on_eval", make_counter("on_eval"))
        trainer.add_hook("on_train_end", make_counter("on_train_end"))

        trainer.train(train_dl, val_dl, start_step=0)

        assert hook_calls["on_train_start"] == 1
        assert hook_calls["on_step_end"] > 0
        assert hook_calls["on_eval"] > 0
        assert hook_calls["on_train_end"] == 1

    def test_state_dict_roundtrip(self, trainer, train_dl):
        """Test state dict save/load."""
        # Do some training
        for i, batch in enumerate(train_dl):
            if i >= 5:
                break
            trainer.train_step(batch)
            trainer.state.step += 1

        # Save state
        state = trainer.state_dict()
        assert "model" in state
        assert "optimizer" in state
        assert "step" in state

        # Create new trainer and load state
        new_model = SimpleModel()
        new_optimizer = torch.optim.AdamW(new_model.parameters())
        new_trainer = Trainer(new_model, new_optimizer, trainer.config)
        new_trainer.load_state_dict(state)

        assert new_trainer.state.step == trainer.state.step

    def test_stop_training(self, trainer, train_dl, val_dl):
        """Test stopping training via stop() method."""
        def stop_after_3_steps(t, m):
            if t.state.step >= 3:
                t.stop()

        trainer.add_hook("on_step_end", stop_after_3_steps)
        trainer.train(train_dl, val_dl)

        # Should have stopped early
        assert trainer.state.step < trainer.config.total_steps

    def test_invalid_hook_raises(self, trainer):
        """Test that invalid hook event raises ValueError."""
        with pytest.raises(ValueError, match="Unknown hook event"):
            trainer.add_hook("invalid_event", lambda t, m: None)


class TestTrainerDistributed:
    """Tests for distributed training support (stubs)."""

    @pytest.fixture
    def trainer(self):
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        config = TrainerConfig(device="cpu")
        return Trainer(model, optimizer, config)

    def test_setup_distributed_no_op(self, trainer):
        """Test that setup_distributed is a no-op."""
        # Should not raise
        trainer.setup_distributed(backend="nccl")
        assert not trainer._is_distributed

    def test_wrap_model_none(self, trainer):
        """Test wrap_model with None returns original model."""
        wrapped = trainer.wrap_model(wrapper=None)
        assert wrapped is trainer.model

    def test_wrap_model_ddp_raises(self, trainer):
        """Test wrap_model with ddp raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="DDP wrapper"):
            trainer.wrap_model(wrapper="ddp")

    def test_wrap_model_fsdp_raises(self, trainer):
        """Test wrap_model with fsdp raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="FSDP wrapper"):
            trainer.wrap_model(wrapper="fsdp")

    def test_wrap_model_invalid_raises(self, trainer):
        """Test wrap_model with invalid wrapper raises ValueError."""
        with pytest.raises(ValueError, match="Unknown wrapper"):
            trainer.wrap_model(wrapper="invalid")

    def test_is_main_process(self, trainer):
        """Test is_main_process returns True for rank 0."""
        assert trainer.is_main_process


class TestTrainerGradAccumulation:
    """Tests for gradient accumulation."""

    def test_grad_accumulation(self):
        """Test gradient accumulation behavior."""
        model = SimpleModel()
        optimizer = torch.optim.AdamW(model.parameters())
        config = TrainerConfig(
            total_steps=8,
            grad_accum_steps=4,
            device="cpu",
        )
        trainer = Trainer(model, optimizer, config)
        train_dl = create_dummy_dataloader(batch_size=4)

        optimizer_steps = 0

        def count_optimizer_steps(t, m):
            nonlocal optimizer_steps
            if "train_loss" in m:  # Only on optimizer step
                optimizer_steps += 1

        trainer.add_hook("on_step_end", count_optimizer_steps)
        trainer.train(train_dl, None)

        # With 8 steps and grad_accum_steps=4, should have 2 optimizer steps
        assert optimizer_steps == 2
