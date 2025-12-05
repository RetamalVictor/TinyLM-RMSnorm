"""Optimizer implementations for TinyLM."""

from typing import Any, Dict, Iterable

import torch
from torch import Tensor
from torch.optim import AdamW, Optimizer


def newton_schulz(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Orthogonalize matrix G using Newton-Schulz iteration.

    Approximates the nearest semi-orthogonal matrix.
    Coefficients (3.4445, -4.7750, 2.0315) are for quintic convergence.

    Args:
        G: Input gradient matrix (2D tensor)
        steps: Number of Newton-Schulz iterations
        eps: Small constant for numerical stability

    Returns:
        Orthogonalized matrix in original dtype.
    """
    assert G.ndim == 2, f"newton_schulz requires 2D tensor, got {G.ndim}D"

    a, b, c = (3.4445, -4.7750, 2.0315)

    # Work in float32 for numerical stability
    X = G.float()
    X = X / (X.norm() + eps)

    # Transpose if rows > cols for better conditioning
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T

    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.T

    return X.to(G.dtype)


def should_orthogonalize(name: str, param: Tensor) -> bool:
    """Decide if a parameter should be orthogonalized.

    Rules:
        - Only 2D tensors
        - Skip embeddings (by name)
        - Skip very skinny / non-square-ish matrices (min dim < 64)

    Args:
        name: Parameter name
        param: Parameter tensor

    Returns:
        True if orthogonalization should be applied
    """
    if param.ndim != 2:
        return False

    lname = name.lower()

    # Skip embeddings by name
    if any(x in lname for x in ["tok", "pos", "emb", "embed"]):
        return False

    # Skip very skinny matrices (not square-ish enough)
    h, w = param.shape
    if min(h, w) < 64:
        return False

    return True


class Muon(Optimizer):
    """Muon optimizer: Momentum + conditional Newton-Schulz orthogonalization.

    Applies to ALL parameters; orthogonalization is conditional:
        - 2D weight matrices (attention/MLP) -> momentum + orthogonalization
        - Embeddings (2D) -> momentum only
        - 1D params (biases, norms) -> momentum only

    Args:
        params: Iterable of parameters or param groups
        lr: Learning rate (default: 0.02)
        momentum: Momentum factor (default: 0.95)
        nesterov: Use Nesterov momentum (default: False for stability)
        ns_steps: Newton-Schulz iteration steps (default: 5)
        orthogonalize_every: Apply Newton-Schulz every N steps (default: 1)
        weight_decay: L2 weight decay on 2D params (default: 0.0)
    """

    def __init__(
        self,
        params: Iterable,
        lr: float = 0.02,
        momentum: float = 0.95,
        nesterov: bool = False,
        ns_steps: int = 5,
        orthogonalize_every: int = 1,
        weight_decay: float = 0.0,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not (0.0 <= momentum < 1.0):
            raise ValueError(f"Invalid momentum: {momentum}")
        if orthogonalize_every < 1:
            raise ValueError("orthogonalize_every must be >= 1")

        defaults: Dict[str, Any] = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            orthogonalize_every=orthogonalize_every,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

        # Internal step counter for frequency-based orthogonalization
        self._step: int = 0

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        self._step += 1

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            weight_decay = group["weight_decay"]
            orth_every = group["orthogonalize_every"]

            do_orth = (self._step % orth_every) == 0

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Name stored when building param groups
                name = state.get("name", "")

                # Initialize momentum buffer
                buf = state.get("momentum_buffer", None)
                if buf is None:
                    buf = torch.zeros_like(p)
                    state["momentum_buffer"] = buf

                # Update momentum: buf = m * buf + grad
                buf.mul_(momentum).add_(grad)

                # Compute update (with optional Nesterov)
                if nesterov:
                    update = grad + momentum * buf
                else:
                    update = buf

                # Optional L2 weight decay for 2D params
                if weight_decay > 0.0 and p.ndim == 2:
                    p.add_(p, alpha=-lr * weight_decay)

                # Orthogonalize only if conditions are met
                if do_orth and should_orthogonalize(name, p):
                    update = newton_schulz(update, steps=ns_steps)

                # Apply update
                p.add_(update, alpha=-lr)

        return loss


def build_optimizer(model, optimizer_type: str, config) -> Optimizer:
    """Factory function to build optimizer based on config.

    Args:
        model: The model to optimize
        optimizer_type: 'adamw' or 'muon'
        config: Hydra config with optimizer parameters

    Returns:
        Configured optimizer instance
    """
    if optimizer_type == "adamw":
        trainable = [p for p in model.parameters() if p.requires_grad]
        return AdamW(
            trainable,
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=tuple(config.betas),
        )

    elif optimizer_type == "muon":
        # Create param groups with associated names
        param_groups = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_groups.append({"params": [param], "name": name})

        optimizer = Muon(
            param_groups,
            lr=config.get("muon_lr", 0.02),
            momentum=config.get("muon_momentum", 0.95),
            nesterov=config.get("muon_nesterov", False),
            ns_steps=config.get("muon_ns_steps", 5),
            orthogonalize_every=config.get("muon_orth_every", 1),
            weight_decay=config.weight_decay,
        )

        # Store names in state for orthogonalization decision
        for group in optimizer.param_groups:
            group_name = group.get("name", "")
            for p in group["params"]:
                optimizer.state[p]["name"] = group_name

        return optimizer

    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


__all__ = ["Muon", "build_optimizer", "newton_schulz", "should_orthogonalize"]
