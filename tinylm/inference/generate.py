"""Text generation utilities for TinyLM."""

import random
from typing import Optional, TYPE_CHECKING
import torch
from tokenizers import Tokenizer

if TYPE_CHECKING:
    from tinylm.model.transformer import TinyLM


def sample_top_p(logits: torch.Tensor, top_p: float = 0.9) -> torch.Tensor:
    """Sample from logits using nucleus (top-p) sampling.

    Args:
        logits: Logits tensor of shape [batch, vocab_size]
        top_p: Cumulative probability threshold (default: 0.9)

    Returns:
        Sampled token indices of shape [batch, 1]
    """
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cdf = torch.cumsum(sorted_probs, dim=-1)
    mask = cdf > top_p
    mask[..., 0] = False  # Keep at least 1 token
    sorted_probs[mask] = 0.0
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    idx = torch.multinomial(sorted_probs, num_samples=1)
    next_token = sorted_idx.gather(-1, idx)
    return next_token


@torch.no_grad()
def generate(
    model: "TinyLM",
    tok: Tokenizer,
    prompt: str,
    max_new_tokens: int = 128,
    temperature: float = 1.0,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    freq_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    seed: Optional[int] = 0,
    stream: bool = False
) -> str:
    """Generate text from a prompt using TinyLM.

    Args:
        model: TinyLM model instance
        tok: Tokenizer instance
        prompt: Input text prompt
        max_new_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature (0=greedy, >0=sampling)
        top_p: Nucleus sampling threshold
        repetition_penalty: Penalty for repeating tokens
        freq_penalty: Frequency-based penalty
        presence_penalty: Presence-based penalty
        seed: Random seed for reproducibility
        stream: Whether to print tokens as they're generated

    Returns:
        Generated text string
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = next(model.parameters()).device
    ids = torch.tensor(tok.encode(prompt).ids, device=device).unsqueeze(0)

    # Create KV cache for generation
    cache = model.create_kv_cache(
        batch_size=1,
        max_seq_len=ids.size(1) + max_new_tokens,
    )

    recent_window = 512

    # Process full prompt first to fill KV cache
    logits = model(ids, cache=cache, start_pos=0)
    logits = logits[:, -1, :]

    for _ in range(max_new_tokens):

        # Apply penalties
        if ids.size(1) > 0 and (repetition_penalty > 1.0 or freq_penalty > 0.0 or presence_penalty > 0.0):
            recent = ids[:, -recent_window:]
            for b in range(ids.size(0)):
                unique, counts = torch.unique(recent[b], return_counts=True)
                if repetition_penalty > 1.0:
                    logits[b, unique] /= repetition_penalty
                if freq_penalty > 0.0:
                    logits[b, unique] -= freq_penalty * counts.to(logits.dtype)
                if presence_penalty > 0.0:
                    logits[b, unique] -= presence_penalty

        # Temperature and sampling
        if temperature > 0:
            if temperature != 1.0:
                logits = logits / temperature
            next_id = sample_top_p(logits, top_p=top_p)
        else:
            next_id = torch.argmax(logits, dim=-1, keepdim=True)

        ids = torch.cat([ids, next_id], dim=1)
        if stream:
            print(tok.decode(ids[0].tolist()), flush=True)

        # Compute logits for next iteration (single token with KV cache)
        logits = model(next_id, cache=cache, start_pos=ids.size(1)-1)
        logits = logits[:, -1, :]

    return tok.decode(ids[0].tolist())
