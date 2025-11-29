"""Data loading utilities for training."""

import os
from pathlib import Path

import torch
from torch.utils.data import Dataset, IterableDataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import ByteLevel as ByteLevelPreTokenizer
from tokenizers.decoders import ByteLevel as ByteLevelDecoder


class CharDataset(Dataset):
    """Map-style dataset for language modeling (loads all into memory)."""

    def __init__(self, text: str, seq_len: int, tokenizer: Tokenizer):
        self.seq_len = seq_len
        self.tok = tokenizer
        self.ids = self.tok.encode(text).ids

    def __len__(self):
        return max(0, len(self.ids) - self.seq_len)

    def __getitem__(self, i):
        x = self.ids[i:i + self.seq_len]
        y = self.ids[i + 1:i + self.seq_len + 1]
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class StreamingDataset(IterableDataset):
    """Streaming dataset that reads file in chunks (memory efficient)."""

    def __init__(self, file_path: str, seq_len: int, tokenizer: Tokenizer, chunk_size: int = 1024 * 1024):
        self.file_path = file_path
        self.seq_len = seq_len
        self.tok = tokenizer
        self.chunk_size = chunk_size  # 1MB chunks by default
        self._stop_iteration = False

    def stop(self):
        """Signal to stop iteration (for graceful shutdown)."""
        self._stop_iteration = True

    def __iter__(self):
        self._stop_iteration = False
        buffer_ids = []
        with open(self.file_path, 'r', encoding='utf-8') as f:
            while not self._stop_iteration:
                chunk = f.read(self.chunk_size)
                if not chunk:
                    break
                # Tokenize chunk and add to buffer
                chunk_ids = self.tok.encode(chunk).ids
                buffer_ids.extend(chunk_ids)

                # Yield sequences from buffer
                while len(buffer_ids) >= self.seq_len + 1 and not self._stop_iteration:
                    x = buffer_ids[:self.seq_len]
                    y = buffer_ids[1:self.seq_len + 1]
                    yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
                    # Slide by seq_len (non-overlapping for streaming)
                    buffer_ids = buffer_ids[self.seq_len:]

        # Yield remaining if enough tokens
        while len(buffer_ids) >= self.seq_len + 1 and not self._stop_iteration:
            x = buffer_ids[:self.seq_len]
            y = buffer_ids[1:self.seq_len + 1]
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
            buffer_ids = buffer_ids[self.seq_len:]


def build_tokenizer(corpus_paths: list, out_path: str, vocab_size: int = 4096) -> Tokenizer:
    """Build BPE tokenizer with ByteLevel encoding (GPT-2 style).

    Uses ByteLevel pre-tokenizer and decoder to properly handle subword merging.
    This prevents the "igh ing ly" problem where subwords decode with spaces.
    """
    tok = Tokenizer(BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevelPreTokenizer(add_prefix_space=False)
    tok.decoder = ByteLevelDecoder()
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["<unk>"],
        initial_alphabet=ByteLevelPreTokenizer.alphabet()
    )

    def line_iter():
        for p in corpus_paths:
            with open(p, 'r', encoding='utf-8') as f:
                for line in f:
                    yield line.strip()

    tok.train_from_iterator(line_iter(), trainer=trainer)
    tok.save(out_path)
    return tok


def get_file_size_mb(path: str) -> float:
    """Get file size in MB."""
    return os.path.getsize(path) / (1024 * 1024)


def create_dataloaders(
    train_path: str,
    val_path: str,
    tokenizer: Tokenizer,
    seq_len: int,
    batch_size: int,
    streaming_threshold_mb: float = 50.0,
    log_fn=None,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Uses streaming for large files to avoid OOM.
    """
    log = log_fn or print
    train_size_mb = get_file_size_mb(train_path)
    val_size_mb = get_file_size_mb(val_path)

    # Training data
    if train_size_mb > streaming_threshold_mb:
        log(f"Using streaming dataset for train ({train_size_mb:.1f}MB)")
        train_ds = StreamingDataset(train_path, seq_len, tokenizer)
        train_dl = DataLoader(train_ds, batch_size=batch_size, drop_last=True)
    else:
        log(f"Loading train data into memory ({train_size_mb:.1f}MB)...")
        with open(train_path, 'r', encoding='utf-8') as f:
            train_text = f.read()
        train_ds = CharDataset(train_text, seq_len, tokenizer)
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)

    # Validation data (always in memory)
    log(f"Loading val data into memory ({val_size_mb:.1f}MB)...")
    with open(val_path, 'r', encoding='utf-8') as f:
        val_text = f.read()
    log(f"Tokenizing val data ({len(val_text):,} chars)...")
    val_ds = CharDataset(val_text, seq_len, tokenizer)
    log(f"Val dataset ready: {len(val_ds):,} sequences")
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_dl, val_dl
