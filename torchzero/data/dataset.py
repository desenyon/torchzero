"""Data pipeline: loading, tokenization, splits, packing, batching,
deterministic shuffling (README §9)."""

from __future__ import annotations

import hashlib
import numpy as np

from ..tokenizer.bpe import BPETokenizer


def load_text(path) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def train_val_split(ids, val_fraction=0.1):
    """Deterministic contiguous split (no leakage across the boundary)."""
    n = len(ids)
    cut = int(n * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def pack_sequences(ids, context_length):
    """Pack a 1-D id stream into non-overlapping rows of ``context_length``.
    Trailing remainder is dropped."""
    n = (len(ids) // context_length) * context_length
    return np.asarray(ids[:n], dtype=np.int64).reshape(-1, context_length)


class BatchSampler:
    """Deterministic shuffling batches over packed rows. Seeding uses a stable
    hash of (seed, epoch) so resume reproduces the same order."""

    def __init__(self, num_rows, batch_size, seed=0):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0

    def _order(self, epoch):
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode()).digest()
        local_seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(local_seed)
        return rng.permutation(self.num_rows)

    def __iter__(self):
        order = self._order(self.epoch)
        for start in range(0, self.num_rows, self.batch_size):
            yield order[start:start + self.batch_size]

    def advance_epoch(self):
        self.epoch += 1


def get_batch(rows, indices):
    """Gather a batch of input/target pairs (target = next token)."""
    x = rows[indices]
    y = np.concatenate([x[:, 1:], x[:, :1]], axis=1)
    return x.astype(np.int64), y.astype(np.int64)


class CharStreamDataset:
    """Tiny built-in deterministic dataset used for validation and tests when
    no external data is present."""

    TEXT = (
        "the sun rises in the east and sets in the west. "
        "rivers run to the sea and rain falls on the plain. "
        "small models learn patterns from simple repeated text. "
    ) * 8

    @classmethod
    def token_ids(cls, tokenizer=None):
        tok = tokenizer or BPETokenizer()
        if not tok.merges:
            tok.train(cls.TEXT, vocab_size=280)
        return tok.encode(cls.TEXT)
