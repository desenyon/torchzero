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
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be between 0 and 1")
    n = len(ids)
    cut = int(n * (1 - val_fraction))
    return ids[:cut], ids[cut:]


def pack_sequences(ids, context_length):
    """Pack a 1-D id stream into non-overlapping rows of ``context_length``.
    Trailing remainder is dropped."""
    if context_length <= 0:
        raise ValueError("context_length must be positive")
    n = (len(ids) // context_length) * context_length
    return np.asarray(ids[:n], dtype=np.int64).reshape(-1, context_length)


def pack_lm_sequences(ids, context_length):
    """Pack T input tokens plus a real lookahead target per row.

    Consecutive rows overlap by one token; the incomplete tail is dropped.
    Split the stream before packing to avoid crossing validation boundaries.
    """
    if not isinstance(context_length, (int, np.integer)) or context_length <= 0:
        raise ValueError("context_length must be a positive integer")
    ids = np.asarray(ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("expected a one-dimensional token stream")
    count = max(0, (len(ids) - 1) // context_length)
    if not count:
        return np.empty((0, context_length + 1), dtype=np.int64)
    windows = np.lib.stride_tricks.sliding_window_view(ids, context_length + 1)
    return windows[::context_length][:count].copy()


class BatchSampler:
    """Deterministic shuffling batches over packed rows. Seeding uses a stable
    hash of (seed, epoch) so resume reproduces the same order."""

    def __init__(self, num_rows, batch_size, seed=0):
        if not isinstance(num_rows, (int, np.integer)) or num_rows <= 0:
            raise ValueError("num_rows must be positive")
        if not isinstance(batch_size, (int, np.integer)) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.num_rows = num_rows
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.cursor = 0
        self._cached_order = None

    def _order(self, epoch):
        digest = hashlib.sha256(f"{self.seed}:{epoch}".encode()).digest()
        local_seed = int.from_bytes(digest[:4], "little")
        rng = np.random.default_rng(local_seed)
        return rng.permutation(self.num_rows)

    def __iter__(self):
        order = self._order(self.epoch)
        for start in range(0, self.num_rows, self.batch_size):
            yield order[start:start + self.batch_size]

    def next_batch(self):
        """Consume one batch, starting a new epoch only after all rows."""
        if self.cursor == self.num_rows:
            self.advance_epoch()
        if self._cached_order is None:
            self._cached_order = self._order(self.epoch)
        end = min(self.cursor + self.batch_size, self.num_rows)
        batch = self._cached_order[self.cursor:end].copy()
        self.cursor = end
        return batch

    def state_dict(self):
        return {key: getattr(self, key) for key in
                ("num_rows", "batch_size", "seed", "epoch", "cursor")}

    def load_state_dict(self, state):
        for key in ("num_rows", "batch_size", "seed"):
            if state.get(key) != getattr(self, key):
                raise ValueError(f"sampler {key} mismatch")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("invalid sampler epoch")
        if not isinstance(cursor, int) or not 0 <= cursor <= self.num_rows:
            raise ValueError("invalid sampler cursor")
        if cursor != self.num_rows and cursor % self.batch_size:
            raise ValueError("sampler cursor must be on a batch boundary")
        self.epoch, self.cursor = epoch, cursor
        self._cached_order = None

    def advance_epoch(self):
        self.epoch += 1
        self.cursor = 0
        self._cached_order = None


def get_batch(rows, indices):
    """Gather a batch of input/target pairs (target = next token)."""
    batch = np.asarray(rows)[indices]
    if batch.ndim != 2 or batch.shape[1] < 2 or batch.dtype.kind not in "iu":
        raise ValueError("expected integer token rows with at least two columns")
    return batch[:, :-1].astype(np.int64), batch[:, 1:].astype(np.int64)


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
