"""Byte-level BPE tokenizer implemented from scratch.

Vocabulary = 256 raw bytes + learned merge tokens. Encoding is deterministic.
Training greedily merges the most frequent adjacent pair (ties broken by pair
id for determinism). Save/load as JSON.
"""

from __future__ import annotations

import json
from collections import Counter


class BPETokenizer:
    def __init__(self):
        self.merges = []            # list[(int, int)] in learned order
        self.vocab = {}             # token id -> bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
        for j, pair in enumerate(self.merges):
            self.vocab[256 + j] = self.vocab[pair[0]] + self.vocab[pair[1]]

    # -------------------------------------------------------------- training
    def train(self, text: str, vocab_size: int, verbose=False):
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256
        for step in range(num_merges):
            counts = Counter(zip(ids[:-1], ids[1:]))
            if not counts:
                break
            # deterministic tie-break: highest count, then lowest pair
            pair = min(counts.items(),
                       key=lambda kv: (-kv[1], kv[0]))[0]
            new_id = 256 + step
            ids = self._merge(ids, pair, new_id)
            self.merges.append(pair)
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]
            if verbose and (step % 50 == 0 or step == num_merges - 1):
                print(f"merge {step + 1}/{num_merges}: {pair} -> {new_id}")
        return self

    @staticmethod
    def _merge(ids, pair, new_id):
        out = []
        i = 0
        n = len(ids)
        a, b = pair
        while i < n:
            if i < n - 1 and ids[i] == a and ids[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(ids[i])
                i += 1
        return out

    # ------------------------------------------------------------ encode/decode
    def _encode_bytes(self, data: bytes) -> list:
        ids = list(data)
        # apply merges in learned order (each merge is strictly "later" in the
        # priority queue; applying in order reproduces training-time encoding)
        for j, pair in enumerate(self.merges):
            new_id = 256 + j
            if len(ids) < 2:
                break
            ids = self._merge(ids, pair, new_id)
        return ids

    def encode(self, text: str) -> list:
        return self._encode_bytes(text.encode("utf-8"))

    def decode(self, ids) -> str:
        data = b"".join(self.vocab[int(i)] for i in ids)
        return data.decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ io
    def save(self, path):
        with open(path, "w") as f:
            json.dump({
                "type": "torchzero.byte_bpe",
                "merges": [list(p) for p in self.merges],
            }, f)

    @classmethod
    def load(cls, path):
        tok = cls()
        with open(path) as f:
            obj = json.load(f)
        if obj.get("type") != "torchzero.byte_bpe":
            raise ValueError(f"unrecognized tokenizer format in {path}")
        tok.merges = [tuple(p) for p in obj["merges"]]
        for j, pair in enumerate(tok.merges):
            tok.vocab[256 + j] = tok.vocab[pair[0]] + tok.vocab[pair[1]]
        return tok

    @property
    def vocab_size(self):
        return 256 + len(self.merges)
