"""Data pipeline tests: splits, packing, batching, deterministic shuffle."""

import numpy as np
import pytest

from torchzero.data import (train_val_split, pack_sequences, BatchSampler,
                            get_batch, CharStreamDataset)
from torchzero.tokenizer import BPETokenizer


class TestSplitPack:
    def test_split_no_overlap(self):
        ids = list(range(100))
        tr, va = train_val_split(ids, val_fraction=0.2)
        assert len(va) == 20 and len(tr) == 80
        assert set(tr).isdisjoint(set(va))

    def test_split_deterministic(self):
        ids = list(range(50))
        assert train_val_split(ids) == train_val_split(ids)

    def test_pack_drops_remainder(self):
        rows = pack_sequences(list(range(10)), context_length=4)
        assert rows.shape == (2, 4)
        assert np.allclose(rows[0], [0, 1, 2, 3])

    def test_pack_exact(self):
        rows = pack_sequences(np.arange(8), 4)
        assert rows.shape == (2, 4)

    def test_pack_empty(self):
        rows = pack_sequences([], 4)
        assert rows.shape == (0, 4)


class TestBatching:
    def test_get_batch_target_is_next_token(self):
        rows = pack_sequences(list(range(10)), 5)
        x, y = get_batch(rows, np.array([0]))
        assert np.allclose(x[0], [0, 1, 2, 3, 4])
        assert np.allclose(y[0], [1, 2, 3, 4, 0])

    def test_sampler_deterministic(self):
        s1 = BatchSampler(10, 3, seed=42)
        s2 = BatchSampler(10, 3, seed=42)
        b1 = list(s1)
        b2 = list(s2)
        assert all((a == b).all() for a, b in zip(b1, b2))

    def test_sampler_epoch_changes_order(self):
        s = BatchSampler(10, 10, seed=7)
        e0 = next(iter(s)).copy()
        s.advance_epoch()
        e1 = next(iter(s))
        assert not (e0 == e1).all()

    def test_sampler_covers_all_rows(self):
        s = BatchSampler(23, 5, seed=0)
        seen = []
        for batch in s:
            seen.extend(batch.tolist())
        assert sorted(seen) == list(range(23))

    def test_invalid_batch_size(self):
        with pytest.raises(ValueError):
            BatchSampler(10, 0)


class TestTokenizationPipeline:
    def test_char_stream_dataset(self):
        tok = BPETokenizer()
        ids = CharStreamDataset.token_ids(tok)
        assert len(ids) > 500
        text = CharStreamDataset.TEXT[:64]
        assert tok.decode(tok.encode(text)) == text

    def test_end_to_end(self):
        tok = CharStreamDataset.token_ids.__self__ if False else None
        tok = BPETokenizer().train(CharStreamDataset.TEXT, vocab_size=280)
        ids = tok.encode(CharStreamDataset.TEXT)
        tr, va = train_val_split(ids, 0.1)
        rows_tr = pack_sequences(tr, 16)
        rows_va = pack_sequences(va, 16)
        sampler = BatchSampler(len(rows_tr), 4, seed=1)
        batches = list(sampler)
        x, y = get_batch(rows_tr, batches[0])
        assert x.shape == (4, 16)
        # every id in range of vocab
        assert x.max() < tok.vocab_size and x.min() >= 0
