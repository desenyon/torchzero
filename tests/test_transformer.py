"""Transformer tests: causal masking, blocks, cross entropy, KV cache
equivalence, generation determinism, full backward."""

import numpy as np
import pytest

from torchzero import Tensor, no_grad
from torchzero.transformer import (Transformer, TransformerConfig,
                                   rope_cache, apply_rope)


def tiny_config(**kw):
    defaults = dict(vocab_size=64, dim=32, n_layers=2, n_heads=4,
                    block_size=16)
    defaults.update(kw)
    return TransformerConfig(**defaults)


def make_model(seed=0, **kw):
    cfg = tiny_config(**kw)
    model = Transformer(cfg, seed=seed)
    model.eval()
    with no_grad():
        pass
    return model


class TestCausalMask:
    def test_attention_causality_zero_future_influence(self):
        """Changing token t must not change logits at positions < t."""
        model = make_model()
        rng = np.random.default_rng(0)
        ids = rng.integers(0, 64, size=(1, 8)).astype(np.int64)
        with no_grad():
            logits1, _ = model.forward(ids)
            ids2 = ids.copy()
            ids2[0, 5] = (ids2[0, 5] + 7) % 64  # perturb position 5
            logits2, _ = model.forward(ids2)
        # positions before 5 identical
        assert np.allclose(logits1.data[0, :5], logits2.data[0, :5])
        # positions >= 5 differ
        assert not np.allclose(logits1.data[0, 5:], logits2.data[0, 5:])

    def test_masked_fill_blocks_gradient(self):
        a = Tensor(np.ones((1, 4)), requires_grad=True)
        mask = np.array([[True, False, False, False]])
        out = a.masked_fill(mask, -1e30).softmax(axis=-1)
        assert out.data[0, 0] < 1e-20
        out.sum().backward()
        assert a.grad[0, 0] == 0

    def test_large_score_stability(self):
        model = make_model()
        ids = np.full((1, 4), 63, dtype=np.int64)  # max vocab index
        with no_grad():
            logits, _ = model.forward(ids)
        assert np.all(np.isfinite(logits.data))


class TestForwardShapes:
    def test_logits_shape_and_loss(self):
        model = make_model()
        ids = np.random.default_rng(1).integers(0, 64, (2, 6)).astype(np.int64)
        targets = np.random.default_rng(2).integers(0, 64, (2, 6)).astype(np.int64)
        logits, loss = model.forward(ids, targets=targets)
        assert logits.shape == (2, 6, 64)
        assert loss.shape == ()
        assert float(loss.item()) > 0

    def test_sequence_too_long_raises(self):
        model = make_model()  # block_size=16
        with pytest.raises(ValueError):
            model.forward(np.zeros((1, 17), dtype=np.int64))

    def test_non_integer_input_raises(self):
        model = make_model()
        with pytest.raises(TypeError):
            model.forward(np.ones((1, 4), dtype=np.float32))


class TestBackward:
    def test_full_model_backward_produces_grads(self):
        model = make_model(dropout=0.0)
        model.train()
        ids = np.random.default_rng(0).integers(0, 64, (2, 8)).astype(np.int64)
        targets = np.random.default_rng(1).integers(0, 64, (2, 8)).astype(np.int64)
        logits, loss = model.forward(ids, targets=targets)
        loss.backward()
        grads_ok = [p.grad is not None and np.isfinite(p.grad).all()
                    for p in model.parameters()]
        assert all(grads_ok)
        assert any(np.abs(p.grad).sum() > 0 for p in model.parameters())

    def test_loss_decreases_with_steps(self):
        model = make_model()
        model.train()
        rng = np.random.default_rng(3)
        ids = rng.integers(0, 64, (4, 8)).astype(np.int64)
        targets = np.roll(ids, -1, axis=1)
        from torchzero.optim import AdamW
        opt = AdamW(model.parameters(), lr=3e-3)
        losses = []
        for _ in range(15):
            _, loss = model.forward(ids, targets=targets)
            opt.zero_grad()
            loss.backward()
            from torchzero.optim import clip_grad_norm
            clip_grad_norm(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        assert losses[-1] < losses[0] * 0.9


class TestKVCache:
    def test_cached_matches_uncached(self):
        model = make_model(seed=1)
        rng = np.random.default_rng(5)
        prompt = rng.integers(0, 64, (1, 6)).astype(np.int64)

        with no_grad():
            # uncached full forward
            full, _ = model.forward(prompt, use_cache=False)

            # cached: feed tokens one at a time
            outs = []
            for i in range(6):
                lg, _ = model.forward(prompt[:, i:i + 1],
                                      use_cache=True,
                                      reset_cache=(i == 0))
                outs.append(lg)
        cached = np.concatenate([o.data for o in outs], axis=1)
        assert np.allclose(full.data, cached, atol=1e-4), \
            f"max diff {np.abs(full.data - cached).max()}"

    def test_generation_deterministic_greedy(self):
        model = make_model(seed=2)
        prompt = [1, 2, 3]
        g1 = model.generate(prompt, 5, temperature=0.0, use_kv_cache=True)
        g2 = model.generate(prompt, 5, temperature=0.0, use_kv_cache=True)
        assert g1 == g2

    def test_kv_vs_no_kv_same_tokens_greedy(self):
        model = make_model(seed=3)
        prompt = [4, 5, 6, 7]
        cached = model.generate(list(prompt), 6, temperature=0.0,
                                use_kv_cache=True)
        uncached = Transformer.__dict__["generate"](
            model, list(prompt), 6, temperature=0.0, use_kv_cache=False)
        # greedy argmax path is deterministic regardless of cache
        assert cached == uncached
        assert len(cached) == len(prompt) + 6

    def test_cache_shapes(self):
        model = make_model(seed=4)
        ids = np.random.default_rng(9).integers(0, 64, (1, 5)).astype(np.int64)
        with no_grad():
            model.forward(ids, use_cache=True)
        for cache in model.kv_caches:
            assert cache["k"].shape[2] == 5
            assert cache["v"].shape[2] == 5


class TestRoPE:
    def test_rope_preserves_norm(self):
        x = Tensor(np.random.default_rng(0).standard_normal((1, 4, 2, 8)))
        cos, sin = rope_cache(4, 8)
        out = apply_rope(x, cos, sin)
        # rotation preserves the norm of each (first-half, second-half) pair
        n_in = x.data[..., :4] ** 2 + x.data[..., 4:] ** 2
        n_out = out.data[..., :4] ** 2 + out.data[..., 4:] ** 2
        assert np.allclose(n_in.sum(), n_out.sum(), atol=1e-4)
        assert np.allclose(n_in, n_out, atol=1e-4)

    def test_rope_relative_property(self):
        """Dot product q_i . k_j depends only on i-j."""
        head_dim = 8
        cos, sin = rope_cache(16, head_dim)
        rng = np.random.default_rng(1)
        q = rng.standard_normal((1, 1, 1, head_dim))
        k = rng.standard_normal((1, 1, 1, head_dim))
        dots = []
        for offset in range(10):
            qt = apply_rope(Tensor(q), cos[offset:offset + 1],
                            sin[offset:offset + 1])
            kt = apply_rope(Tensor(k), cos[:1], sin[:1])
            dots.append(float((qt.data @ kt.data.transpose(0, 1, 3, 2)).squeeze()))
        # same relative distance from different absolute start -> same dot
        q2t = apply_rope(Tensor(q), cos[3:4], sin[3:4])
        k2t = apply_rope(Tensor(k), cos[3:4], sin[3:4])
        d2 = float((q2t.data @ k2t.data.transpose(0, 1, 3, 2)).squeeze())
        assert abs(dots[0] - d2) < 1e-5


class TestConfig:
    def test_invalid_heads(self):
        with pytest.raises(ValueError):
            TransformerConfig(vocab_size=10, dim=30, n_heads=4)

    def test_roundtrip_dict(self):
        c = tiny_config()
        assert TransformerConfig.from_dict(c.to_dict()).dim == c.dim

    def test_param_count_positive(self):
        model = make_model()
        assert model.num_parameters() > 1000
