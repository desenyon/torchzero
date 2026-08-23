"""Reference parity tests (README §12).

TorchZero results are compared against independent textbook implementations
(tests/reference_impls.py, plain Python + math module). TorchZero runs its
side in float32 like production; references run in float64.

Documented tolerances:
  * elementwise / normalization outputs: atol 2e-4 (float32 accumulation
    noise vs float64 reference)
  * attention probabilities: atol 1e-4
  * loss values: atol 1e-3
  * optimizer trajectories: atol 1e-5 per step
"""

import math

import numpy as np
import pytest

from torchzero import Tensor
from torchzero.nn.layers import Linear, LayerNorm, RMSNorm, gelu
from tests.reference_impls import (ref_matmul, ref_transpose,
                                   ref_softmax_row, ref_layernorm_row,
                                   ref_rmsnorm_row, ref_gelu, ref_sigmoid,
                                   ref_causal_attention, ref_cross_entropy,
                                   ref_adam_step)

RNG = np.random.default_rng(123)


def to_list(a):
    return [[float(v) for v in row] for row in np.asarray(a)]


class TestPrimitiveParity:
    def test_matmul(self):
        a = RNG.standard_normal((5, 6))
        b = RNG.standard_normal((6, 4))
        got = Tensor(a.astype(np.float32)) @ Tensor(b.astype(np.float32))
        want = ref_matmul(to_list(a), to_list(b))
        assert np.allclose(got.data, np.array(want), atol=2e-4)

    def test_softmax(self):
        x = RNG.standard_normal((3, 8)).astype(np.float64) * 2
        got = Tensor(x).softmax(axis=-1)
        for i in range(3):
            want = ref_softmax_row([float(v) for v in x[i]])
            assert np.allclose(got.data[i], want, atol=1e-5)

    def test_sigmoid_gelu(self):
        vals = [-3.0, -0.5, 0.0, 0.7, 4.0]
        t = Tensor(np.array(vals, dtype=np.float64), requires_grad=False)
        sig = t.sigmoid()
        for v, g in zip(vals, sig.data):
            assert abs(float(g) - ref_sigmoid(v)) < 1e-9
        ge = gelu(Tensor(np.array(vals, dtype=np.float64)))
        for v, g in zip(vals, ge.data):
            assert abs(float(g) - ref_gelu(v)) < 1e-9


class TestLayerParity:
    def test_linear_output(self):
        layer = Linear(6, 4, seed=0)
        x = RNG.standard_normal((3, 6))
        out = layer(Tensor(x.astype(np.float32)))
        want = ref_matmul(to_list(x),
                          to_list(layer.weight.data)) 
        want = [[v + float(b) for v, b in zip(row, layer.bias.data)]
                for row, b in zip(want, [layer.bias.data] * len(want))]
        assert np.allclose(out.data, np.array(want), atol=2e-4)

    def test_layernorm(self):
        ln = LayerNorm(8)
        x = RNG.standard_normal((2, 8))
        out = ln(Tensor(x.astype(np.float32)))
        for i in range(2):
            want = ref_layernorm_row(
                [float(v) for v in x[i]],
                [float(v) for v in ln.weight.data],
                [float(v) for v in ln.bias.data])
            assert np.allclose(out.data[i], want, atol=2e-4)

    def test_rmsnorm(self):
        rn = RMSNorm(8)
        x = RNG.standard_normal((2, 8))
        out = rn(Tensor(x.astype(np.float32)))
        for i in range(2):
            want = ref_rmsnorm_row([float(v) for v in x[i]],
                                   [float(v) for v in rn.weight.data])
            assert np.allclose(out.data[i], want, atol=2e-4)


class TestAttentionParity:
    def test_single_head_causal_attention(self):
        """Compare the exact attention math inside CausalSelfAttention
        against the scalar reference on one head with fixed weights."""
        T, d = 5, 4
        q = RNG.standard_normal((T, d))
        k = RNG.standard_normal((T, d))
        v = RNG.standard_normal((T, d))
        scale = 1.0 / math.sqrt(d)

        # TorchZero side: same ops as CausalSelfAttention.forward core
        scores_t = (Tensor(q.astype(np.float32))
                    @ Tensor(k.astype(np.float32)).transpose()) \
            * float(scale)
        scores_t = scores_t.masked_fill(~np.tril(np.ones((T, T), bool)),
                                        -1e30)
        probs_t = scores_t.softmax(axis=-1)
        out_t = probs_t @ Tensor(v.astype(np.float32))

        want = ref_causal_attention(to_list(q), to_list(k), to_list(v),
                                    scale)
        assert np.allclose(out_t.data, np.array(want), atol=1e-4)
        # probabilities match too
        full_scores = []
        for i in range(T):
            row = []
            for j in range(T):
                s = sum(q[i][c] * k[j][c] for c in range(d)) * scale
                row.append(s if j <= i else -np.inf)
            full_scores.append(row)
        for i in range(T):
            finite = [s for s in full_scores[i] if s != -np.inf]
            want_p = ref_softmax_row(finite)
            assert np.allclose(probs_t.data[i][:i + 1], want_p, atol=1e-5)


class TestLossParity:
    def test_cross_entropy(self):
        logits = RNG.standard_normal((4, 10)) * 2
        targets = [0, 3, 9, 5]
        got = Tensor.cross_entropy(
            Tensor(logits.astype(np.float32)), targets)
        want = ref_cross_entropy(to_list(logits), targets)
        assert abs(float(got.item()) - want) < 1e-3

    def test_full_model_logits_parity(self):
        """Tiny transformer forward vs an independent re-computation of the
        same architecture using the reference primitives and the model's own
        weights."""
        from torchzero.transformer import Transformer, TransformerConfig
        from torchzero.autograd.engine import no_grad

        cfg = TransformerConfig(vocab_size=20, dim=12, n_layers=1,
                                n_heads=2, block_size=6)
        model = Transformer(cfg, seed=11)
        model.eval()
        ids = np.array([[1, 2, 3, 4]])
        with no_grad():
            logits, _ = model.forward(ids)

        # independent recomputation
        w_emb = model.tok_emb.weight.data.astype(np.float64)
        x = w_emb[ids[0]]                       # (T, dim)
        blk = model.blocks[0]

        def rmsnorm_rows(xm, w):
            return np.array([ref_rmsnorm_row(row, list(w))
                             for row in xm])

        h = rmsnorm_rows(x, blk.norm_attn.weight.data)
        # qkv projections
        Wq = blk.attn.wq.weight.data.astype(np.float64)
        Wk = blk.attn.wk.weight.data.astype(np.float64)
        Wv = blk.attn.wv.weight.data.astype(np.float64)
        q = h @ Wq
        k = h @ Wk
        vv = h @ Wv
        # apply RoPE exactly as the module does: split halves per (T,H,Dh)
        from torchzero.transformer.model import rope_cache
        cos, sin = rope_cache(4, cfg.head_dim)
        H, Dh = cfg.n_heads, cfg.head_dim
        q4 = torch_free_rope(q.reshape(4, H, Dh), cos, sin)
        k4 = torch_free_rope(k.reshape(4, H, Dh), cos, sin)
        outs = []
        for hh in range(H):
            outs.append(ref_causal_attention(
                q4[:, hh, :].tolist(), k4[:, hh, :].tolist(),
                vv.reshape(4, H, Dh)[:, hh, :].tolist()))
        attn = np.stack(outs, axis=1).reshape(4, cfg.dim)
        Wo = blk.attn.proj.weight.data.astype(np.float64)
        attn_proj = attn @ Wo
        x = x + attn_proj
        h2 = rmsnorm_rows(x, blk.norm_mlp.weight.data)
        W1 = blk.mlp.fc.weight.data.astype(np.float64)
        b1_ = blk.mlp.fc.bias.data.astype(np.float64)
        hidden = h2 @ W1 + b1_
        hidden = np.vectorize(ref_gelu)(hidden)
        W2 = blk.mlp.proj.weight.data.astype(np.float64)
        b2_ = blk.mlp.proj.bias.data.astype(np.float64)
        mlp_out = hidden @ W2 + b2_
        x = x + mlp_out
        xf = rmsnorm_rows(x, model.norm_f.weight.data)
        # weight-tied head: logits = h @ E^T
        Wh = model.tok_emb.weight.data.astype(np.float64)
        ref_logits = xf @ Wh.T

        assert np.allclose(logits.data[0], ref_logits, atol=2e-3), \
            f"max diff {np.abs(logits.data[0] - ref_logits).max()}"


def torch_free_rope(x_thd, cos, sin):
    """RoPE applied with plain numpy loops (half-split convention)."""
    T, H, Dh = x_thd.shape
    half = Dh // 2
    out = np.empty_like(x_thd)
    for t in range(T):
        for h in range(H):
            for c in range(half):
                x1 = x_thd[t, h, c]
                x2 = x_thd[t, h, c + half]
                co, si = cos[t][c], sin[t][c]
                out[t, h, c] = x1 * co - x2 * si
                out[t, h, c + half] = x1 * si + x2 * co
    return out


class TestOptimizerParity:
    def test_adam_trajectory_matches_scalar_reference(self):
        from torchzero.optim import Adam
        p = Tensor(np.array([0.7, -1.3], dtype=np.float32),
                   requires_grad=True)
        grads = [np.array([0.5, -0.25], dtype=np.float32),
                 np.array([-0.1, 0.05], dtype=np.float32)]
        opt = Adam([p], lr=0.01)
        m = [0.0, 0.0]
        v = [0.0, 0.0]
        want = [0.7, -1.3]
        for step, g in enumerate(grads, start=1):
            p.grad = g
            opt.step()
            for i in range(2):
                want[i], m[i], v[i] = ref_adam_step(
                    want[i], float(g[i]), m[i], v[i], step, lr=0.01)
            assert np.allclose(p.data, np.array(want, dtype=np.float32),
                               atol=1e-5)
