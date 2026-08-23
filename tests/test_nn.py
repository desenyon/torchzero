"""Module system tests: registration, serialization, modes, layers."""

import numpy as np
import pytest

from torchzero import Tensor, tensor
from torchzero.nn import (Module, Parameter, ModuleList, Sequential,
                          Linear, Embedding, LayerNorm, RMSNorm, Dropout,
                          gelu)


class MLP(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(4, 8, seed=0)
        self.norm = LayerNorm(8)
        self.fc2 = Linear(8, 3, seed=1)
        self.blocks = ModuleList([Linear(3, 3, seed=2), Linear(3, 3, seed=3)])
        self.head = Sequential(Linear(3, 2, seed=4))

    def forward(self, x):
        h = self.norm(self.fc1(x)).tanh()
        h = self.fc2(h)
        for block in self.blocks:
            h = h + block(h)
        return self.head(h)


class TestParameterRegistration:
    def test_automatic_registration(self):
        m = MLP()
        params = m.parameters()
        # fc1(w,b) norm(w,b) fc2(w,b) blocks 2x(w,b) head(w,b) = 12
        assert len(params) == 12

    def test_named_parameters_nested(self):
        m = MLP()
        names = set(m.named_parameters())
        assert "fc1.weight" in names
        assert "fc1.bias" in names
        assert "blocks.0.weight" in names
        assert "head.layers.0.weight" in names

    def test_no_duplicates_for_shared_parameter(self):
        m = Module()
        shared = Parameter(np.ones(3, dtype=np.float32))
        m.a = shared
        m.b = shared
        assert len(m.parameters()) == 1

    def test_zero_grad(self):
        m = MLP()
        x = tensor(np.random.default_rng(0).random((2, 4)).astype(np.float32))
        loss = m(x).sum()
        loss.backward()
        assert all(p.grad is not None for p in m.parameters())
        m.zero_grad()
        assert all(p.grad is None for p in m.parameters())

    def test_train_eval_mode(self):
        m = Module()
        child = Dropout(0.5)
        m.child = child
        m.eval()
        assert child.training is False and m.training is False
        m.train()
        assert child.training is True and m.training is True


class TestSerialization:
    def test_state_dict_roundtrip(self):
        src = MLP()
        dst = MLP()
        # perturb destination so weights differ
        for p in dst.parameters():
            p.data += np.random.default_rng(7).standard_normal(p.shape).astype(np.float32)
        before = {k: v.copy() for k, v in src.state_dict().items()}
        dst.load_state_dict(src.state_dict())
        after = dst.state_dict()
        for k in before:
            assert np.array_equal(before[k], after[k])

    def test_load_mismatch_raises(self):
        m = MLP()

        class Tiny(Module):
            def __init__(self):
                super().__init__()
                self.l = Linear(1, 1)

        with pytest.raises(KeyError):
            Tiny().load_state_dict(m.state_dict())

    def test_shape_mismatch_raises(self):
        m = MLP()
        bad = m.state_dict()
        bad["fc1.weight"] = np.zeros((2, 2), np.float32)
        with pytest.raises(ValueError):
            MLP().load_state_dict(bad)

    def test_pickle_save_load(self, tmp_path):
        m = MLP()
        path = tmp_path / "model.pkl"
        m.save(path)
        other = MLP()
        other.load(path)
        for k in m.state_dict():
            assert np.array_equal(m.state_dict()[k], other.state_dict()[k])


class TestLayers:
    def test_linear_forward(self):
        layer = Linear(3, 2, seed=0)
        x = tensor(np.ones((2, 3), dtype=np.float32))
        out = layer(x)
        expected = np.ones((2, 3)) @ layer.weight.data + layer.bias.data
        assert np.allclose(out.data, expected)

    def test_linear_backward_matches_reference(self):
        layer = Linear(3, 2, seed=0)
        rng = np.random.default_rng(0)
        x_np = rng.standard_normal((4, 3)).astype(np.float64)
        x = Tensor(x_np, requires_grad=True)
        out = x @ layer.weight + layer.bias
        out.sum().backward()
        # reference: dL/dx = ones @ W.T ; dL/dW = X.T @ ones ; db = sum
        ref_dx = np.ones((4, 2)) @ layer.weight.data.astype(float).T
        ref_dw = x_np.T @ np.ones((4, 2))
        assert np.allclose(x.grad, ref_dx, atol=1e-5)
        assert np.allclose(layer.weight.grad, ref_dw, atol=1e-5)

    def test_embedding_lookup_and_grad(self):
        emb = Embedding(10, 4, seed=0)
        ids = np.array([1, 1, 9])
        out = emb(ids)
        assert out.shape == (3, 4)
        assert np.allclose(out.data, emb.weight.data[ids])
        out.sum().backward()
        assert np.allclose(emb.weight.grad[1], [2] * 4)

    def test_embedding_out_of_range(self):
        emb = Embedding(5, 2)
        with pytest.raises(IndexError):
            emb(np.array([5]))
        with pytest.raises(IndexError):
            emb(np.array([-1]))

    def test_layernorm_values(self):
        ln = LayerNorm(4)
        x_np = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)
        out = ln(Tensor(x_np))
        mu = x_np.mean(-1, keepdims=True)
        var = x_np.var(-1, keepdims=True)
        expected = (x_np - mu) / np.sqrt(var + 1e-5)
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_rmsnorm_values(self):
        n = RMSNorm(4)
        x_np = np.random.default_rng(1).standard_normal((2, 4)).astype(np.float32)
        out = n(Tensor(x_np))
        expected = x_np / np.sqrt((x_np ** 2).mean(-1, keepdims=True) + 1e-6)
        assert np.allclose(out.data, expected, atol=1e-5)

    def test_norms_gradient_check(self):
        from tests.gradcheck import assert_gradients_close
        x = np.random.default_rng(2).standard_normal((3, 5))

        def f(t):
            return LayerNorm(5)(t).sum() + RMSNorm(5)(t).sum()
        res = assert_gradients_close(f, x, rtol=1e-4)
        assert res["max_error"] < 1e-6

    def test_dropout_train_vs_eval(self):
        d = Dropout(0.5, seed=0)
        d.train()
        x = tensor(np.ones((1000,), dtype=np.float32))
        out = d(x)
        zeros_frac = float((out.data == 0).mean())
        assert 0.35 < zeros_frac < 0.65
        d.eval()
        out = d(x)
        assert np.allclose(out.data, x.data)

    def test_gelu_value(self):
        import math
        x = np.array([[-1.0, 0.0, 1.0]])
        t = Tensor(x.astype(np.float64), requires_grad=True)
        g = gelu(t)
        beta, kappa = math.sqrt(2 / math.pi), 0.044715
        for i, v in enumerate([-1.0, 0.0, 1.0]):
            expect = 0.5 * v * (1 + math.tanh(beta * v + beta * kappa * v ** 3))
            assert abs(g.data[0, i] - expect) < 1e-12
