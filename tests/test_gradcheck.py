"""Finite-difference gradient checks for every differentiable primitive
(README §4). All checks run in float64 with central differences."""

import numpy as np
import pytest

from torchzero import Tensor, tensor
from tests.gradcheck import assert_gradients_close

RNG = np.random.default_rng(42)


def sample(shape, positive=False):
    a = RNG.standard_normal(shape).astype(np.float64)
    if positive:
        a = np.abs(a) + 0.5
    return a


class TestPrimitiveGradientChecks:
    def test_add(self):
        x = sample((3, 4))
        y = sample((3, 4))
        assert_gradients_close(lambda t: (t + tensor(y)).sum(), x)

    def test_sub(self):
        x = sample((3, 4))
        assert_gradients_close(lambda t: (t - 1.5).sum(), x)

    def test_mul(self):
        x = sample((2, 5))
        y = sample((2, 5))
        assert_gradients_close(lambda t: t * tensor(y), x)

    def test_div(self):
        x = sample((4,))
        y = sample((4,), positive=True)
        assert_gradients_close(lambda t: t / tensor(y), x)

    def test_pow_scalar(self):
        x = sample((3,), positive=True)
        assert_gradients_close(lambda t: t ** 3.0, x)

    def test_neg(self):
        x = sample((5,))
        assert_gradients_close(lambda t: (-t).sum(), x)

    def test_matmul_2d(self):
        x = sample((3, 4))
        w = sample((4, 2))

        def f(t):
            return t @ Tensor(w, requires_grad=True)
        assert_gradients_close(f, x)

    def test_matmul_grads_both_operands(self):
        x = sample((3, 4))
        w = sample((4, 2))

        def f(a):
            wt = Tensor(w, requires_grad=True)
            out = a @ wt
            out.sum().backward()
            return out.sum()
        res = assert_gradients_close(f, x)
        # also verify weight gradient numerically
        from tests.gradcheck import numerical_gradient
        num_w = numerical_gradient(
            lambda arr: float((np.ones((3, 2)) * (x @ arr)).sum()),
            w)
        xt = Tensor(x)
        wt = Tensor(w, requires_grad=True)
        (xt @ wt).sum().backward()
        assert np.allclose(wt.grad, num_w, atol=1e-6)

    def test_matmul_batched(self):
        x = sample((2, 3, 4))
        w = sample((2, 4, 3))

        def f(t):
            wt = Tensor(w, requires_grad=True)
            return ((t @ wt) ** 2.0).sum()
        assert_gradients_close(f, x)

    def test_sum_axes(self):
        x = sample((2, 3, 4))
        assert_gradients_close(lambda t: t.sum(axis=(0, 2)), x)
        assert_gradients_close(lambda t: t.sum(axis=1, keepdims=True), x)

    def test_mean_axes(self):
        x = sample((3, 5))
        assert_gradients_close(lambda t: t.mean(axis=0), x)
        assert_gradients_close(lambda t: t.mean(), x)

    def test_max(self):
        # finite differencing max is only valid away from ties; use spaced values
        x = np.array([[1., 5., 2.], [7., 3., 9.]])
        res = assert_gradients_close(lambda t: t.max(axis=0), x, h=1e-4)
        assert res["max_error"] < 1e-8

    def test_transpose(self):
        x = sample((2, 3, 4))
        assert_gradients_close(lambda t: t.transpose(2, 0, 1) ** 2, x)

    def test_reshape(self):
        x = sample((2, 6))
        assert_gradients_close(lambda t: (t.reshape(3, 4) ** 2).sum(), x)

    def test_broadcast_to(self):
        x = sample((3, 1))
        assert_gradients_close(lambda t: t.broadcast_to((3, 4)) * 2.0, x)

    def test_exp_log_sqrt_tanh_sigmoid_relu(self):
        for name in ["exp", "tanh", "sigmoid"]:
            fn = getattr(Tensor(sample((5,)), requires_grad=False), name)
            x = sample((5,))
            assert_gradients_close(lambda t: getattr(t, name)().sum(), x,
                                   rtol=1e-5)
        x = sample((5,), positive=True)
        assert_gradients_close(lambda t: t.log().sum(), x)
        assert_gradients_close(lambda t: t.sqrt().sum(), x)
        xr = np.abs(sample((5,))) + 0.05
        assert_gradients_close(lambda t: t.relu().sum(), xr)

    def test_softmax(self):
        x = sample((3, 6)) * 3
        weights = tensor(RNG.random((3, 1)))

        def f(t):
            return (t.softmax(axis=-1) * weights).sum()
        assert_gradients_close(f, x, rtol=1e-4)
        # full softmax Jacobian via squared-row-sum trick
        assert_gradients_close(lambda t: (t.softmax(axis=-1) ** 2.0).sum(), x,
                               rtol=1e-4)

    def test_log_softmax(self):
        x = sample((4, 5)) * 2
        assert_gradients_close(lambda t: t.log_softmax(axis=-1).sum(), x)

    def test_getitem_gather(self):
        x = sample((6, 3))
        idx = np.array([0, 3, 3, 5])

        def f(t):
            gathered = t[idx] * tensor(np.arange(len(idx), dtype=float)[:, None])
            return gathered.sum()
        assert_gradients_close(f, x)

    def test_concat(self):
        x = sample((3, 2))
        other = sample((2, 2))

        def f(t):
            c = t.concat(Tensor(other, requires_grad=True), axis=0)
            return (c ** 2).sum()
        assert_gradients_close(f, x)

    def test_split(self):
        x = sample((4, 4))

        def f(t):
            a, b = t.split(2, axis=1)
            return (a ** 2).sum() + b.sum()
        assert_gradients_close(f, x)

    def test_masked_fill(self):
        x = sample((3, 4))
        mask = RNG.random((3, 4)) > 0.5

        def f(t):
            return (t.masked_fill(mask, 0.0) ** 2).sum()
        assert_gradients_close(f, x)


class TestCompositeChecks:
    def test_mlp_like_composition(self):
        x = sample((4, 3))
        w1 = sample((3, 8))
        b1 = sample((8,))
        w2 = sample((8, 2))

        def f(t):
            W1 = Tensor(w1, requires_grad=True)
            B1 = Tensor(b1, requires_grad=True)
            W2 = Tensor(w2, requires_grad=True)
            h = (t @ W1 + B1).tanh()
            return (h @ W2).softmax(axis=-1).log().sum()
        assert_gradients_close(f, x, rtol=1e-4)

    def test_cross_entropy_gradient(self):
        logits = sample((4, 7))
        targets = np.array([0, 3, 6, 1])

        def f(t):
            loss = Tensor.cross_entropy(t, targets)
            return loss
        assert_gradients_close(f, logits, h=1e-6, rtol=1e-4)

    def test_attention_like_expression(self):
        x = sample((2, 4, 6))
        wq = sample((6, 6))

        def f(t):
            WQ = Tensor(wq, requires_grad=True)
            q = t @ WQ
            scores = q @ q.transpose(0, 2, 1) / np.sqrt(6.0)
            probs = scores.softmax(axis=-1)
            ctx = probs @ q
            return (ctx ** 2.0).sum()
        assert_gradients_close(f, x, rtol=1e-3, h=1e-6)

    def test_branched_and_reused(self):
        x = sample((3,))

        def f(t):
            return (t * t + t.exp() * t - t / (t.abs() + 1.0)).sum()
        assert_gradients_close(f, x)
