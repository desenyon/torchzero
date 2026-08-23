"""Optimizer correctness: several deterministic steps compared against a
trusted reference calculation computed explicitly in this file."""

import numpy as np
import pytest

from torchzero import Tensor
from torchzero.optim import SGD, Adam, AdamW, clip_grad_norm, LambdaLR


def make_params():
    p1 = Tensor(np.array([1.0, 2.0], dtype=np.float32), requires_grad=True)
    p2 = Tensor(np.array([[-0.5]], dtype=np.float32), requires_grad=True)
    p1.grad = np.array([0.1, -0.2], dtype=np.float32)
    p2.grad = np.array([[0.3]], dtype=np.float32)
    return [p1, p2]


class TestSGD:
    def test_vanilla_two_steps(self):
        params = make_params()
        opt = SGD(params, lr=0.5)

        ref = [p.data.astype(np.float64).copy() for p in params]
        grads = [p.grad.copy().astype(float) for p in params]
        for _ in range(2):
            for i in range(len(ref)):
                ref[i] -= 0.5 * grads[i]
        opt.step()
        opt.step()
        # gradients stay constant here (we do not refresh them), matching ref
        for got, expected in zip(params, ref):
            assert np.allclose(got.data, expected.astype(np.float32))

    def test_momentum_matches_reference(self):
        params = make_params()
        mu, lr = 0.9, 0.1
        opt = SGD(params, lr=lr, momentum=mu)

        p = params[0].data.astype(np.float64).copy()
        g = params[0].grad.astype(float).copy()
        v = np.zeros_like(p)
        for _ in range(3):
            v = mu * v + g
            p -= lr * v
        for _ in range(3):
            opt.step()
        assert np.allclose(params[0].data, p.astype(np.float32), atol=1e-6)

    def test_weight_decay(self):
        params = make_params()
        wd, lr = 0.01, 0.1
        opt = SGD(params, lr=lr, weight_decay=wd)
        p = params[0].data.astype(np.float64).copy()
        g = params[0].grad.astype(float).copy()
        p -= lr * (g + wd * p)
        opt.step()
        assert np.allclose(params[0].data, p.astype(np.float32))


class TestAdam:
    def test_adam_reference_three_steps(self):
        """Reference Adam update computed step by step with plain numpy."""
        params = make_params()
        lr, b1, b2, eps = 0.05, 0.9, 0.999, 1e-8
        opt = Adam(params, lr=lr, betas=(b1, b2), eps=eps)

        m = [np.zeros(p.shape) for p in params]
        v = [np.zeros(p.shape) for p in params]
        ref = [p.data.astype(np.float64).copy() for p in params]
        grads = [p.grad.astype(np.float64).copy() for p in params]

        for t in range(1, 4):
            for i in range(len(ref)):
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
                mhat = m[i] / (1 - b1 ** t)
                vhat = v[i] / (1 - b2 ** t)
                ref[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            opt.step()

        for got, expected in zip(params, ref):
            assert np.allclose(got.data, expected.astype(np.float32),
                               atol=1e-6)

    def test_adam_state_dict_roundtrip(self):
        # run two steps; snapshot params + optimizer state after one
        params = make_params()
        opt = Adam(params, lr=0.02)
        opt.step()
        sd = opt.state_dict()
        param_snapshot = [p.data.copy() for p in params]
        opt.step()
        expected = [p.data.copy() for p in params]

        # fresh run: restore params + optimizer snapshot -> next step matches
        params_b = make_params()
        for pb, snap in zip(params_b, param_snapshot):
            pb.data = snap.copy()
        opt_b = Adam(params_b, lr=0.02)
        opt_b.load_state_dict(sd)
        opt_b.step()
        for got, want in zip(params_b, expected):
            assert np.allclose(got.data, want, atol=1e-7)


class TestAdamW:
    def test_adamw_decoupled_decay_reference(self):
        params = make_params()
        lr, b1, b2, eps, wd = 0.05, 0.9, 0.999, 1e-8, 0.1
        opt = AdamW(params, lr=lr, betas=(b1, b2), eps=eps,
                    weight_decay=wd)

        m = [np.zeros(p.shape) for p in params]
        v = [np.zeros(p.shape) for p in params]
        ref = [p.data.astype(np.float64).copy() for p in params]
        grads = [p.grad.astype(np.float64).copy() for p in params]

        for t in range(1, 4):
            for i in range(len(ref)):
                ref[i] -= lr * wd * ref[i]  # decoupled decay first
                m[i] = b1 * m[i] + (1 - b1) * grads[i]
                v[i] = b2 * v[i] + (1 - b2) * grads[i] ** 2
                mhat = m[i] / (1 - b1 ** t)
                vhat = v[i] / (1 - b2 ** t)
                ref[i] -= lr * mhat / (np.sqrt(vhat) + eps)
            opt.step()

        for got, expected in zip(params, ref):
            assert np.allclose(got.data, expected.astype(np.float32),
                               atol=1e-6)

    def test_adamw_differs_from_adam_with_wd(self):
        pa = make_params()
        pb = make_params()
        adam = Adam(pa, lr=0.1, weight_decay=0.1)
        adamw = AdamW(pb, lr=0.1, weight_decay=0.1)
        adam.step()
        adamw.step()
        assert not np.allclose(pa[0].data, pb[0].data)


class TestClipping:
    def test_clip_scales_to_max_norm(self):
        params = make_params()
        total_norm = float(np.sqrt(
            sum((p.grad.astype(float) ** 2).sum() for p in params)))
        clipped = clip_grad_norm(params, max_norm=total_norm / 2)
        new_norm = float(np.sqrt(
            sum((p.grad.astype(float) ** 2).sum() for p in params)))
        assert abs(new_norm - total_norm / 2) < 1e-6
        assert abs(clipped - total_norm) < 1e-6

    def test_no_clip_when_under_threshold(self):
        params = make_params()
        before = [p.grad.copy() for p in params]
        clip_grad_norm(params, max_norm=1e6)
        for b, p in zip(before, params):
            assert np.allclose(b, p.grad)


def test_lr_schedule_lambda():
    params = make_params()
    opt = SGD(params, lr=1.0)
    sched = LambdaLR(opt, lambda s: 0.5 ** s)
    lrs = []
    for _ in range(3):
        sched.step()
        lrs.append(opt.lr)
    assert np.allclose(lrs, [0.5, 0.25, 0.125])
