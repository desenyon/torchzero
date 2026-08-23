"""Finite-difference gradient checking utilities.

Central difference estimate:
    (f(x + h) - f(x - h)) / (2h)
run in float64 with h ~ 1e-6 for ~1e-10..1e-7 accuracy on smooth functions.
"""

from __future__ import annotations

import numpy as np

from torchzero import Tensor


def numerical_gradient(f, x: np.ndarray, h: float = 1e-6) -> np.ndarray:
    """Central finite-difference gradient of scalar-valued ``f`` at ``x``."""
    x = np.array(x, dtype=np.float64)
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = x[idx]
        x[idx] = orig + h
        f_hi = float(f(x))
        x[idx] = orig - h
        f_lo = float(f(x))
        x[idx] = orig
        grad[idx] = (f_hi - f_lo) / (2 * h)
        it.iternext()
    return grad


def analytical_gradient(f_tensor_fn, x: np.ndarray) -> np.ndarray:
    """Backward-mode gradient of ``f_tensor_fn(Tensor(x))`` w.r.t. x."""
    xt = Tensor(np.array(x, dtype=np.float64), requires_grad=True)
    out = f_tensor_fn(xt)
    if isinstance(out, tuple):
        loss = sum(o.sum() for o in out if o.requires_grad)
    else:
        loss = out
    loss.backward()
    return xt.grad


def check_gradients(f_tensor_fn, x: np.ndarray, h: float = 1e-6,
                    atol: float = 1e-7, rtol: float = 1e-5) -> dict:
    """Compare analytical vs numerical gradients. Returns diagnostics."""
    num = numerical_gradient(lambda a: _scalarize(f_tensor_fn, Tensor(a)), x, h=h)
    ana = analytical_gradient(f_tensor_fn, x)
    num = num.reshape(ana.shape)
    diff = np.abs(ana - num)
    tol = atol + rtol * np.abs(num)
    max_err = float(diff.max()) if diff.size else 0.0
    bad = int((diff > tol).sum())
    return {
        "ok": bad == 0,
        "max_error": max_err,
        "num_bad": bad,
        "analytical": ana,
        "numerical": num,
    }


def assert_gradients_close(f_tensor_fn, x: np.ndarray, **kwargs):
    result = check_gradients(f_tensor_fn, x, **kwargs)
    assert result["ok"], (
        f"gradient check failed: {result['num_bad']} elements exceed "
        f"tolerance; max error {result['max_error']:.3e}\n"
        f"analytical:\n{result['analytical']}\nnumerical:\n{result['numerical']}")
    return result


def _scalarize(fn, t):
    """Reduce any output of ``fn`` to a scalar for finite differencing."""
    out = fn(t)
    if isinstance(out, tuple):
        return float(sum(o.data.sum() for o in out))
    if out.data.size == 1:
        return float(out.data)
    return float(out.data.sum())
