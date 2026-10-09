"""Regression checks against independent central finite differences."""

import numpy as np
import pytest

from torchzero import Tensor, no_grad
from tests.gradcheck import numerical_gradient


def check_binary(fn, ref, a, b):
    ta, tb = Tensor(a.copy(), True), Tensor(b.copy(), True)
    out = fn(ta, tb)
    weights = np.arange(1, out.size + 1).reshape(out.shape)
    out.backward(weights)
    for actual, expected in (
        (ta.grad, numerical_gradient(lambda x: (ref(x, b) * weights).sum(), a)),
        (tb.grad, numerical_gradient(lambda x: (ref(a, x) * weights).sum(), b)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("shapes", [
    ((3,), (3,)), ((3,), (2, 3, 4)), ((2, 4, 3), (3,)),
    ((1, 2, 3), (4, 3, 2)), ((2, 1, 4, 3), (1, 5, 3, 2)),
])
def test_matmul_all_numpy_ranks(shapes):
    rng = np.random.default_rng(42)
    a, b = (rng.normal(size=shape) for shape in shapes)
    check_binary(lambda x, y: x @ y, np.matmul, a, b)


def test_tensor_power_both_broadcast_operands():
    check_binary(lambda x, y: x ** y, np.power,
                 np.array([[1.2], [2.3]]), np.array([[0.5, 2., 3.1]]))


def test_reverse_scalar_power():
    x = Tensor(np.array([0.2, 1.2]), True)
    (3. ** x).sum().backward()
    np.testing.assert_allclose(x.grad, 3. ** x.data * np.log(3.))


@pytest.mark.parametrize("axis,keepdims", [(None, False), (0, True), (-1, False)])
def test_min_gradient(axis, keepdims):
    data = np.array([[1., 6., 3.], [7., 2., 9.]])
    x = Tensor(data.copy(), True)
    x.min(axis=axis, keepdims=keepdims).sum().backward()
    expected = numerical_gradient(lambda a: a.min(axis=axis, keepdims=keepdims).sum(), data)
    np.testing.assert_allclose(x.grad, expected, atol=1e-7)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_gather_duplicate_indices_accumulate(axis):
    data = np.arange(6., dtype=float).reshape(2, 3)
    idx = np.array([[0, 0, 1], [1, 0, 1]])
    x = Tensor(data.copy(), True)
    weights = np.arange(1., 7.).reshape(2, 3)
    x.gather(Tensor(idx), axis=axis).backward(weights)
    expected = numerical_gradient(
        lambda a: (np.take_along_axis(a, idx, axis=axis) * weights).sum(), data)
    np.testing.assert_allclose(x.grad, expected, atol=1e-7)


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_scatter_add_forward_and_both_gradients(axis):
    idx = np.array([[0, 0, 1], [1, 0, 1]])
    a = np.arange(6., dtype=float).reshape(2, 3)
    b = np.arange(1., 7.).reshape(2, 3)

    def reference(base, src):
        result = base.copy()
        for coord in np.ndindex(idx.shape):
            dest = list(coord)
            dest[axis] = idx[coord]
            result[tuple(dest)] += src[coord]
        return result

    check_binary(lambda x, y: x.scatter_add(Tensor(idx), y, axis), reference, a, b)
    np.testing.assert_array_equal(Tensor(a).scatter_add(Tensor(idx), Tensor(b), axis).data,
                                  reference(a, b))
    with no_grad():
        out = Tensor(a, True).scatter_add(Tensor(idx), Tensor(b, True), axis)
    assert not out.requires_grad and not out._parents


def test_scatter_add_legacy_row_indices():
    base = Tensor(np.zeros((3, 2)), True)
    src = Tensor(np.array([[1., 2.], [3., 4.]]), True)
    out = base.scatter_add(Tensor(np.array([1, 1])), src)
    np.testing.assert_array_equal(out.data, [[0., 0.], [4., 6.], [0., 0.]])
    out.sum().backward()
    np.testing.assert_array_equal(src.grad, np.ones((2, 2)))
    np.testing.assert_array_equal(base.grad, np.ones((3, 2)))


def test_gather_broadcast_non_axis_dimensions():
    data = np.array([[2., 3., 5.]])
    idx = np.array([[0, 0], [1, 2]])
    x = Tensor(data.copy(), True)
    x.gather(Tensor(idx)).sum().backward()
    np.testing.assert_array_equal(x.grad, [[2., 1., 1.]])


@pytest.mark.parametrize("exponent", [0, Tensor(0.)])
def test_zero_power_has_finite_zero_gradient_at_zero(exponent):
    x = Tensor(np.array([0., 2.]), True)
    with np.errstate(all="raise"):
        (x ** exponent).sum().backward()
    np.testing.assert_array_equal(x.grad, [0., 0.])


def test_negative_transpose_axes_backward():
    data = np.arange(24., dtype=float).reshape(2, 3, 4)
    x = Tensor(data.copy(), True)
    weights = np.arange(24.).reshape(4, 2, 3)
    x.transpose(-1, 0, 1).backward(weights)
    np.testing.assert_array_equal(x.grad, weights.transpose(1, 2, 0))


def test_array_exponents_remain_supported():
    x = Tensor(np.array([[0.], [2.]]), True)
    exponent = np.array([[0., 2., 3.]])
    (x ** exponent).sum().backward()
    np.testing.assert_array_equal(x.grad, [[0.], [16.]])
