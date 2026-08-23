"""Tensor arithmetic, broadcasting, shape ops, dtypes, edge cases."""

import numpy as np
import pytest

from torchzero import Tensor, tensor, zeros, randn, no_grad


class TestArithmetic:
    def test_add_sub_mul_div_values(self):
        a = tensor([[1., 2.], [3., 4.]])
        b = tensor([[5., 6.], [7., 8.]])
        assert np.allclose((a + b).data, a.data + b.data)
        assert np.allclose((a - b).data, a.data - b.data)
        assert np.allclose((a * b).data, a.data * b.data)
        assert np.allclose((a / b).data, a.data / b.data)

    def test_scalar_ops(self):
        a = tensor([1., 2., 3.])
        assert np.allclose((a + 2).data, [3, 4, 5])
        assert np.allclose((2 + a).data, [3, 4, 5])
        assert np.allclose((a * 2).data, [2, 4, 6])
        assert np.allclose((2 * a).data, [2, 4, 6])
        assert np.allclose((a - 1).data, [0, 1, 2])
        assert np.allclose((10 - a).data, [9, 8, 7])
        assert np.allclose((a / 2).data, [.5, 1, 1.5])
        assert np.allclose((6 / a).data, [6, 3, 2])

    def test_neg(self):
        a = tensor([1., -2.])
        assert np.allclose((-a).data, [-1, 2])

    def test_pow(self):
        a = tensor([1., 2., 3.], requires_grad=True)
        p = (a ** 3).sum()
        p.backward()
        assert np.allclose(a.grad, 3 * a.data ** 2)

    def test_matmul_2d(self):
        a = tensor(np.arange(6.).reshape(2, 3))
        b = tensor(np.arange(12.).reshape(3, 4))
        assert np.allclose((a @ b).data, a.data @ b.data)
        assert (a @ b).shape == (2, 4)

    def test_matmul_batched(self):
        a = randn(2, 3, 4, seed=0)
        b = randn(2, 4, 5, seed=1)
        assert np.allclose((a @ b).data, a.data @ b.data)

    def test_matmul_vec_forms(self):
        v = tensor([1., 2.])
        m = tensor([[1., 0.], [0., 1.]])
        assert np.allclose((v @ m).data, v.data @ m.data)
        assert np.allclose((m @ v).data, m.data @ v.data)
        assert np.allclose((v @ v).item(), 5.0)


class TestBroadcasting:
    def test_broadcast_shapes(self):
        a = randn(3, 4, seed=0)
        b = randn(4, seed=1)
        assert (a + b).shape == (3, 4)
        c = randn(1, 4, seed=2)
        d = randn(3, 1, seed=3)
        assert (c + d).shape == (3, 4)

    def test_broadcast_grad_reduction(self):
        x = tensor(np.ones((2, 3)), requires_grad=True)
        y = tensor([1., 2., 3.], requires_grad=True)
        ((x * y) ** 2).sum().backward()
        # d/dy sum((x*y)^2) = sum_x 2*y*x^2
        expected = 2 * y.data * (x.data ** 2).sum(axis=0)
        assert np.allclose(y.grad, expected)
        expected_x = 2 * x.data * y.data ** 2
        assert np.allclose(x.grad, expected_x)

    def test_scalar_broadcast_grad(self):
        s = tensor(3.0, requires_grad=True)
        x = randn(4, 5, seed=0, requires_grad=True)
        ((s * x).sum()).backward()
        assert s.grad.shape == ()
        assert np.isclose(s.grad, x.data.sum())
        assert x.grad.shape == (4, 5)


class TestReductions:
    def test_sum_mean_all(self):
        a = randn(3, 4, seed=0)
        assert np.isclose(a.sum().item(), a.data.sum())
        assert np.isclose(a.mean().item(), a.data.mean())

    def test_sum_axis_keepdims(self):
        a = randn(2, 3, 4, seed=1)
        assert a.sum(axis=(0, 2), keepdims=True).shape == (1, 3, 1)
        assert a.sum(axis=1).shape == (2, 4)

    def test_max_min(self):
        a = tensor([[1., 5., 2.], [7., 3., 9.]])
        assert a.max().item() == 9
        assert np.allclose(a.max(axis=0).data, [7, 5, 9])
        assert np.allclose(a.min(axis=1).data, [1, 3])

    def test_max_tie_routes_to_first(self):
        a = tensor([1., 2., 2.], requires_grad=True)
        a.max().backward()
        assert np.allclose(a.grad, [0, 1, 0])


class TestShapeOps:
    def test_reshape_view_transpose(self):
        a = randn(2, 3, 4, seed=0)
        assert a.reshape(6, 4).shape == (6, 4)
        assert a.view(-1).shape == (24,)
        assert a.transpose(1, 0, 2).shape == (3, 2, 4)
        assert a.T.shape == (4, 3, 2)

    def test_squeeze_unsqueeze_flatten(self):
        a = randn(1, 3, 1, seed=0)
        assert a.squeeze().shape == (3,)
        assert a.squeeze(axis=0).shape == (3, 1)
        assert a.unsqueeze(1).shape == (1, 1, 3, 1)
        assert a.flatten().shape == (3,)

    def test_broadcast_to(self):
        a = tensor([1., 2., 3.], requires_grad=True)
        b = a.broadcast_to((2, 3))
        b.sum().backward()
        assert np.allclose(a.grad, [2, 2, 2])


class TestElementwise:
    def test_elementwise_functions(self):
        data = np.abs(randn(5, seed=0).data) + 0.1
        a = tensor(data)
        for fn in ["exp", "log", "sqrt", "tanh", "sin", "cos",
                   "sigmoid", "relu", "abs"]:
            out = getattr(a, fn)()
            assert out.shape == a.shape

    def test_relu_zero_negative(self):
        a = tensor([-1., 0., 2.])
        assert np.allclose(a.relu().data, [0, 0, 2])

    def test_softmax_stability_large_inputs(self):
        a = tensor([[10000.0, 10001.0, 9999.0]])
        out = a.softmax()
        assert np.all(np.isfinite(out.data))
        assert np.isclose(out.data.sum(), 1.0)

    def test_softmax_rows_independent(self):
        a = randn(4, 10, seed=3)
        out = a.softmax(axis=-1).data
        assert np.allclose(out.sum(axis=-1), 1.0)


class TestIndexing:
    def test_basic_indexing(self):
        a = randn(4, 5, seed=0)
        assert np.allclose(a[1:3].data, a.data[1:3])
        assert np.allclose(a[2, 1].data, a.data[2, 1])

    def test_gather_backward_scatters(self):
        a = tensor(np.zeros(4), requires_grad=True)
        idx = tensor(np.array([0, 2, 2]))
        out = a[idx] + 1.0
        out.sum().backward()
        assert np.allclose(a.grad, [1, 0, 2, 0])

    def test_embedding_gather(self):
        table = randn(5, 3, seed=1, requires_grad=True)
        ids = tensor(np.array([4, 4, 0]), dtype=np.int64)
        rows = table[ids]
        assert rows.shape == (3, 3)
        assert np.allclose(rows.data, table.data[[4, 4, 0]])
        rows.sum().backward()
        assert np.allclose(table.grad[4], [2, 2, 2])
        assert np.allclose(table.grad[0], [1, 1, 1])
        assert np.allclose(table.grad[1], 0)

    def test_masked_fill(self):
        a = randn(2, 3, seed=0, requires_grad=True)
        mask = np.array([[True, False, True], [False, True, False]])
        out = a.masked_fill(mask, -1e30)
        assert np.all(out.data[mask] == -1e30)
        out.sum().backward()
        assert np.all(a.grad[mask] == 0)


class TestConcatSplit:
    def test_concat(self):
        a = randn(2, 3, seed=0)
        b = randn(4, 3, seed=1)
        c = a.concat(b)
        assert c.shape == (6, 3)
        assert np.allclose(c.data, np.concatenate([a.data, b.data]))

    def test_concat_backward(self):
        a = randn(2, 3, seed=0, requires_grad=True)
        b = randn(3, 3, seed=1, requires_grad=True)
        c = a.concat(b)
        (c * c).sum().backward()
        assert np.allclose(a.grad, 2 * a.data)
        assert np.allclose(b.grad, 2 * b.data)

    def test_split(self):
        a = randn(4, 6, seed=0)
        parts = a.split(3, axis=1)  # numpy semantics: 3 equal sections of size 2
        assert len(parts) == 3 and parts[0].shape == (4, 2)
        assert np.allclose(parts[1].data, a.data[:, 2:4])


class TestMisc:
    def test_dtype_and_stride(self):
        a = randn(3, 4)
        assert a.dtype == np.float32
        assert len(a.stride) == 2
        f64 = tensor(np.ones((2, 2)), dtype=np.float64)
        assert f64.dtype == np.float64

    def test_requires_grad_float_only(self):
        with pytest.raises(ValueError):
            Tensor(np.array([1, 2], dtype=np.int64), requires_grad=True)

    def test_detach_shares_storage(self):
        a = tensor([1., 2.], requires_grad=True)
        d = a.detach()
        assert d.requires_grad is False
        d.data[0] = 99.0
        assert a.data[0] == 99.0

    def test_no_grad_disables_graph(self):
        a = tensor([1.], requires_grad=True)
        with no_grad():
            b = a * 2 + 1
        assert b._backward_fn is None
        assert not b.requires_grad or b._backward_fn is None

    def test_zero_sized_tensor(self):
        a = zeros(0, 3)
        assert a.shape == (0, 3)
        b = tensor(np.zeros((0, 3)), requires_grad=True)
        (b.sum()).backward()
        assert b.grad.shape == (0, 3)

    def test_invalid_shapes_raise(self):
        a = randn(2, 3)
        with pytest.raises(ValueError):
            a.reshape((5, 5))
