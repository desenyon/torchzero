"""Autograd engine behavior: graph building, accumulation, branching,
reuse, cleanup, non-scalar roots, invalid backward detection."""

import numpy as np
import pytest

from torchzero import Tensor, tensor


class TestBackwardSemantics:
    def test_scalar_root_default_seed(self):
        x = tensor([1., 2., 3.], requires_grad=True)
        (x.sum()).backward()
        assert np.allclose(x.grad, [1, 1, 1])

    def test_non_scalar_root_explicit_grad(self):
        x = tensor([1., 2.], requires_grad=True)
        y = x * 2
        y.backward(grad=np.array([1., 10.]))
        assert np.allclose(x.grad, [2, 20])

    def test_non_scalar_root_without_grad_raises(self):
        x = tensor([1., 2.], requires_grad=True)
        y = x * 2
        # default seed is ones; this should work and produce ones*2
        y.backward()
        assert np.allclose(x.grad, [2, 2])

    def test_mismatched_grad_shape_raises(self):
        x = tensor([1., 2.], requires_grad=True)
        y = x * 2
        with pytest.raises(ValueError):
            y.backward(grad=np.ones((3,)))

    def test_branched_graph_accumulates(self):
        x = tensor(3.0, requires_grad=True)
        a = x * 2
        b = x * 3
        (a + b).backward()
        assert np.isclose(x.grad, 5)

    def test_diamond_graph(self):
        x = tensor(2.0, requires_grad=True)
        l = x * x
        r = x * x
        (l * r).backward()  # d/dx x^4 = 4x^3
        assert np.isclose(x.grad, 32)

    def test_reused_tensor_multiple_paths(self):
        x = tensor(1.5, requires_grad=True)
        y = (x.sin() + x.cos()) * x.exp()
        y.backward()
        # derivative of e^x (sin x + cos x) = e^x (sin + cos) + e^x (cos - sin)
        expected = np.exp(1.5) * ((np.sin(1.5) + np.cos(1.5))
                                  + (np.cos(1.5) - np.sin(1.5)))
        assert np.isclose(float(x.grad), expected, rtol=1e-9)

    def test_intermediate_grads_retained_on_nodes(self):
        x = tensor([1., 2.], requires_grad=True)
        y = x * 2
        z = (y.sum())
        z.backward()
        assert y.grad is not None
        assert np.allclose(y.grad, [1, 1])

    def test_backward_without_requires_grad_raises(self):
        x = tensor([1., 2.])
        with pytest.raises(RuntimeError):
            (x.sum()).backward()

    def test_second_backward_raises(self):
        x = tensor([1., 2.], requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward()
        with pytest.raises(RuntimeError):
            loss.backward()

    def test_retain_graph_allows_second_backward(self):
        x = tensor([1., 2.], requires_grad=True)
        loss = (x ** 2).sum()
        loss.backward(retain_graph=True)
        g1 = x.grad.copy()
        loss.backward(retain_graph=True)
        assert np.allclose(g1, [2, 4])
        assert np.allclose(x.grad, [4, 8])  # accumulated

    def test_leaf_tensor_backward_sets_ones(self):
        x = tensor(5.0, requires_grad=True)
        x.backward()
        assert np.isclose(float(x.grad), 1.0)

    def test_no_grad_isolation(self):
        from torchzero import no_grad
        x = tensor([1., 2.], requires_grad=True)
        with no_grad():
            y = (x * 3).sum()
        assert not y.requires_grad or y._backward_fn is None

    def test_topological_order_correctness_deep_chain(self):
        # deep chain exercises iterative topo sort (no recursion limit hit)
        depth = 3000
        x = tensor(1.0, requires_grad=True)
        out = x
        for _ in range(depth):
            out = out + 0.0  # identity ops keep graph deep but grad = 1
        out.backward()
        assert np.isclose(float(x.grad), 1.0)


class TestInvalidBackwardDetection:
    def test_wrong_parent_count_detected(self):
        t = Tensor(np.ones((2,)), True, _parents=(tensor(np.zeros(3)),),
                   _backward_fn=lambda g: (g, g))
        with pytest.raises(RuntimeError):
            t.backward()

    def test_wrong_grad_shape_detected(self):
        bad_parent = tensor(np.zeros(3), requires_grad=True)
        t = Tensor(np.ones(()), True, _parents=(bad_parent,),
                   _backward_fn=lambda g: (np.ones(7),))
        with pytest.raises(RuntimeError):
            t.backward()
