"""Reverse-mode automatic differentiation engine.

Implements README §4:

1. dynamic computation graph (built eagerly by Tensor operations),
2. iterative topological ordering of dependencies,
3. gradient propagation backwards through the order,
4. accumulation of gradients from multiple paths,
5. scalar and tensor outputs,
6. explicit gradient arguments for non-scalar roots,
7. graph release after the pass (unless ``retain_graph``),
8. detection of invalid backward operations where practical.
"""

from __future__ import annotations

import numpy as np

_grad_enabled = True


class no_grad:
    """Context manager disabling computation-graph construction."""

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = False
        return self

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev
        return False


def get_grad_mode():
    return _grad_enabled


class GradMode:
    """Explicit toggle used by runtime code (e.g. generation/eval loops)."""

    def __init__(self, enabled):
        self._enabled = enabled

    def __enter__(self):
        global _grad_enabled
        self._prev = _grad_enabled
        _grad_enabled = self._enabled

    def __exit__(self, *exc):
        global _grad_enabled
        _grad_enabled = self._prev


def topological_order(root) -> list:
    """Iterative post-order DFS producing dependencies-first order."""
    order = []
    visited = set()
    stack = [(root, False)]
    while stack:
        node, processed = stack.pop()
        if processed:
            order.append(node)
            continue
        if id(node) in visited:
            continue
        visited.add(id(node))
        stack.append((node, True))
        for parent in node._parents:
            if id(parent) not in visited:
                stack.append((parent, False))
    return order


def backward(tensor, grad=None, retain_graph=False):
    """Run reverse mode autodiff from ``tensor``.

    ``grad`` seeds the output gradient; defaults to ones (scalar roots) or a
    full-ones tensor (non-scalar roots, per requirement 6).
    """
    if not tensor.requires_grad:
        raise RuntimeError(
            "backward called on a tensor that does not require grad")
    if getattr(tensor, "_freed", False):
        raise RuntimeError(
            "trying to backward through a graph that has already been "
            "freed; pass retain_graph=True to keep it")
    if grad is None:
        if tensor.data.size == 1:
            seed = np.ones_like(tensor.data)
        else:
            seed = np.ones_like(tensor.data)
    else:
        seed = np.asarray(grad, dtype=tensor.data.dtype)
        if seed.shape != tensor.shape:
            raise ValueError(
                f"explicit gradient shape {seed.shape} does not match "
                f"tensor shape {tensor.shape}")

    order = topological_order(tensor)

    # Detect nodes with multiple consumers in the graph so we know whether the
    # saved buffers can be freed safely.
    grads = {id(tensor): seed}
    tensor._accumulate_grad(seed.copy())

    freed_ok = not retain_graph
    for node in reversed(order):
        g = grads.pop(id(node), None)
        if g is None or node._backward_fn is None:
            continue
        parent_grads = node._backward_fn(g)
        parents = node._parents
        if len(parent_grads) != len(parents):
            raise RuntimeError(
                f"invalid backward: produced {len(parent_grads)} gradients "
                f"for {len(parents)} parents")
        for parent, pg in zip(parents, parent_grads):
            if pg is None or not parent.requires_grad:
                continue
            pg = np.asarray(pg, dtype=parent.data.dtype)
            if pg.shape != parent.shape:
                raise RuntimeError(
                    "invalid backward: gradient shape "
                    f"{pg.shape} does not match parent shape {parent.shape}")
            grads[id(parent)] = (grads.get(id(parent), 0.0) + pg) \
                if id(parent) in grads else pg
            parent._accumulate_grad(pg)

    # Graph cleanup: cut graph edges everywhere so buffers (saved activations
    # captured in backward closures) can be garbage collected. The root now
    # reports is_leaf, which lets a second backward call fail loudly instead
    # of silently doing nothing.
    if freed_ok:
        for node in order:
            node._parents = ()
            node._backward_fn = None
            node._freed = True
