"""TorchZero Tensor.

A Tensor wraps a numpy ``ndarray`` (the delegated low-level storage/kernel
backend, see ARCHITECTURE.md "External kernel boundary") and carries the
metadata TorchZero's reverse-mode autodiff needs:

* ``data``          : numpy ndarray storage (strided memory model of numpy)
* ``shape``         : tuple of ints
* ``stride``        : element strides of the underlying buffer
* ``dtype``         : numpy dtype
* ``grad``          : gradient storage (ndarray or None)
* ``requires_grad`` : gradient requirement flag
* ``_parents``      : parent tensors in the dynamic computation graph
* ``_backward_fn``  : function mapping outgoing grad -> tuple of parent grads

Operations build the graph eagerly; ``torchzero.autograd.engine.backward``
performs the reverse pass.
"""

from __future__ import annotations

import numpy as np

from ..autograd.engine import get_grad_mode


def _to_array(data, dtype=None):
    if isinstance(data, Tensor):
        arr = data.data
    else:
        arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Reduce ``grad`` back to ``shape`` after numpy broadcasting.

    Sums over leading broadcast dimensions, then over dimensions that were
    expanded from size 1.
    """
    # Sum extra leading dimensions.
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # Sum dimensions that were broadcast from 1 to n.
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] != 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad.reshape(shape)


class Tensor:
    __array_priority__ = 1000.0  # ensure numpy defers to our __r*__ operators

    def __init__(self, data, requires_grad=False, dtype=None,
                 _parents=(), _backward_fn=None, _name=None):
        self.data = _to_array(data, dtype=dtype)
        self.grad = None
        self.requires_grad = bool(requires_grad)
        self._parents = tuple(_parents)
        self._backward_fn = _backward_fn
        self._name = _name
        self._retained = False
        if self.dtype.kind != "f" and requires_grad:
            raise ValueError("only floating point tensors support requires_grad")

    # ------------------------------------------------------------- properties
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def stride(self):
        """Element strides of the underlying numpy buffer (documented memory
        model: row-major C-contiguous strided layout managed by numpy)."""
        return self.data.strides

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def device(self):
        return "cpu"

    def item(self):
        return self.data.item()

    def numpy(self):
        return self.data

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        req = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{req})"

    # ------------------------------------------------------------ graph utils
    def detach(self):
        """Return a new Tensor sharing storage but cut off from the graph."""
        return Tensor(self.data, requires_grad=False)

    def zero_grad(self):
        self.grad = None

    @property
    def is_leaf(self):
        return not self._parents

    def retain_grad(self):
        self._retained = True

    def _accumulate_grad(self, g):
        if self.grad is None:
            self.grad = g
        else:
            self.grad = self.grad + g

    # ------------------------------------------------------------- arithmetic
    def __add__(self, other):
        other = _ensure_tensor(other)
        out_data = self.data + other.data
        if not get_grad_mode():
            return Tensor(out_data)
        parents = (self, other)

        def bfn(g):
            return (unbroadcast(g, self.shape), unbroadcast(g, other.shape))

        return Tensor(out_data, self._requires_any(parents), _parents=parents,
                      _backward_fn=bfn if self._requires_any(parents) else None)

    def __radd__(self, other):
        return _ensure_tensor(other) + self

    def __sub__(self, other):
        other = _ensure_tensor(other)
        out_data = self.data - other.data
        parents = (self, other)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            return (unbroadcast(g, self.shape), unbroadcast(-g, other.shape))

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

    def __rsub__(self, other):
        return _ensure_tensor(other) - self

    def __mul__(self, other):
        other = _ensure_tensor(other)
        out_data = self.data * other.data
        parents = (self, other)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            return (unbroadcast(g * other.data, self.shape),
                    unbroadcast(g * self.data, other.shape))

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

    def __rmul__(self, other):
        return _ensure_tensor(other) * self

    def __truediv__(self, other):
        other = _ensure_tensor(other)
        out_data = self.data / other.data
        parents = (self, other)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            g_self = g / other.data
            g_other = -g * self.data / (other.data ** 2)
            return (unbroadcast(g_self, self.shape),
                    unbroadcast(g_other, other.shape))

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

    def __rtruediv__(self, other):
        return _ensure_tensor(other) / self

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            return exponent.__rpow__(self)
        out_data = self.data ** exponent
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            return (unbroadcast(g * exponent * self.data ** (exponent - 1),
                                self.shape),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def __rpow__(self, base):
        base_t = _ensure_tensor(base)
        out_data = base_t.data ** self.data
        if not get_grad_mode():
            return Tensor(out_data)

        def bfn(g):
            ln_b = np.log(np.where(base_t.data > 0, base_t.data, 1.0))
            return (unbroadcast(g * out_data * ln_b, base_t.shape),
                    unbroadcast(g * out_data / np.where(base_t.data != 0, base_t.data, 1.0), self.shape))

        return Tensor(out_data, self.requires_grad or base_t.requires_grad,
                      _parents=(base_t, self), _backward_fn=bfn)

    def __neg__(self):
        return self * -1.0

    def __matmul__(self, other):
        other = _ensure_tensor(other)
        out_data = self.data @ other.data
        parents = (self, other)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            a, b = self.data, other.data
            if a.ndim == 1 and b.ndim == 1:      # vec @ vec -> scalar
                return (g * b, g * a)
            ga = None
            gb = None
            if a.ndim == 1:                       # vec @ mat
                ga = (g @ b.T)
                gb = np.outer(a, g) * np.ones_like(b)
            elif b.ndim == 1:                     # mat @ vec
                ga = np.outer(g, b) * np.ones_like(a)
                gb = a.T @ g
            elif a.ndim == 2 and b.ndim == 2:
                ga = g @ b.T
                gb = a.T @ g
            else:                                 # batched (>=3d)
                ga = g @ np.swapaxes(b, -1, -2)
                gb = np.swapaxes(a, -1, -2) @ g
                ga = unbroadcast(ga, a.shape)
                gb = unbroadcast(gb, b.shape)
            return (ga, gb)

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

    def __rmatmul__(self, other):
        return _ensure_tensor(other) @ self

    # ------------------------------------------------------------ comparisons
    def __eq__(self, other):
        if isinstance(other, Tensor):
            return np.array_equal(self.data, other.data)
        return NotImplemented

    __hash__ = object.__hash__

    # ------------------------------------------------------------- reductions
    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            if axis is None:
                return (np.broadcast_to(g, self.shape).copy(),)
            gexp = np.expand_dims(g, axis) if not keepdims else g
            axes = axis if isinstance(axis, tuple) else (axis,)
            for ax in sorted(a % self.ndim for a in axes):
                gexp = np.expand_dims(gexp, ax)
            return (np.broadcast_to(gexp, self.shape).copy(),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        n = self.data.size / out_data.size

        def bfn(g):
            gexp = np.broadcast_to(g, out_data.shape).copy()
            if axis is not None and not keepdims:
                axes = axis if isinstance(axis, tuple) else (axis,)
                for ax in sorted(a % self.ndim for a in axes):
                    gexp = np.expand_dims(gexp, ax)
            return (np.broadcast_to(gexp / n, self.shape).copy(),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def max(self, axis=None, keepdims=False):
        """Max reduction; gradient flows to (first) argmax elements."""
        out_data = self.data.max(axis=axis, keepdims=keepdims)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        idx = self.data.argmax(axis=axis)

        def bfn(g):
            gfull = np.zeros_like(self.data)
            if axis is None:
                gfull.reshape(-1)[idx] = g
                return (gfull,)
            gexp = np.expand_dims(g, axis) if not keepdims else g
            ax = axis if isinstance(axis, int) else axis[0]
            np.put_along_axis(gfull, np.expand_dims(idx, ax), gexp, axis=ax)
            return (gfull,)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def min(self, axis=None, keepdims=False):
        neg = (-self).max(axis=axis, keepdims=keepdims)
        out = Tensor(-neg.data)
        if not neg._backward_fn:
            return out

        orig = neg._backward_fn

        def bfn(g):
            return tuple(-x for x in orig(-g))

        return Tensor(out.data, True, _parents=neg._parents, _backward_fn=bfn)

    # ------------------------------------------------------------- shape ops
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes if axes else tuple(reversed(range(self.ndim)))
        out_data = self.data.transpose(axes)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        inv = np.argsort(axes)

        def bfn(g):
            return (g.transpose(tuple(inv)),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    @property
    def T(self):
        return self.transpose()

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        out_data = self.data.reshape(shape)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            return (g.reshape(self.shape),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def view(self, *shape):
        """Alias of reshape on the contiguous buffer (numpy strided memory
        model: reshape returns a view when possible, otherwise a copy)."""
        return self.reshape(*shape)

    def flatten(self, start_dim=0, end_dim=-1):
        shape = self.shape
        end = end_dim % self.ndim
        new_shape = (shape[:start_dim]
                     + (int(np.prod(shape[start_dim:end + 1])),)
                     + shape[end + 1:])
        return self.reshape(new_shape)

    def squeeze(self, axis=None):
        return Tensor(self.data.squeeze(axis)) if not (
            get_grad_mode() and self.requires_grad) else \
            self.reshape(self.data.squeeze(axis).shape)

    def unsqueeze(self, axis):
        shape = list(self.shape)
        shape.insert(axis % (self.ndim + 1), 1)
        return self.reshape(tuple(shape))

    def broadcast_to(self, shape):
        out_data = np.broadcast_to(self.data, shape)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(np.ascontiguousarray(out_data))

        def bfn(g):
            return (unbroadcast(g, self.shape),)

        return Tensor(np.ascontiguousarray(out_data), True, _parents=(self,),
                      _backward_fn=bfn)

    # ------------------------------------------------------------- elementwise
    def _unary(self, fwd, bwd):
        out_data = fwd(self.data)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            return (bwd(g, out_data, self.data),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def exp(self):
        return self._unary(np.exp, lambda g, o, x: g * o)

    def log(self):
        return self._unary(np.log, lambda g, o, x: g / x)

    def sqrt(self):
        return self._unary(np.sqrt, lambda g, o, x: g / (2 * o))

    def tanh(self):
        return self._unary(np.tanh, lambda g, o, x: g * (1 - o ** 2))

    def sin(self):
        return self._unary(np.sin, lambda g, o, x: g * np.cos(x))

    def cos(self):
        return self._unary(np.cos, lambda g, o, x: -g * np.sin(x))

    def sigmoid(self):
        return self._unary(
            lambda x: 1.0 / (1.0 + np.exp(-x)),
            lambda g, o, x: g * o * (1 - o))

    def relu(self):
        return self._unary(lambda x: np.maximum(x, 0.0),
                           lambda g, o, x: g * (x > 0))

    def abs(self):
        return self._unary(np.abs, lambda g, o, x: g * np.sign(x))

    # ------------------------------------------------------- softmax family
    def softmax(self, axis=-1):
        """Numerically stable softmax (max-subtraction before exp)."""
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        out_data = e / e.sum(axis=axis, keepdims=True)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            inner = (g * out_data).sum(axis=axis, keepdims=True)
            return (out_data * (g - inner),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def log_softmax(self, axis=-1):
        shifted = self.data - self.data.max(axis=axis, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
        out_data = shifted - lse
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            return (g - np.exp(out_data) * g.sum(axis=axis, keepdims=True),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    # ------------------------------------------------------------- indexing
    def __getitem__(self, index):
        """Basic and advanced indexing. Integer-array indexing acts as gather;
        gradients are scattered back to the selected positions."""
        idx = index.data.astype(np.int64) if isinstance(index, Tensor) else index
        out_data = self.data[idx]
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            gfull = np.zeros_like(self.data)
            np.add.at(gfull, idx, g)
            return (gfull,)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def gather(self, indices, axis=-1):
        """Gather elements along ``axis``. ``indices`` is an int Tensor of the
        same ndim. Gradient scatter-adds back to gathered source positions."""
        idx = indices.data.astype(np.int64)
        out_data = np.take_along_axis(self.data, idx, axis=axis)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            gfull = np.zeros_like(self.data)
            np.put_along_axis(gfull, idx, g, axis=axis)
            return (gfull,)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def scatter_add(self, indices, src, axis=0):
        """Scatter-add ``src`` values into this tensor at ``indices`` along
        ``axis``. Used by embedding-style gathers in reverse."""
        idx = indices.data.astype(np.int64)
        src_arr = src.data if isinstance(src, Tensor) else np.asarray(src)
        out_data = self.data.copy()
        np.add.at(out_data, idx, src_arr)
        return Tensor(out_data, requires_grad=self.requires_grad)

    def masked_fill(self, mask, value):
        """``mask`` is a boolean array/Tensor; filled positions have zero
        gradient flow."""
        m = mask.data if isinstance(mask, Tensor) else np.asarray(mask)
        out_data = np.where(m, value, self.data)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            return (np.where(m, 0.0, g),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    # ----------------------------------------------------------- concatenation
    def concat(self, others, axis=0):
        """Concatenate this tensor with ``others`` (list of Tensors)."""
        if not isinstance(others, (list, tuple)):
            others = [others]
        tensors = [self] + [_ensure_tensor(t) for t in others]
        out_data = np.concatenate([t.data for t in tensors], axis=axis)
        any_req = any(t.requires_grad for t in tensors)
        if not get_grad_mode() or not any_req:
            return Tensor(out_data)
        offsets = []
        off = 0
        for t in tensors:
            offsets.append(off)
            off += t.shape[axis]

        def bfn(g):
            grads = []
            for t, start in zip(tensors, offsets):
                end = start + t.shape[axis]
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(start, end)
                grads.append(g[tuple(sl)])
            return tuple(grads)

        return Tensor(out_data, True, _parents=tuple(tensors), _backward_fn=bfn)

    def split(self, sections_or_size, axis=0):
        """Split along axis; returns list of Tensors sharing graph parents."""
        parts = np.split(self.data, sections_or_size, axis=axis)
        results = []
        starts = []
        off = 0
        for p in parts:
            starts.append(off)
            off += p.shape[axis]
        if not (get_grad_mode() and self.requires_grad):
            return [Tensor(p) for p in parts]

        def make_bfn(start, size_on_axis):
            def bfn(g):
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(start, start + size_on_axis)
                return (g[tuple(sl)],)
            return bfn

        for p, s in zip(parts, starts):
            results.append(Tensor(p, True, _parents=(self,),
                                  _backward_fn=make_bfn(s, p.shape[axis])))
        return results

    # -------------------------------------------------------------- losses
    @staticmethod
    def cross_entropy(logits, targets):
        """Fused cross entropy between logits (..., V) float Tensor and integer
        targets (Tensor or array of same leading shape). Returns scalar Tensor
        equal to mean negative log likelihood. Numerically stable."""
        targets_arr = targets.data.astype(np.int64) if isinstance(targets, Tensor) \
            else np.asarray(targets, dtype=np.int64)
        x = logits.data
        shifted = x - x.max(axis=-1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(axis=-1))
        picked = np.take_along_axis(
            shifted, np.expand_dims(targets_arr, -1), -1).squeeze(-1)
        nll = lse - picked
        out_data = np.asarray(nll.mean())
        if not get_grad_mode() or not logits.requires_grad:
            return Tensor(out_data)
        flat_x = x.reshape(-1, x.shape[-1])
        n_rows = flat_x.shape[0]

        def bfn(g):
            soft = np.exp(shifted) / np.exp(shifted).sum(-1, keepdims=True)
            flat_soft = soft.reshape(-1, x.shape[-1])
            flat_t = targets_arr.reshape(-1)
            d = flat_soft.copy()
            d[np.arange(n_rows), flat_t] -= 1.0
            d /= nll.size  # mean reduction
            return ((d.reshape(x.shape)) * float(g.item() if g.ndim == 0 else g),)

        return Tensor(out_data, True, _parents=(logits,), _backward_fn=bfn)

    # --------------------------------------------------------------- helper
    @staticmethod
    def _requires_any(parents):
        return any(p.requires_grad for p in parents)

    def backward(self, grad=None, retain_graph=False):
        from ..autograd.engine import backward as engine_backward
        engine_backward(self, grad=grad, retain_graph=retain_graph)


def _ensure_tensor(value):
    return value if isinstance(value, Tensor) else Tensor(value)


def tensor(data, requires_grad=False, dtype=None):
    return Tensor(data, requires_grad=requires_grad, dtype=dtype)


def zeros(*shape, requires_grad=False, dtype=np.float32):
    return Tensor(np.zeros(shape, dtype=dtype), requires_grad)


def randn(*shape, requires_grad=False, dtype=np.float32, seed=None):
    rng = np.random.default_rng(seed)
    return Tensor(rng.standard_normal(shape).astype(dtype), requires_grad)


def arange(n, dtype=np.int64):
    return Tensor(np.arange(n, dtype=dtype))


def eye(n, dtype=np.float32):
    return Tensor(np.eye(n, dtype=dtype))
