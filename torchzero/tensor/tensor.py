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
        if isinstance(exponent, Tensor) or not np.isscalar(exponent):
            return _ensure_tensor(exponent).__rpow__(self)
        out_data = self.data ** exponent
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            if exponent == 0:
                return (np.zeros_like(self.data),)
            return (unbroadcast(g * exponent * self.data ** (exponent - 1),
                                self.shape),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def __rpow__(self, base):
        base_t = _ensure_tensor(base)
        out_data = base_t.data ** self.data
        parents = (base_t, self)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            gb = ge = None
            if base_t.requires_grad:
                # x**0 is constant, including x=0; avoid evaluating 0**-1.
                power = np.zeros_like(out_data)
                np.power(base_t.data, self.data - 1, out=power,
                         where=self.data != 0)
                gb = unbroadcast(g * self.data * power, base_t.shape)
            if self.requires_grad:
                ge = unbroadcast(g * out_data * np.log(base_t.data), self.shape)
            return gb, ge

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

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
            # NumPy matmul promotes vectors to matrices and then removes the
            # synthetic axes. Undo that removal before the common VJP.
            am = a[np.newaxis, :] if a.ndim == 1 else a
            bm = b[:, np.newaxis] if b.ndim == 1 else b
            gm = g
            if b.ndim == 1:
                gm = np.expand_dims(gm, -1)
            if a.ndim == 1:
                gm = np.expand_dims(gm, -2)
            ga = gm @ np.swapaxes(bm, -1, -2)
            gb = np.swapaxes(am, -1, -2) @ gm
            if a.ndim == 1:
                ga = np.squeeze(ga, -2)
            if b.ndim == 1:
                gb = np.squeeze(gb, -1)
            return unbroadcast(ga, a.shape), unbroadcast(gb, b.shape)

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
    def _expand_reduction_grad(self, g, out_shape, axis, keepdims):
        """Expand a reduction output gradient back to input rank."""
        if keepdims:
            return g
        if axis is None:
            axes = tuple(range(self.ndim))
        elif isinstance(axis, tuple):
            axes = axis
        else:
            axes = (axis,)
        return np.expand_dims(
            g, tuple(sorted(a % self.ndim for a in axes)))

    def sum(self, axis=None, keepdims=False):
        out_data = self.data.sum(axis=axis, keepdims=keepdims)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)

        def bfn(g):
            if axis is None:
                return (np.broadcast_to(g, self.shape).copy(),)
            gexp = self._expand_reduction_grad(g, out_data.shape, axis, keepdims)
            return (np.broadcast_to(gexp, self.shape).copy(),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def mean(self, axis=None, keepdims=False):
        out_data = self.data.mean(axis=axis, keepdims=keepdims)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        n = self.data.size / out_data.size

        def bfn(g):
            gexp = self._expand_reduction_grad(g, out_data.shape, axis, keepdims)
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
        return -((-self).max(axis=axis, keepdims=keepdims))

    # ------------------------------------------------------------- shape ops
    def transpose(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = tuple(axes[0])
        axes = axes if axes else tuple(reversed(range(self.ndim)))
        out_data = self.data.transpose(axes)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        inv = np.argsort([axis % self.ndim for axis in axes])

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
        idx = _index_array(indices)
        axis = _normalize_axis(axis, self.ndim)
        out_data = np.take_along_axis(self.data, idx, axis=axis)
        if not get_grad_mode() or not self.requires_grad:
            return Tensor(out_data)
        coords = _axis_coordinates(idx, out_data.shape, axis)

        def bfn(g):
            # Non-axis dimensions can broadcast in take_along_axis.
            shape = list(out_data.shape)
            shape[axis] = self.shape[axis]
            gfull = np.zeros(shape, dtype=self.dtype)
            np.add.at(gfull, coords, g)
            return (unbroadcast(gfull, self.shape),)

        return Tensor(out_data, True, _parents=(self,), _backward_fn=bfn)

    def scatter_add(self, indices, src, axis=0):
        """Return a differentiable copy with source values added at indices.

        Full-rank indices must match src and all non-scatter destination
        dimensions. A 1-D index vector also supports whole-row scatter at
        axis=0, preserving the original embedding-style API.
        """
        idx = _index_array(indices)
        src = _ensure_tensor(src)
        axis = _normalize_axis(axis, self.ndim)
        if idx.ndim == 1 and self.ndim > 1 and axis == 0:
            expected = (len(idx),) + self.shape[1:]
            if src.shape != expected:
                raise ValueError(f"src shape must be {expected}")
            coords = (idx,)
        else:
            if idx.ndim != self.ndim or src.shape != idx.shape:
                raise ValueError("indices and src must have matching destination rank and shape")
            if any(idx.shape[d] != self.shape[d] for d in range(self.ndim) if d != axis):
                raise ValueError("non-scatter dimensions must match destination")
            coords = _axis_coordinates(idx, idx.shape, axis)
        out_data = self.data.copy()
        np.add.at(out_data, coords, src.data)
        parents = (self, src)
        if not get_grad_mode() or not self._requires_any(parents):
            return Tensor(out_data)

        def bfn(g):
            return g.copy(), g[coords]

        return Tensor(out_data, True, _parents=parents, _backward_fn=bfn)

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
                gfull = np.zeros_like(self.data)
                sl = [slice(None)] * g.ndim
                sl[axis] = slice(start, start + size_on_axis)
                gfull[tuple(sl)] = g
                return (gfull,)
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


def _normalize_axis(axis, ndim):
    if not isinstance(axis, (int, np.integer)) or not -ndim <= axis < ndim:
        raise ValueError(f"axis {axis} out of bounds for rank {ndim}")
    return axis % ndim


def _index_array(indices):
    idx = indices.data if isinstance(indices, Tensor) else np.asarray(indices)
    if idx.dtype.kind not in "iu":
        raise TypeError("indices must be integers")
    return idx.copy()


def _axis_coordinates(indices, shape, axis):
    return tuple(np.broadcast_to(indices, shape) if d == axis else
                 np.arange(size).reshape((1,) * d + (size,) + (1,) * (len(shape) - d - 1))
                 for d, size in enumerate(shape))


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
