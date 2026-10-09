"""Core neural network layers."""

from __future__ import annotations

import numpy as np

from .module import Module, Parameter
from ..tensor.tensor import Tensor


def _init_rng(seed=None):
    return np.random.default_rng(seed)


class Linear(Module):
    """Affine layer: y = x @ W + b, with Kaiming-uniform style init."""

    def __init__(self, in_features, out_features, bias=True, seed=None):
        super().__init__()
        rng = _init_rng(seed)
        bound = np.sqrt(1.0 / in_features)
        self.weight = Parameter(
            (rng.uniform(-bound, bound, size=(in_features, out_features))
             ).astype(np.float32))
        self.bias = Parameter(np.zeros(out_features, dtype=np.float32)) \
            if bias else None
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """Lookup table mapping integer ids to dense vectors."""

    def __init__(self, num_embeddings, embedding_dim, seed=None):
        super().__init__()
        rng = _init_rng(seed)
        self.weight = Parameter(
            (rng.standard_normal((num_embeddings, embedding_dim))
             * 0.02).astype(np.float32))
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def forward(self, indices) -> Tensor:
        if isinstance(indices, Tensor):
            if indices.dtype.kind not in "iu":
                raise TypeError("Embedding expects integer indices")
            idx = indices.data.astype(np.int64)
        else:
            idx = np.asarray(indices, dtype=np.int64)
        if idx.size and (idx.max() >= self.num_embeddings or idx.min() < 0):
            raise IndexError(
                f"embedding index out of range "
                f"[0, {self.num_embeddings}): got min {idx.min()}, max {idx.max()}")
        return self.weight[idx]


class LayerNorm(Module):
    """Layer normalization over the last dimension with learnable gain/bias."""

    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        shape = (normalized_shape,) if isinstance(normalized_shape, int) \
            else tuple(normalized_shape)
        self.weight = Parameter(np.ones(shape, dtype=np.float32))
        self.bias = Parameter(np.zeros(shape, dtype=np.float32))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered ** 2).mean(axis=-1, keepdims=True)
        x_hat = centered / ((var + self.eps) ** 0.5)
        return x_hat * self.weight + self.bias


class RMSNorm(Module):
    """Root-mean-square normalization (no mean subtraction, no bias)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = Parameter(np.ones(dim, dtype=np.float32))
        self.dim = dim
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        scale = ((x * x).mean(axis=-1, keepdims=True) + self.eps) ** -0.5
        return (x * scale) * self.weight


class Dropout(Module):
    """Inverted dropout; active only in training mode."""

    def __init__(self, p=0.1, seed=None):
        super().__init__()
        if not 0.0 <= p < 1.0:
            raise ValueError(f"dropout probability must be in [0, 1): {p}")
        self.p = p
        self.seed = seed
        self._rng = _init_rng(seed)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x
        mask = (self._rng.random(x.shape) >= self.p).astype(x.data.dtype)
        mask /= (1.0 - self.p)
        return x * Tensor(mask)


def gelu(x: Tensor) -> Tensor:
    """GELU tanh approximation as a composition of primitives so autodiff is
    exact w.r.t. this approximation."""
    kBeta = 0.7978845608028654   # sqrt(2/pi)
    kKappa = 0.044715
    return x * 0.5 * (
        1.0 + (((kBeta * x) + (kBeta * kKappa * x ** 3))).tanh())


def silu(x: Tensor) -> Tensor:
    return x * x.sigmoid()
