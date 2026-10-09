"""TorchZero: a from-scratch neural network and transformer stack.

Public API. The only external numerical dependency is numpy (low-level array
storage and primitive kernels); everything else is implemented in this package.
"""

from .tensor import Tensor, tensor, zeros, randn, arange, eye
from .autograd import backward, no_grad, GradMode

__version__ = "0.2.0"

__all__ = [
    "Tensor", "tensor", "zeros", "randn", "arange", "eye",
    "backward", "no_grad", "GradMode",
]
