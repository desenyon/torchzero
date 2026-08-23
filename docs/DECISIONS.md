# Decisions Log

Append-only log of meaningful design decisions with rationale.

## D1: numpy as the sole external numerical backend

README §1 permits a numerical array library as low-level storage/kernel backend.
numpy is used for: ndarray storage, elementwise ufuncs (add/mul/exp/log/...),
`matmul`, reductions (`sum`, `max`, `mean`), `concatenate`, `take_along_axis`,
basic indexing, reshape/strides. Everything above that — the Tensor abstraction,
broadcasting gradient reduction, computation graph, reverse-mode autodiff,
module system, optimizers, transformer layers, training loop, checkpointing,
generation, tokenizer, debugger — is TorchZero code in this repository.

Boundary enforcement: no torch/tf/jax/mlx import anywhere in the repo; a static
check test scans source for forbidden imports.

## D2: No external ML framework even for reference tests

README §12 says "where possible". A trusted reference is provided by independent
double-precision reference implementations (pure Python / math module) inside
tests/reference_impls.py. Tolerances are documented per test. This keeps the
clean-environment audit lightweight and dependency-free beyond numpy.

## D3: Dynamic graph on Tensor nodes

Each Tensor carries `_parents` and `_backward_fn`; autograd does an iterative
topological sort (no recursion: avoids stack overflow on deep graphs) and
accumulates gradients into `.grad`. Graph is freed by default after backward
(`retain_graph=False`) to satisfy README §4.7.

## D4: Broadcasting via explicit unbroadcast helper

Gradients flowing into broadcast operands are reduced back to operand shapes by
`unbroadcast(grad, shape)` which sums leading dims then aligned dims. Covered by
finite-difference checks.

## D5: float32 default, float64 for gradient checks

Runtime uses float32 (like mainstream frameworks). Gradient-check tests run in
float64 for numerical accuracy.

## D6: RoPE positional encoding

Rotary embeddings chosen over learned/sinusoidal absolute positions: better
length extrapolation for the small models used here and simple to make KV-cache
correct.

## D7: RMSNorm (pre-norm) architecture

Pre-norm RMSNorm transformer blocks (Llama-style): stable training without
warmup sensitivity, fewer ops than LayerNorm, still includes LayerNorm in nn
for completeness/tests.

## D8: Debugger as real-execution tracer

debugger/ wraps a forward/backward pass with hooks recording op name, shapes,
requires_grad, grad norms, wall time, memory estimates. The `inspect` CLI
renders these traces; nothing is fabricated.
