# TorchZero Architecture

## Overview

TorchZero is a compact, transparent ML stack. A training step flows:

```
text -> BPE tokenizer -> packed id rows -> batch sampler
     -> Tensor graph (embedding, RoPE attention blocks, LM head)
     -> fused cross entropy -> reverse-mode autodiff
     -> grad clipping -> AdamW -> checkpoint / resume
     -> KV-cached generation
```

Every arrow above is TorchZero code in this repository except the numpy
primitive kernels documented below.

## Module map

| Path | Responsibility |
|------|----------------|
| `torchzero/tensor/tensor.py` | `Tensor`: storage, shape/strides/dtype, grad fields, all differentiable ops |
| `torchzero/autograd/engine.py` | dynamic-graph traversal, topological order, gradient accumulation, no_grad, graph cleanup |
| `torchzero/nn/module.py` | `Module`, `Parameter`, `ModuleList`, `Sequential`, state_dict I/O, train/eval |
| `torchzero/nn/layers.py` | `Linear`, `Embedding`, `LayerNorm`, `RMSNorm`, `Dropout`, gelu/silu |
| `torchzero/optim/optim.py` | SGD (momentum/nesterov), Adam, AdamW, global-norm clipping, LR schedule |
| `torchzero/tokenizer/bpe.py` | byte-level BPE: train/encode/decode/save/load |
| `torchzero/data/dataset.py` | split, packing, deterministic seeded batching |
| `torchzero/transformer/model.py` | decoder-only transformer + RoPE + KV cache + generation |
| `torchzero/runtime/trainer.py` | training loop, eval, LR schedule, checkpoints, resume |
| `torchzero/cli/main.py` | `train` / `generate` / `inspect` / `benchmark` commands |
| `debugger/trace.py` | forward/backward instrumentation (real timings, real grads) |
| `debugger/report.py` | text + self-contained HTML rendering |

## Tensor engine

A `Tensor` wraps a numpy `ndarray` plus:

- `grad` (ndarray or None), `requires_grad`
- `_parents`, `_backward_fn` — the dynamic computation graph edge
- memory model: numpy's row-major strided buffers; `.stride` exposes element strides

Operations build nodes eagerly. Broadcasting uses standard numpy semantics in
the forward pass; gradients are reduced back to operand shapes with
`unbroadcast()` (sum over leading broadcast dims, then over size-1 expanded
dims). This path is exhaustively checked by finite differences.

## Autodiff design

- Reverse-mode AD over the dynamic graph.
- Iterative DFS post-order produces a dependency-first order (no recursion,
  so 3000-deep chains work; tested).
- Each node's `_backward_fn(grad_out)` returns one gradient per parent; the
  engine validates count and shape and raises on invalid backwards.
- Gradients accumulate into `.grad` on every visited node that requires grad
  (multiple paths sum correctly; reused tensors accumulate).
- After a backward without `retain_graph`, every traversed node drops its
  parents/backward closure so saved activations can be freed; a second
  backward raises loudly instead of silently doing nothing.

## Transformer

Decoder-only, pre-norm:

```
x = E[ids]                                   # tied embedding
per block: x = x + Attn(RMSNorm(x)); x = x + MLP(GELU)(RMSNorm(x))
logits = RMSNorm(x) @ E^T                    # weight-tied head (real tying)
```

Attention is multi-head with RoPE (half-split rotation applied to q/k before
head transpose) and an additive `-1e30` causal mask via `masked_fill`.
Softmax is max-shifted for stability. Cross entropy is a fused stable kernel.

**KV cache**: per layer `{k, v}` of shape `(B, H, T_past, Dh)`. During cached
decode, new k/v are concatenated onto the cache and RoPE tables are sliced at
the absolute position offset, so cached generation is numerically identical to
uncached generation (tested; greedy outputs match exactly).

## External kernel boundary (README §1)

numpy is the single delegated backend. TorchZero calls numpy for exactly these
primitives:

- storage/strides/reshape/transpose/broadcast (`ndarray` machinery)
- elementwise ufuncs: add/sub/mul/div/pow/exp/log/sqrt/tanh/sin/cos/abs/max
- reductions: `sum`, `mean`, `argmax`, `put_along_axis`, `take_along_axis`
- `matmul` (all batched ranks)
- `concatenate`, `split`, `tril`, scatter (`np.add.at`)
- RNG (`np.random.default_rng`) for initialization/dropout/sampling

Everything above those kernels — broadcasting gradient reduction, the graph,
backward traversal, modules, optimizers, tokenizer, transformer logic, masking,
RoPE, caching, schedules, checkpointing — is implemented here. No torch /
TensorFlow / JAX / MLX import exists under `torchzero/` or `debugger/`; this
is enforced by a static test. PyTorch appears only inside `benchmarks/` as an
optional external measurement reference (README §12, §16).

Reference comparisons inside tests use independent plain-Python textbook
implementations (`tests/reference_impls.py`), so correctness validation does
not depend on any framework either.

## Debugger

`debugger/trace.py` records, from real executions only:

- forward: per-stage shapes, attention probability matrices, residual streams,
  norm statistics, logits/loss, wall time, activation byte estimates
- backward: per-node timed wrappers around actual backward closures, graph
  structure (shapes, parent counts, requires_grad), per-parameter gradient
  norms, global grad norm

`torchzero inspect` renders this as text (interactive commands:
stages/attn/grads/graph/params) or a self-contained HTML page.

## Deviations from README §14 structure

None material. All target directories exist as specified.
