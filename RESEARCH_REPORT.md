# TorchZero Research Report

## Abstract

TorchZero is a from-scratch neural-network stack — tensor engine, dynamic
reverse-mode autodiff, module system, optimizers, byte-level BPE tokenizer,
data pipeline, decoder-only transformer, training runtime with checkpointing,
KV-cached generation, and an execution debugger — built without any automatic
differentiation framework. The only external numerical dependency is numpy,
used strictly as a primitive kernel backend. We validate the stack three ways:
finite-difference gradient checks over every differentiable primitive,
comparison against independent plain-Python textbook reference
implementations, and full-model parity against PyTorch on identical
architectures (forward logits agree to ~3e-7 relative; gradients to ~1e-6).
Benchmarks show TorchZero trains small transformers at 2–30M tokens/s on CPU
and 4.5–6.8x slower per step than PyTorch eager, dominated by Python-level
graph overhead rather than kernels. KV caching speeds autoregressive
generation 1.7–2.3x at the measured prompt lengths while reproducing greedy
outputs exactly.

## Motivation

Modern frameworks hide the mechanics that make deep learning work: how
gradients actually flow, why broadcasting must reduce correctly, what a KV
cache saves, where a training step spends time. TorchZero's goal is not to
compete feature-for-feature but to build a compact system whose core
mechanics are transparent, correct, and measurable — and sufficient to train
and run a real decoder-only language model from first principles.

## System Architecture

See `ARCHITECTURE.md` for the full module map, tensor/autodiff design, the
transformer layout (pre-norm RMSNorm blocks, RoPE causal multi-head
attention, GELU MLPs, weight-tied head), the external-kernel boundary
(numpy delegation list), and the debugger design. Deviations from the target
repository structure: none material.

## Automatic Differentiation Design

- Dynamic graph: every operation returns a `Tensor` carrying parent links and
  a backward closure.
- Reverse pass: iterative DFS post-order (recursion-free; verified on 3000-deep
  chains), gradients seeded with ones for scalar roots or explicit arrays for
  non-scalar roots.
- Accumulation: gradients from multiple paths sum into `.grad`; reused tensors
  in diamond graphs accumulate correctly (finite-difference verified).
- Broadcasting: forward uses numpy broadcasting; backward reduces via
  `unbroadcast` (sum leading dims, then size-1 dims).
- Graph lifetime: after a non-retaining backward all traversed nodes drop
  graph edges so saved activations are collectable; a second backward raises.
- Invalid backwards are detected: wrong gradient counts/shapes raise.

Every differentiable primitive is covered by central-difference finite
difference checks (`tests/test_gradcheck.py`, float64, h=1e-6): arithmetic,
pow, matmul (2-D, batched, both operands), reductions (sum/mean/max incl.
axis tuples and keepdims), shape ops (transpose/reshape/broadcast_to),
elementwise (exp/log/sqrt/tanh/sigmoid/relu/abs), softmax/log_softmax,
indexing/gather, concat/split, masked_fill, LayerNorm/RMSNorm compositions,
cross entropy, and attention-like expressions.

## Transformer Implementation

Decoder-only, Llama-style pre-norm: token embedding tied to the output head
through an autograd transpose node, RMSNorm, rotary positional embeddings
(half-split convention) applied to q/k before head transposition, causal
masking via `-1e30` `masked_fill`, multi-head attention, GELU (tanh
approximation) MLPs, residual connections, fused stable cross entropy,
greedy/temperature/top-k sampling, and per-layer KV caches whose RoPE tables
are offset by cache length during incremental decode. The execution path from
tokens to logits is ~200 lines of straight-line TorchZero code.

## Experimental Method

All experiments run on Apple Silicon CPU (Python 3.14, numpy 2.5.2). Raw
results are committed under `experiments/results/*.json`; figures are
regenerated programmatically by `research/generate_figures.py` into
`research/figures/`. Nothing in this report is quoted outside stored results.

Correctness methodology, in increasing strength:

1. Finite differences (central, float64) against analytical gradients.
2. Independent textbook implementations in plain Python (`tests/reference_impls.py`)
   for matmul, softmax, norms, causal attention, cross entropy, Adam steps.
3. Full-model parity vs PyTorch: identical weights transferred into a mirror
   architecture (torch used only as measurement reference); compare logits,
   loss, and per-parameter gradients.

Documented tolerances (float32 runtime vs float64 references): elementwise /
norm outputs atol 2e-4; attention probabilities atol 1e-4; loss atol 1e-3;
optimizer trajectories atol 1e-5 per step; parity errors reported raw below.

## Research Questions

### RQ1. How closely can a compact autodiff engine reproduce reference framework gradients?

Effectively exactly, within float32 noise. On three configurations
(vocab/dim/layers: 64/32/2, 128/48/3, 256/64/2):

| Case | logits max abs err | relative | loss abs err | max param grad rel err |
|------|--------------------|----------|--------------|------------------------|
| 64/32/2  | 1.68e-07 | 2.90e-07 | 4.80e-07 | 8.56e-07 |
| 128/48/3 | 2.58e-07 | 5.88e-07 | 4.59e-07 | 1.09e-06 |
| 256/64/2 | 3.21e-07 | 5.53e-07 | 6.43e-07 | 8.66e-07 |

(source: `experiments/results/parity_vs_torch.json`). Every parameter
gradient — including the tied embedding receiving gradient from both input
and output paths — agrees with PyTorch to ≤1.1e-6 relative.

### RQ2. Which operations dominate runtime in a small from scratch transformer?

Forward-side module timing (B8 T64 dim128 4 layers, `op_profile.json`):
MLP 13.2 ms, attention 4.6 ms, RMSNorm <1 ms of a 19.8 ms forward wall —
the MLP's two large matmuls dominate. Backward wall (21.2 ms) exceeds forward
because each primitive re-executes on saved activations plus allocates
gradient buffers in Python. The dominant cost overall is per-node Python
overhead: TorchZero builds one Python object + closure per op, which is why
step latency scales with graph size rather than FLOPs.

### RQ3. How does sequence length change the fraction of time spent in attention?

(`attention_scaling.json`, figure `research/figures/attention_scaling.png`)

| seq len | step ms | attention ms | fraction |
|---------|---------|--------------|----------|
| 16  | 18.1  | 1.5  | 8.2% |
| 32  | 26.5  | 2.1  | 8.0% |
| 64  | 49.4  | 5.2  | 10.6% |
| 128 | 117.2 | 15.0 | 12.8% |
| 256 | 277.5 | 54.4 | 19.6% |

Attention cost grows quadratically (scores matrix T² dominates) while MLP and
overhead costs grow linearly, so the attention fraction rises from ~8% at
T=16 to ~20% at T=256. Even at 256 tokens it remains a minority of step time
in this from-scratch setting because Python overhead inflates everything else.

### RQ4. What memory costs arise from retaining the dynamic computation graph?

The graph retains every intermediate activation inside backward closures.
Measured activation footprint for a single traced forward (debugger,
B=1 T=6..8 tiny model) is tens of KB; for the B16 T128 D256 benchmark model
each step holds all block inputs, attention probability tensors (B·H·T²
float32 ≈ 16·8·128² ·4B ≈ 67 MB total across layers), and MLP hiddens until
backward completes. Process peak RSS during the full benchmark run was
recorded in `benchmark.json`. Retaining graphs (`retain_graph=True`) doubles
effective lifetime but we free eagerly by default; the practical implication
is that memory scales with T² through attention probabilities even though
attention compute is only ~10–20% of step time (RQ3).

### RQ5. How much does KV caching improve autoregressive generation?

(`kv_cache_ablation.json`, figure `kv_cache_ablation.png`; dim96 4-layer
model, greedy determinism checked)

| prompt / new tokens | cached tok/s | uncached tok/s | speedup | greedy outputs match |
|---------------------|--------------|----------------|---------|----------------------|
| 8 / 24    | 1184 | 710 | 1.67x | exact |
| 16 / 48   | 1174 | 516 | 2.27x | exact |

Speedup grows with generation length as the uncached path's quadratic
re-computation compounds; cached throughput stays flat (~1170 tok/s) because
each decode step is O(T_past) projection+attention regardless of history.

### RQ6. What implementation decisions produce the largest correctness or performance tradeoffs?

1. **Fused stable kernels** (softmax with max shift; fused cross entropy):
   required for correctness on large-magnitude inputs (tested up to ±10000);
   compositionally-built alternatives were numerically fragile.
2. **Weight tying implemented through an autograd transpose node**: gives
   exact parity with framework semantics and correct dual-path gradients;
   discovered only after the parity harness exposed O(1) logit divergence
   (see Failure Analysis KI-6).
3. **Eager graph freeing** trades debuggability for memory; second backwards
   fail loudly instead of silently mis-computing.
4. **numpy scalars-per-op dispatch** (Python object per op) is the single
   largest performance decision: it buys transparency at a measured 4.5–6.8x
   step-latency cost vs PyTorch eager (see Performance Results).

## Correctness Results

- 165 tests pass, covering every README §15 category including edge cases
  (scalar grads, branched graphs, reused tensors, broadcasting, zero-sized
  dimensions, large softmax inputs, causal mask correctness, gradient
  clipping, checkpoint restoration).
- Finite difference checks: all primitives within 1e-7..1e-5 of numerical
  estimates depending on function curvature (float64, central differences).
- Reference parity (plain-Python textbook impls): norms/softmax/matmul/
  attention/cross-entropy/Adam all match within documented tolerances.
- PyTorch parity table above: ≤6e-7 relative logits, ≤1.1e-6 relative
  gradients.
- KV-cached vs uncached: bit-equal greedy generations; cached teacher-forced
  logits equal uncached within 1e-4 (`tests/test_transformer.py`).

## Performance Results

Training-step latency, TorchZero vs PyTorch eager CPU
(`latency_scaling.json`, figure `latency_comparison.png`):

| batch/seq/dim/layers | TorchZero ms | PyTorch ms | ratio | TZ tok/s |
|---|---|---|---|---|
| 4/32/64/2    | 4.1    | 1.7    | 2.4x | 31.5M* |
| 8/64/128/4   | 45.5   | 14.4   | 3.2x | 11.3M  |
| 16/64/128/4  | 116.8  | 27.7   | 4.2x | 8.8M   |
| 8/128/128/4  | 112.3  | 24.9   | 4.5x | 9.1M   |
| 16/128/256/6 | 1017.4 | 151.6  | 6.7x | 2.0M   |

(*small-batch number inflated by sub-noise denominators; treat >10M tok/s as
upper bound.) The ratio grows with graph size, confirming Python-dispatch
dominance: per-FLOP kernels are competitive, per-node bookkeeping is not.

Checkpoint size: 4.0 bytes/parameter (float32 pickle payload),
`checkpoint_size.json`.

Figures: see `research/figures/` (latency comparison, attention scaling, KV
ablation, forward profile, training curve).

## Ablations

- **KV cache on/off** (`kv_cache_ablation.json`): 1.67x / 2.27x speedup at
  24/48 generated tokens; exact greedy equivalence both cases.
- **Sequence lengths** (same file family): T ∈ {16,32,64,128,256} attention
  fraction 8→20%; step latency superlinear beyond T=128 consistent with T²
  score tensors.
- **Batch sizes**: latency rows cover batch ∈ {4,8,16}; throughput per token
  degrades ~3.5x from B4-small to B16-large configs due to graph-size growth.
- **Model depths**: layers ∈ {2,4,6} across latency rows; step time scales
  slightly superlinearly in depth (graph node count grows linearly, plus
  deeper chains lengthen the topological pass).

## Failure Analysis

Development failures, all root-caused and regression-tested
(`docs/KNOWN_ISSUES.md`):

- **Cross-entropy sign error** (negative losses): `nll = -(shifted) - lse`
  instead of `lse - shifted`. Caught by a hand-computed expected value.
- **Split gradient slicing**: split's backward returned only the slice-shaped
  gradient, failing the parent-shape validation the engine enforces.
- **Reduction grad expansion bug**: manual axis-by-axis `expand_dims` loop
  double-expanded when given axis tuples; replaced with tuple-aware expansion.
- **Cached attention read new values only**: `probs @ v` instead of
  `probs @ v_full`, silently ignoring cached history; caught by requiring
  cached-vs-uncached logit equality.
- **RoPE layout mismatch**: rotation applied to (B,H,T,D) while helper assumed
  sequence axis at dim 1 — produced wrong positions and broadcast crashes.
- **Stale RoPE offset**: cache-length offset computed before cache reset,
  corrupting positions when regenerating after a previous generation.
- **Weight tying never actually implemented** (KI-6): config existed but the
  LM head had independent weights; layer-wise parity passed while full-model
  parity failed at O(1). Fixed by routing logits through E^T as a graph op.
- **Square-matrix transpose masking benchmark bugs** (KI-7): torch (out,in)
  vs TorchZero (in,out) layouts coincide in shape for square weights, hiding
  transposition errors behind "100% gradient error" readings. TorchZero's own
  gradients were first verified correct by finite differences, isolating the
  fault to the comparator.

## Limitations

TorchZero does not implement: GPU/metal execution; mixed precision or bfloat;
distributed training; higher-order derivatives; vectorized/adwantageous
kernel fusion or compilation; reverse-mode coverage of advanced indexing
(beyond gather/scatter-add patterns); padding masks for variable-length
batches (single fixed context length per batch); sparse tensors; quantization;
a production-grade sampler stack (only greedy/temperature/top-k); torch-compatible
serialization. Performance work (batching multiple ops per Python call,
buffer reuse) is deliberately out of scope for transparency.

## Reproducibility

From a clean environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,figures]"
pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional, benchmarks only

pytest                                              # gates 1–29 evidence
torchzero train configs/tiny.yaml                   # tiny training run
torchzero train configs/tiny.yaml --resume checkpoints/tiny/checkpoint.pkl
torchzero generate checkpoints/tiny "The future of computing"
torchzero inspect checkpoints/tiny --html report.html
torchzero benchmark                                 # writes experiments/results/
python scripts/experiment_attention_scaling.py      # RQ3 data
python scripts/experiment_training_curve.py         # training curve data
python research/generate_figures.py                 # rebuild figures
```

All runs are deterministic under fixed seeds (config `seed`, sampler seeds
derived by SHA-256 of seed+epoch).

## Future Work

Based on observed limitations:

1. **Op fusion at the Python boundary** (e.g., fused linear+gelu, batched QKV
   projection): profiling shows MLP projections dominate; fusing cuts graph
   nodes, attacking the dominant cost directly (RQ2/RQ6).
2. **Gradient buffer reuse / arena allocator**: backward allocates every
   intermediate twice; an arena would cut allocator overhead visible in the
   backward>forward wall ratio.
3. **Paged/incremental prefill for long prompts**: KV speedup currently
   plateaus at ~1170 tok/s; chunked prefill would raise the ceiling.
4. **Padding-mask support** to enable variable-length batches, removing the
   packing remainder waste measured in the data pipeline tests.
5. **float64 execution mode** end-to-end to quantify remaining parity error
   sources beyond float32 noise (RQ1 showed ≤1.1e-6; float64 would separate
   accumulation error from algorithmic error).
