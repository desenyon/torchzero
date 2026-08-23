# TorchZero Research Report

**Version 2.0 — experiment-backed analysis of a from-scratch neural network stack**

Every number in this report is quoted from committed JSON files under
`experiments/results/`; figures are regenerated programmatically by
`research/generate_figures.py` into `research/figures/` (10 figures). Commands
to reproduce every artifact are listed in §Reproducibility.

---

## Abstract

TorchZero is a from-scratch neural-network stack — tensor engine, dynamic
reverse-mode autodiff, module system, optimizers, byte-level BPE tokenizer,
data pipeline, decoder-only transformer, training runtime with reproducible
checkpointing, KV-cached generation, and an execution debugger — built without
any automatic differentiation framework. numpy serves strictly as the
primitive kernel backend (elementwise ufuncs, `matmul`, reductions, gather /
scatter); everything above that level is implemented in this repository.

We validate correctness at three increasing strengths: (1) central finite-
difference gradient checks on every differentiable primitive; (2) comparison
against independent plain-Python textbook reference implementations; (3)
full-model weight-transferred parity against PyTorch. Parity results show
forward logits agree to ≤5.9e-7 relative error and every parameter gradient to
≤1.6e-6 across depths 2–8, with error growing approximately linearly in depth
— consistent with float32 accumulation noise and no algorithmic divergence.
The composed autodiff graph retains every intermediate activation, costing up
to ~1.5 GB for an 8-layer batch-8 T=256 forward (~34× the analytic minimum
set of tensors needed for backward), which quantifies the memory price of
per-op graph construction without kernel fusion. KV caching accelerates
autoregressive generation 1.39×→4.05× as generation length grows 8→128 tokens
while reproducing greedy outputs exactly. Against PyTorch eager on CPU,
TorchZero trains identical architectures 2.4–6.7× slower per step, with the
gap widening in graph size — evidence that Python-level per-op dispatch, not
kernel efficiency, dominates the from-scratch overhead.

---

## Motivation

Modern frameworks hide the mechanics that make deep learning work: how
gradients actually flow through a computation, why broadcasting must reduce
gradients correctly, what a KV cache actually saves, where a training step
spends its time, and what retaining a dynamic graph costs in memory. The goal
of TorchZero is not feature parity with mature frameworks but a *transparent*
system: compact enough to read end-to-end, correct enough to trust its
numbers, and instrumented enough to measure its own costs. This report holds
the implementation to that standard: every architectural claim about
correctness or performance below is backed by a stored experiment, and each
development failure we encountered is documented with its root cause.

## System Architecture

Full details in `ARCHITECTURE.md`. Summary:

- **Tensor engine** (`torchzero/tensor/tensor.py`): strided numpy-backed
  storage exposing shape/strides/dtype, gradient storage, requires-flag, and
  parent/backward links forming the dynamic graph.
- **Autodiff** (`torchzero/autograd/engine.py`): iterative topological
  traversal, gradient accumulation, shape/count validation of backward
  outputs, eager graph freeing.
- **Modules** (`torchzero/nn`): automatic parameter registration;
  Linear/Embedding/LayerNorm/RMSNorm/Dropout/GELU.
- **Optimizers** (`torchzero/optim`): SGD(momentum, nesterov, decay), Adam,
  AdamW (decoupled), global-norm clipping, warmup-cosine scheduling.
- **Tokenizer/data**: byte-level BPE trained greedily with deterministic
  tie-breaking; SHA-256(seed, epoch)-shuffled batching over packed rows.
- **Transformer** (`torchzero/transformer/model.py`): decoder-only, pre-norm
  RMSNorm blocks, RoPE causal multi-head attention, GELU MLPs, tied LM head.
- **Runtime/CLI/debugger**: checkpoint/resume with full state; four CLI
  commands; real-execution tracers.

Boundary policy: numpy provides only primitive kernels (documented list in
ARCHITECTURE.md); a static test forbids ML-framework imports under runtime
code; PyTorch appears only inside `benchmarks/` and experiment scripts as a
measurement reference.

## Automatic Differentiation Design

- Dynamic graph built eagerly; each node stores parents plus a backward
  closure mapping outgoing grad → tuple of parent grads.
- Reverse pass uses iterative DFS post-order (no recursion; verified on
  3000-deep chains, `tests/test_autograd.py`). Scalar roots are seeded with
  ones; non-scalar roots require an explicit seed argument whose shape is
  validated.
- Gradients accumulate across branches and reused tensors (`d/dx x⁴` diamond
  test; deep-chain identity tests).
- Broadcasting gradients reduce via `unbroadcast` (sum leading dims, then
  size-1 dims); exhaustively finite-difference checked.
- After a non-retaining backward, every traversed node drops its edges so
  closures release saved activations; a second backward raises loudly rather
  than silently propagating nothing.
- Invalid backwards are detected at runtime: wrong parent-gradient counts and
  mismatched shapes raise immediately (tested).

## Transformer Implementation

Decoder-only, pre-norm:

```
h = E[ids]
repeat n_layers times:
    h = h + Attn(RMSNorm(h))        # RoPE q/k, -1e30 causal mask, MHA
    h = h + MLP_GELU(RMSNorm(h))
logits = RMSNorm(h) @ E^T           # tying realized as a graph transpose node
```

RoPE uses the half-split convention applied to (B,T,H,Dh) q/k before head
transposition; cached decode slices RoPE tables at the absolute cache offset.
Cross entropy is one fused stable kernel (max-shift + log-sum-exp). Sampling
supports greedy, temperature, and top-k with seeded RNG. The token→logits
path is straight-line code inspectable without framework internals.

## Experimental Method

**Environment.** Apple Silicon macOS (arm64), Python 3.14, numpy 2.5.2,
PyTorch 2.13 (CPU wheel, reference only). Timings use `time.perf_counter`
medians over ≥3 runs after ≥2 warmups; training ablations repeat over 3
seeds and report mean ± std.

**Correctness methodology**, in increasing strength:

1. Central finite differences (float64, h=1e-6) against analytical gradients
   for every differentiable primitive, including compositions
   (`tests/test_gradcheck.py`).
2. Independent textbook implementations written in plain Python +
   `math` (`tests/reference_impls.py`) for matmul, softmax, norms, causal
   attention, cross entropy, and Adam updates.
3. Full-model parity vs PyTorch: TorchZero weights transferred into a mirror
   architecture (Linear weights transposed for torch's (out,in) layout),
   same inputs/targets, compare logits, loss, and every parameter gradient.

**Documented tolerances.** float32 runtime vs float64 references:
elementwise/norm outputs atol 2e-4; attention probabilities atol 1e-4;
loss atol 1e-3; optimizer trajectories atol 1e-5/step. Parity errors are
reported raw (never clipped to a threshold): the observed values are
float32-noise scale, so no looser gate was needed.

**Threats to validity.** Single machine, CPU-only; three seeds may under-sample
variance for training curves; `ru_maxrss` is a process-lifetime high-water mark
(used only directionally in `benchmark.json`; graph-memory claims rest on
direct byte counts instead); wall-clock includes Python interpreter effects
that vary between versions; the mirror torch model was written by the same
authors — mitigated by matching against finite differences and independent
plain-Python references, which agree to 1e-7..1e-9.

## Research Questions

### RQ1 — How closely can a compact autodiff engine reproduce reference framework gradients?

**To float32 precision, exactly.**

Full-model logits/loss parity (weight-transferred, same batches;
`parity_vs_torch.json`):

| config (vocab/dim/L) | logits max abs err | logits rel err | loss abs err |
|---|---|---|---|
| 64/32/2  | 1.68e-07 | 2.90e-07 | 4.80e-07 |
| 128/48/3 | 2.58e-07 | 5.88e-07 | 4.59e-07 |
| 256/64/2 | 3.21e-07 | 5.53e-07 | 6.43e-07 |

Per-tensor gradient error distributions across depth
(`grad_error_distribution.json`, figure `grad_error_distribution.png`;
relative error = max|Δg| / max|g_ref| over each tensor):

| depth | tensors compared | median | p95 | max |
|---|---|---|---|---|
| 2 | 26 | 4.93e-07 | 7.01e-07 | 7.65e-07 |
| 4 | 46 | 5.87e-07 | 7.95e-07 | 8.64e-07 |
| 6 | 66 | 4.54e-07 | 8.55e-07 | 1.18e-06 |
| 8 | 86 | 6.51e-07 | 1.22e-06 | 1.60e-06 |

Median error stays flat (~5–7e-7, i.e., tens of float32 ulps) while the max
grows roughly linearly with depth (~1.4e-7 per added layer) — the signature
of accumulated rounding, not algorithmic error. By parameter type (median
across configs): attention weights ≈ norms ≈ embedding within 2× of each
other; no class of parameters shows systematic divergence.

### RQ2 — Which operations dominate runtime in a small from scratch transformer?

Module-level instrumentation of live steps (B8 T64 dim128 L4;
`op_profile.json`, figure `op_profile.png`):

| module | median forward time | share of instrumented time |
|---|---|---|
| MLP (2 Linear + GELU chain) | 13.17 ms | 68% |
| Attention (q/k/v/proj + scores + softmax + ·V) | 4.64 ms | 24% |
| RMSNorm ×2 | 0.69 ms | 4% |

Backward wall time (21.2 ms) exceeds the forward wall (19.8 ms) despite
ideally cheaper backward math: each primitive re-executes elementwise work on
saved activations and allocates fresh gradient arrays in Python. Combined with
the latency scaling below, the dominant cost of the stack is *per-node Python
dispatch*, not FLOPs — the MLP wins on share precisely because it computes the
most useful work per graph node created.

### RQ3 — How does sequence length change the fraction of time spent in attention?

(`attention_scaling.json`, figure `attention_scaling.png`; dim128, 4 layers,
batch 8; full train step incl. optimizer)

| T | step ms | attention ms | fraction |
|---|---|---|---|
| 16 | 18.1 | 1.5 | 8.2% |
| 32 | 26.5 | 2.1 | 8.0% |
| 64 | 49.4 | 5.2 | 10.6% |
| 128 | 117.2 | 15.0 | 12.8% |
| 256 | 277.5 | 54.4 | 19.6% |

Attention time scales quadratically (the B·H·T² score/probability tensors
dominate) while MLP and dispatch costs scale linearly, so the attention
fraction climbs from 8% to 20% over T=16→256. Even at T=256 attention remains
a minority of step time in this stack because per-node overhead inflates the
non-attention baseline — in a fused implementation the crossover would arrive
at shorter lengths.

### RQ4 — What memory costs arise from retaining the dynamic computation graph?

Direct measurement of the live graph after forward, before backward
(`graph_memory.json`, figure `graph_memory.png`; B8 dim96):

| layers \ T | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|
| 2 | 16.6 MB | 34.1 MB | 72.2 MB | 161 MB | 389 MB |
| 4 | 31.7 MB | 65.4 MB | 139 MB | 311 MB | 756 MB |
| 8 | 61.9 MB | 128 MB | 272 MB | 611 MB | 1491 MB |

Three findings:

1. **Node count is depth-proportional and length-independent** (158 / 304 /
   596 nodes for 2/4/8 layers): memory growth in T comes from tensor sizes,
   not graph shape.
2. **Bytes-per-token rise modestly with T** (L4: 1.98 → 2.95 MB/token,
   +49%) because quadratic attention tensors grow faster than the linear
   majority; attention probabilities still account for only 0.4%→5% of
   retained bytes at T=256.
3. **Composition without fusion has a large multiplier.** The analytic set of
   tensors that must be saved for backward in this architecture is ~7·B·T·dim
   floats/layer (normed inputs, attention out, GELU hidden, residuals);
   for L4/T256/B8 that is ≈22 MB, versus 756 MB retained — **~34×** — because
   every intermediate of every unfused op (RoPE splits/muls, masked_fill
   copies, softmax numerator pieces, transposes-as-copies) is itself a saved
   node. This is the precise memory tax that kernel fusion pays down in
   production frameworks, here quantified for the first time in-stack.

### RQ5 — How much does KV caching improve autoregressive generation?

(`kv_cache_scaling.json`, figure `kv_cache_scaling.png`; prompt 16 tokens,
dim96 4-layer model, median of 3 runs; greedy equivalence verified at every
length)

| generated tokens | cached tok/s | uncached tok/s | speedup | greedy match |
|---|---|---|---|---|
| 8 | 1198 | 862 | 1.39× | exact |
| 16 | 1314 | 782 | 1.68× | exact |
| 32 | 1367 | 679 | 2.02× | exact |
| 64 | 1395 | 525 | 2.66× | exact |
| 128 | 1388 | 343 | 4.05× | exact |

Two regimes are visible: uncached throughput decays monotonically (each step
re-runs a full forward over a growing context — O(T²) total), while cached
throughput is flat and even rises slightly (~1190→1388 tok/s) as fixed
per-generation overheads amortize. Speedup therefore grows without sign of
saturation through the measured range, and correctness is preserved exactly
(cached teacher-forced logits match uncached within 1e-4; greedy token
streams are identical at all lengths).

### RQ6 — What implementation decisions produce the largest correctness or performance tradeoffs?

1. **Fused stable kernels** (max-shifted softmax; fused cross entropy):
   required for correctness — compositional softmax was numerically fragile
   on large-magnitude inputs. Quantified in `stability.json`: loss matches a
   float64 reference to 3.7e-7 at unit scale and remains finite/accurate to
   float32 resolution (1.3e-3 absolute on losses >21000, i.e. at machine
   epsilon) even at logit magnitude 10000, while |gradient|max stays constant
   at 1/batch exactly as the mean reduction predicts (figure `stability.png`).
2. **Weight tying as a graph transpose node** gives exact framework semantics
   and correct dual-path gradients; discovered only after the parity harness
   exposed O(1) logit divergence (Failure Analysis). Trade-off: the transpose
   adds a full copy node to the graph (memory, RQ4).
3. **Eager graph freeing** trades reusability for memory; second backwards
   fail loudly. Given RQ4's multiplier, freeing aggressively is worth more
   here than in fused frameworks.
4. **One Python object per op** buys transparency at a measured 2.4–6.7×
   step-latency cost vs PyTorch eager (Performance Results) and the ~34×
   transient activation multiplier (RQ4). Any future optimization effort
   should target node count first.

## Correctness Results

- **165 tests pass**, covering every README §15 category including edge
  cases: scalar gradients, branched graphs, reused tensors, broadcasting,
  zero-sized dimensions, softmax under ±10⁴ inputs, causal mask correctness,
  gradient clipping, checkpoint restoration, forbidden-import static check.
- **Finite differences**: all primitives match numerical estimates within
  1e-7..1e-5 depending on curvature (float64 central differences).
- **Independent references** (plain Python/math): matmul, softmax, LayerNorm/
  RMSNorm rows, single-head causal attention, cross entropy, Adam trajectory
  — all within documented tolerances (≤1e-9 for scalar math paths).
- **Gradient error distributions vs PyTorch**: see RQ1 tables — median
  ~5e-7, worst tensor 1.6e-6 at depth 8.
- **KV-cache determinism**: greedy streams bit-identical at every tested
  generation length; teacher-forced cached logits equal uncached within 1e-4.
- **Numerical stability**: cross entropy finite and float32-accurate to logit
  scale 10⁴ (`stability.json`).
- **End-to-end training**: controlled-task loss decreases (tests), tiny CLI
  model reaches val loss 0.18/perplexity 1.20 on the repeated corpus, and the
  400-step research run reaches val loss 0.39 / train loss 0.008
  (`training_curve.json`, figure `training_curve.png`) — the widening
  train/val gap is genuine memorization of a synthetic repeated corpus, not a
  pipeline bug (validation data never enters training rows).

## Performance Results

Training-step latency, identical weights/architecture, TorchZero vs PyTorch
eager CPU (`latency_scaling.json`, figure `latency_comparison.png`):

| batch/seq/dim/layers | TorchZero ms | PyTorch ms | ratio | TZ tokens/s |
|---|---|---|---|---|
| 4/32/64/2    | 4.1    | 1.7   | 2.4× | 31.5M* |
| 8/64/128/4   | 45.5   | 14.4  | 3.2× | 11.3M |
| 16/64/128/4  | 116.8  | 27.7  | 4.2× | 8.8M |
| 8/128/128/4  | 112.3  | 24.9  | 4.5× | 9.1M |
| 16/128/256/6 | 1017.4 | 151.6 | 6.7× | 2.0M |

*Small-config tok/s is inflated by sub-millisecond denominators; treat as an
upper bound. The monotone ratio growth with graph size identifies Python
dispatch as the bottleneck: kernels themselves are numpy-grade, bookkeeping
is not.

Checkpoint size: 4.0 bytes/parameter (float32 pickle payload),
`checkpoint_size.json`.

Figure manifest (all regenerated from raw JSON): `latency_comparison`,
`attention_scaling`, `kv_cache_ablation`, `kv_cache_scaling`, `op_profile`,
`training_curve`, `graph_memory`, `optimizer_ablation`,
`grad_error_distribution`, `stability`.

## Ablations

**Optimizers** (`optimizer_ablation.json`, figure `optimizer_ablation.png`;
controlled next-token task, 250 steps, mean±std of final-20-step loss over 3
seeds):

| optimizer | final loss (mean ± std) |
|---|---|
| SGD (momentum 0.9) | 0.0715 ± 0.0024 |
| Adam | 0.0522 ± 0.0033 |
| AdamW | 0.0533 ± 0.0034 |

Adam-family converges ~35% lower than tuned-momentum SGD on this task;
AdamW's decoupled decay costs nothing measurable at this horizon.

**LR schedule** (same file): AdamW with constant LR reaches
0.0172 ± 0.0011 versus 0.0533 ± 0.0034 with warmup-cosine at 250 steps. On a
short, clean memorization task the cosine decay retires learning rate before
the run finishes; the schedule's value should be sought at longer horizons
and noisier objectives. We report the unfavorable outcome deliberately.

**Sequence lengths**: T ∈ {16..256} — attention share 8→20% (RQ3); step
latency superlinear beyond T≈128, matching the T² score tensors.

**Batch sizes**: latency rows cover batch ∈ {4,8,16}; throughput per token
falls ~3.5× from the smallest to largest config as graph size grows.

**Model depths**: layers ∈ {2,4,6} (latency) and {2,4,6,8} (gradient-error
distribution); step time scales slightly superlinearly in depth; gradient
parity error max grows linearly in depth (RQ1).

**KV cache on/off and length scaling**: RQ5 table; speedup 1.39×→4.05×.

## Failure Analysis

All failures found during development were root-caused, fixed at the source,
and pinned with regression tests (`docs/KNOWN_ISSUES.md`):

| ID | Failure | Root cause | Detection |
|----|---------|------------|-----------|
| KI-1 | Negative cross-entropy loss | sign error: `-(shifted) − lse` vs `lse − shifted` | hand-computed expected value |
| KI-2 | split() backward shape crash | returned slice-shaped grad instead of zero-filled parent-shaped grad | engine's own shape validation |
| KI-3 | KV-cached attention ignored history | `probs @ v` instead of `probs @ v_full` | cached-vs-uncached equality requirement |
| KI-4 | Wrong RoPE positions | rotation applied to (B,H,T,D); helper assumed seq axis at dim 1 | broadcast crashes + causality test |
| KI-5 | Stale RoPE offset on regenerate-after-generate | offset computed before cache reset | cached-vs-uncached equality |
| KI-6 | Weight tying configured but never implemented; LM head had independent random weights → O(1) full-model divergence while layer parity passed | tying existed only as a config field | PyTorch full-model parity harness |
| KI-7 | Benchmark showed fake "~100% gradient error" | torch (out,in) vs TorchZero (in,out) layouts; square matrices made shape-based detection silently pass | TorchZero grads first verified correct by finite differences, isolating the comparator |
| KI-8 | Clean-env audit: `torchzero benchmark` crashed | benchmarks package not installed | README §21 audit in a fresh venv |

KI-6 and KI-7 are the most instructive: layer-by-layer checks can all pass
while the composed system is wrong, and a broken measuring stick can indict
correct code — finite differencing is the arbiter that separates the two.

## Limitations

TorchZero does not implement: GPU/accelerator execution; mixed precision or
bfloat16; distributed training; higher-order derivatives; kernel fusion or
compilation; vectorized multi-op dispatch; padding masks for variable-length
batches (single context length per batch); sparse/quantized tensors;
production sampler stacks (beam search, penalties); framework-compatible
serialization. Memory results (RQ4) quantify why fusion matters here but the
fusion itself is future work. Training-scale claims rest on models ≤ a few M
parameters on CPU.

## Reproducibility

From a clean environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,figures]"
pip install torch --index-url https://download.pytorch.org/whl/cpu  # optional; benchmarks/reference only

pytest                                              # 165 tests, gates 1–29
torchzero train configs/tiny.yaml                   # tiny training run
torchzero train configs/tiny.yaml --resume checkpoints/tiny/checkpoint.pkl
torchzero generate checkpoints/tiny "The future of computing"
torchzero inspect checkpoints/tiny --html report.html
```

Regenerate every artifact behind this report:

```bash
torchzero benchmark                                  # benchmark.json, parity_vs_torch, latency_scaling, kv_cache_ablation, op_profile, checkpoint_size
python scripts/experiment_attention_scaling.py      # attention_scaling.json
python scripts/experiment_training_curve.py         # training_curve.json
python scripts/experiment_graph_memory.py           # graph_memory.json          (RQ4)
python scripts/experiment_kv_scaling.py             # kv_cache_scaling.json       (RQ5)
python scripts/experiment_optimizer_ablation.py     # optimizer_ablation.json     (Ablations)
python scripts/experiment_grad_error_distribution.py# grad_error_distribution.json (RQ1)
python scripts/experiment_stability.py              # stability.json              (RQ6)
python research/generate_figures.py                 # rebuild research/figures/*.png
```

Determinism: fixed seeds everywhere; batch order derived from
SHA-256(seed, epoch); sampling seeds explicit. Total regeneration time is
minutes on a laptop CPU.

## Future Work

Grounded in the measured limitations above:

1. **Op fusion at the Python boundary** (fused Linear+GELU, batched QKV,
   fused RoPE): profiling shows MLP modules dominate forward time (RQ2) while
   unfused composition multiplies retained activations ~34× (RQ4) — fusion
   attacks both the speed and memory bottlenecks simultaneously.
2. **Gradient buffer reuse / arena allocation**: backward exceeds forward
   wall time partly from per-node gradient array allocation (Performance
   Results).
3. **Chunked prefill for long prompts**: cached generation throughput
   plateaus ~1400 tok/s; prefill chunking would raise the ceiling and widen
   the measured 4× advantage further.
4. **Padding masks** for variable-length batches, eliminating packing waste
   currently dropped by the packer.
5. **float64 execution mode**: with parity already at float32 noise (RQ1), a
   float64 toggle would separate accumulation error from algorithmic error
   empirically and serve as a reference mode for users.
6. **Longer-horizon scheduler study**: the constant-LR result (Ablations)
   motivates measuring whether warmup-cosine wins at 10–50× longer horizons
   before claiming it as a default.
