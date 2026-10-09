# TorchZero

> A neural network and transformer stack built **without PyTorch, TensorFlow,
> JAX, or any automatic differentiation framework**.

TorchZero makes the machinery normally hidden by ML frameworks readable:
NumPy-backed tensors, reverse-mode autodiff, modules, optimizers, a byte-level
BPE tokenizer, a decoder-only transformer, training and resume, KV-cached
inference, and a debugger that traces real executions. It is a CPU learning
and experimentation project, not a production LLM training system.

NumPy is the only external numerical dependency. It supplies array storage,
elementwise kernels, reductions, matrix multiplication and random-number
generators. The computation graph, derivatives, neural-network components,
training state and tokenizer are implemented here. PyYAML reads configuration.
See [ARCHITECTURE.md](ARCHITECTURE.md) for the boundary and implementation map.

## Install

Python **3.10 or later** is required. From a source checkout:

```bash
git clone https://github.com/desenyon/torchzero.git
cd torchzero
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

Use `python -m pip install -e .` for runtime dependencies only. The `dev`
extra adds pytest, Ruff and the distribution build frontend. Optional figures
require `python -m pip install -e ".[figures]"`. A virtual environment keeps
TorchZero separate from similarly named packages. No accelerator or model
weights download is required.

## Train, pause, resume, generate

The tiny configuration uses a bundled repeated-text corpus and a small model:

```bash
# Pause after 20 updates while keeping the planned 60-step LR schedule.
torchzero train configs/tiny.yaml --stop-after 20

# Continue the same plan through step 60.
torchzero train configs/tiny.yaml --resume checkpoints/tiny/final.pkl

# Greedy generation; omit --temperature 0 to sample.
torchzero generate checkpoints/tiny "the sun" --tokens 12 --temperature 0

# Seeded sampling; --no-cache runs the full growing prefix at each step.
torchzero generate checkpoints/tiny "the sun" --tokens 12 --seed 42 --top-k 20

# Inspect real forward/backward activations and gradients.
torchzero inspect checkpoints/tiny "the sun" --html debugger_report.html
```

For an uninterrupted run, omit `--stop-after`. Its value is an **absolute step**,
not an additional number of steps. `max_steps` stays fixed when resuming;
changing it would change the warmup/cosine schedule used by earlier updates.
Do not use `max_steps` to simulate an interruption.

Each run writes:

| File | Contents |
| --- | --- |
| `tokenizer.json` | Byte vocabulary plus learned merge rules |
| `checkpoint.pkl` | Complete state at the latest periodic checkpoint or requested stop |
| `final.pkl` | Complete state at the last completed step, including a paused run |
| `history.json` | Loss, LR, gradient norm, timings, token rate and periodic validation loss |

`final.pkl` means the end of this invocation; inspect its `step` to determine
whether the full plan completed. Both checkpoint files support exact resume.
A checkpoint directory passed to `generate` or `inspect` prefers `final.pkl`,
then `checkpoint.pkl`; an explicit file path is also accepted. Keep the matching
`tokenizer.json` beside the checkpoint. Text checkpoints verify tokenizer
identity before inference. A model-only checkpoint without tokenizer metadata
can accept space-separated integer token IDs instead.

The text debugger is interactive when stdin is a terminal (`stages`, `attn`,
`grads`, `graph`, `params`, `quit`). `--html` produces a self-contained report.
Reports contain model activations and prompt-derived data; treat them as run
artifacts when working with private corpora.

## Architecture and data flow

```text
UTF-8 file / inline text / built-in corpus
  -> byte-level BPE -> contiguous train/validation token split
  -> T+1-token windows -> deterministic epoch/cursor batch sampler
  -> embedding -> pre-norm transformer blocks -> tied or independent LM head
  -> stable cross entropy -> reverse-mode autodiff -> gradient clipping
  -> scheduled SGD / Adam / AdamW update -> atomic training-state checkpoint
```

A transformer block computes RMSNorm, causal multi-head attention with RoPE,
a residual connection, then RMSNorm and a GELU MLP with another residual.
Configured dropout applies to attention probabilities and both residual
branches during training. Every dropout module owns a persistent RNG stream;
evaluation does not consume it. `dim / n_heads` must be an even integer for
half-split rotary embeddings.

The default output projection shares the embedding parameter through a
transpose operation, so gradients from both uses accumulate into one weight.
`tie_weights: false` creates an independent output matrix.

Autodiff builds an eager graph of parent tensors and backward closures. The
engine uses iterative traversal, accumulates branched gradients, validates
returned gradient shapes and releases the graph after backward unless
`retain_graph=True`. Finite-difference tests cover both operands of broadcasted
powers and matmul, duplicate gathers, scatter-add, reductions and shape ops.
`no_grad()` prevents graph creation during inference.

| Component | Source |
| --- | --- |
| Tensor operations and broadcasting gradients | `torchzero/tensor/tensor.py` |
| Backward traversal and grad mode | `torchzero/autograd/engine.py` |
| Parameter discovery, module modes and RNG snapshots | `torchzero/nn/module.py` |
| Linear, Embedding, norms, dropout, activations | `torchzero/nn/layers.py` |
| Optimizers, clipping and LR schedules | `torchzero/optim/optim.py` |
| Byte-level BPE | `torchzero/tokenizer/bpe.py` |
| Packing, next-token pairs and sampler state | `torchzero/data/dataset.py` |
| Transformer, RoPE, KV cache and generation | `torchzero/transformer/model.py` |
| Training and weighted evaluation | `torchzero/runtime/trainer.py` |
| Versioned checkpoint I/O and restoration | `torchzero/runtime/checkpoint.py` |
| Commands | `torchzero/cli/main.py` |
| Execution traces and reports | `debugger/` |

## Configuration

Copy [configs/tiny.yaml](configs/tiny.yaml) or use
[configs/research.yaml](configs/research.yaml) as a larger CPU example. For your
own corpus, save the following as `configs/corpus.yaml` and place UTF-8 text in
`data/raw/corpus.txt`:

```yaml
out_dir: checkpoints/corpus
model:
  vocab_size: auto
  dim: 48
  n_layers: 2
  n_heads: 4
  block_size: 64
  dropout: 0.1
  tie_weights: true
data:
  text: ../data/raw/corpus.txt
  val_fraction: 0.1
train:
  optimizer: adamw
  lr: 3.0e-3
  weight_decay: 0.01
  grad_clip: 1.0
  batch_size: 8
  max_steps: 200
  warmup_steps: 20
  min_lr_ratio: 0.1
  eval_every: 20
  ckpt_every: 50
  log_every: 10
  seed: 0
```

Run `torchzero train configs/corpus.yaml`. Corpus and `tokenizer_path` paths are
resolved **relative to the YAML file**. `out_dir`, `--resume` and report paths
are relative to the current working directory. Large datasets and checkpoints
are ignored by Git and should remain outside source control.

| Setting | Meaning / default |
| --- | --- |
| `data.text` | UTF-8 file path; null/omitted uses the built-in corpus |
| `data.inline_text` | Explicit literal corpus text; mutually exclusive with `data.text` |
| `data.tokenizer_path` | Existing tokenizer JSON; otherwise train BPE for a new run |
| `data.val_fraction` | Contiguous held-out fraction, strictly between 0 and 1; default `0.1` |
| `model.vocab_size` | BPE vocabulary ceiling; `auto`/null uses 288; explicit values must be at least 256 |
| `model.dim`, `n_layers`, `n_heads` | Positive model dimensions; defaults 128, 4, 4 |
| `model.block_size` | Maximum input and total generated context; default 64 |
| `model.ffn_hidden` | Positive hidden dimension; defaults to `4 * dim` |
| `model.dropout` | Probability in `[0, 1)`; default `0.0` |
| `model.norm_eps` | Positive finite normalization epsilon; default `1e-6` |
| `model.tie_weights` | Share embedding/output parameters; default true |
| `train.optimizer` | Required: `sgd`, `adam` or `adamw`; trainer SGD uses momentum 0.9 |
| `train.lr`, `batch_size`, `max_steps` | Required positive LR, batch size and planned update count |
| `train.weight_decay` | Default `0.01`; coupled in SGD/Adam, decoupled in AdamW |
| `train.warmup_steps`, `min_lr_ratio` | Linear warmup then cosine decay; defaults 20 and 0.1 |
| `train.grad_clip` | Positive global gradient-norm limit; default 1.0 |
| `train.eval_every`, `ckpt_every`, `log_every` | Positive update intervals; defaults 50, 100, 10 |
| `train.seed` | Model/dropout initialization and batch-order seed; default 0 |

The model uses the tokenizer's **actual** vocabulary size, which can be below
its training ceiling if the text cannot supply more merges. This prevents
sampling undecodable output IDs. Resume always loads the tokenizer beside the
checkpoint; it never retrains or substitutes the configured external tokenizer.

Each data split must contain at least `block_size + 1` encoded tokens. Too-small
splits fail with an actionable error instead of borrowing validation examples
from training. BPE is learned on the whole supplied corpus before the token
split; rigorous held-out-tokenizer experiments should supply a tokenizer fitted
on independent training text via `tokenizer_path`.

## Python API

```python
import numpy as np
from torchzero import Tensor, no_grad
from torchzero.transformer import Transformer, TransformerConfig

x = Tensor(np.array([[1., 2.], [3., 4.]]), requires_grad=True)
y = (x @ x.T).mean()
y.backward()
print(x.grad)

model = Transformer(TransformerConfig(
    vocab_size=256, dim=32, n_layers=2, n_heads=4, block_size=16,
), seed=7)
with no_grad():
    logits, _ = model(np.array([[1, 2, 3]], dtype=np.int64))
ids = model.generate([1, 2, 3], max_new_tokens=5, temperature=0., seed=7)
```

Build real next-token batches with the language-model packer:

```python
from torchzero.data import pack_lm_sequences, get_batch, BatchSampler

rows = pack_lm_sequences(list(range(11)), context_length=4)
# rows: [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]]
x, y = get_batch(rows, [0])
# x: [[0, 1, 2, 3]], y: [[1, 2, 3, 4]]
sampler = BatchSampler(len(rows), batch_size=1, seed=7)
indices = sampler.next_batch()       # consumes its persistent cursor
state = sampler.state_dict()         # epoch, cursor, seed, sizes
```

The next row shares the lookahead token, so every retained input has a real
successor and no row wraps from its last token back to its first. An incomplete
tail is dropped. Split before packing. `pack_sequences` still provides generic
non-overlapping rows of the requested width; passing them to `get_batch`
produces inputs/targets one token shorter. `iter(sampler)` is a non-consuming
view of the current epoch; `next_batch()` is the consuming training API and
includes the final partial batch before advancing the epoch.

`Trainer(model, train_rows, val_rows, cfg, out_dir)` accepts integer rows of
2 through `block_size + 1` tokens. Use `fit(stop_after=...)`, `fit(resume_from=...)`
and `save(path)` for complete state. `evaluate` computes token-weighted loss,
including partial batches, and restores every module's previous training mode.
For optimizer-only use, `state_dict()`/`load_state_dict()` preserve buffers,
hyperparameters and step count; state is tied to parameter order and shapes.

`gather(indices, axis)` accumulates repeated indices in backward.
`scatter_add(indices, src, axis)` is out of place and differentiable in the
base and source. Full-rank indices must match the source shape and destination
non-scatter dimensions. One-dimensional indices at axis 0 also support
whole-row scatter. Tensor-exponent differentiation requires positive bases
for real-valued exponent gradients; NumPy domain errors/NaNs are not masked.

## Checkpoint and reproducibility contract

Version-2 checkpoints contain model weights/config, optimizer type and
hyperparameters/buffers, LR scheduler state, completed update count, metric
history, sampler epoch/cursor, actual dropout RNG states and a fingerprint of
both packed data splits. CLI checkpoints also record corpus and tokenizer
identity. Periodic and final files use the same state schema.

Writes use a temporary file in the destination directory, flush/fsync it, then
atomically replace the destination. A failed serialization leaves the previous
checkpoint intact. This is protection against partial checkpoint writes, not a
claim of cross-platform power-loss durability.

Exact resume requires the same training plan, encoded data, model, numerical
runtime and parameter order. Logging/evaluation/checkpoint intervals may change;
training settings, including `max_steps`, must match. Losses, model arrays,
gradient norms, LR, sampler position and RNG state are compared in regression
tests after real interruptions for SGD, Adam and AdamW with dropout enabled.
Wall-clock metrics are intentionally not reproducible. Different NumPy/BLAS
versions or hardware can change floating-point results.

**Load only trusted checkpoints.** `.pkl` files use Python pickle, which can
execute code when loaded. Version/schema validation happens after unpickling
and is not a security boundary. The same warning applies to `Module.load()`.
Do not load checkpoints from untrusted users, URLs or uploads.

## Generation behavior

Generation runs in `no_grad()`, temporarily enters evaluation mode, restores
all prior module modes even on failure, and clears its temporary KV cache.
Seeded sampling consumes one RNG stream across all generated tokens. It does
not advance training dropout streams. `temperature=0` or `top_k=1` is greedy;
negative/non-finite temperatures and non-positive top-k values are rejected.
Tied logits in top-k retain the lowest token IDs, with exactly k candidates.

The returned list includes the prompt. Generation stops at `block_size`, so it
may return fewer tokens than requested; a full-context prompt or zero-token
budget returns the prompt unchanged. Empty prompts and out-of-vocabulary IDs
are rejected by the Python API (the CLI supplies token 0 for an empty prompt).
There is no EOS policy or sliding-window extension. Cached and uncached logits
are compared within numerical tolerance, and tests compare complete generated
sequences rather than only their prompt prefix.

## Migrating from 0.1

- `data.text` now does what the old README advertised: it reads a file. Move
  literal text configurations to `data.inline_text`.
- `get_batch` and trainer rows use real adjacent pairs. Use
  `pack_lm_sequences(ids, T)` for T-token inputs, or pass existing T-wide rows
  and receive T-1-token inputs. Training loss curves change because circular
  row-end targets were incorrect.
- Samplers now consume all rows per epoch. Do not call `advance_epoch()` after
  each training step. Warmup applies before the first optimizer update.
- Nonzero transformer dropout now takes effect; repeated training calls use
  advancing RNG streams. Old configs silently ignored this value.
- Version-1 checkpoints remain readable for weights/inference with the same
  architecture and tokenizer. **Exact training resume is rejected** because
  version 1 did not preserve the necessary state and trained different targets.
  To reuse weights, build the matching model, load `payload['model_state']`,
  and start a fresh optimizer/training plan; do not relabel an old checkpoint.
- Seeded sampled text can change due to the repaired RNG stream and top-k ties.
  Generation preserves modes, clears caches and respects total context capacity.
- Numerical fixes intentionally change gradients for powers, min, vector/batch
  matmul, negative-axis transpose, repeated gather and scatter-add. SGD momentum
  is restored correctly; AdamW skips decay for parameters without gradients.

## Verification and development

```bash
python -m pytest                         # full suite, including real CLI subprocesses
python -m pytest tests/test_numerical_regressions.py
python -m pytest tests/test_runtime_reliability.py
python -m ruff check .                   # configured syntax/undefined-name correctness rules
python -m compileall -q torchzero debugger benchmarks
python -m build                         # wheel + source distribution
```

There is no configured type-checker gate; annotations are currently partial.
Ruff intentionally starts with correctness rules (`E9`, `F63`, `F7`, `F82`), not
a repository-wide style reformat. CI runs tests from both checkout and built
source archive, these checks, the build and an installed-wheel
train/resume/generate/inspect smoke test on Python 3.10 and 3.13.
The framework-import guard prevents ML-framework imports in runtime modules.

For considerate CPU use on a shared machine, set `OPENBLAS_NUM_THREADS=1`,
`OMP_NUM_THREADS=1` and, on macOS, `VECLIB_MAXIMUM_THREADS=1` before tests/training.
See [docs/RELIABILITY_UPGRADE.md](docs/RELIABILITY_UPGRADE.md) for this upgrade's
scope and verification design, and [CHANGELOG.md](CHANGELOG.md) for changes.

## Benchmarks and research

```bash
torchzero benchmark --quick
python benchmarks/run.py --quick
python scripts/experiment_attention_scaling.py
python scripts/experiment_training_curve.py
python research/generate_figures.py
```

Benchmark and experiment code may use PyTorch only as an optional measurement
reference. It never supplies TorchZero runtime execution. Without it, reference
comparisons are skipped. These commands can overwrite generated results under
`experiments/results/`; run deliberately. Existing results and
[RESEARCH_REPORT.md](RESEARCH_REPORT.md) describe their recorded runs, **not a
new performance claim for version 0.2**. Training/data/RNG changes mean prior
curves should be regenerated before drawing conclusions about this version.

## Limitations

- CPU/NumPy only; no CUDA, distributed training, mixed precision or accelerator
  backend. Autodiff is first order; there is no higher-order derivative engine.
- Corpus loading, BPE training, token arrays and data fingerprints are in memory;
  this is not a streaming loader. Built-in text is repetitive demonstration
  data and is not evidence of language-model quality.
- Grad-mode state is process-global, and model RNG/cache/trainer state is mutable;
  shared instances are not designed for concurrent training/inference threads.
- No padding/attention-mask batching, gradient accumulation, beam search, EOS
  stopping or contexts beyond the configured block size.
- Optimizer checkpoints match parameters by order/shape, not stable names.
  Custom model reordering requires a new optimizer. Adam's `amsgrad` argument
  is rejected when true; use standard Adam or AdamW.
- Pickle is for trusted local artifacts. There is no safe remote checkpoint
  loader or external model-weight format importer.

## License

MIT. See [LICENSE](LICENSE).
