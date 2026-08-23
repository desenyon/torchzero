# TorchZero

> A neural network and transformer stack built **without PyTorch, TensorFlow, JAX, or any automatic differentiation framework**.

TorchZero implements the machinery normally hidden by ML frameworks: tensors,
broadcasting, reverse-mode autodiff, optimizers, neural-network modules,
attention, a decoder-only transformer, training with checkpointing and resume,
autoregressive generation with KV caching, an execution debugger, and a
byte-level BPE tokenizer.

## Install

Requires Python >= 3.10.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # runtime + test dependencies (numpy, pyyaml, pytest)
pip install -e ".[figures]"      # optional: matplotlib for research figures
```

The **only** external numerical dependency is `numpy`, used strictly as a
low-level array storage / primitive kernel backend. The tensor abstraction,
computation graph, autodiff, module system, optimizers, transformer layers,
training loop, checkpointing, generation, tokenizer, and debugger are all
implemented in this repository. See `ARCHITECTURE.md` ("External kernel
boundary") for the exact delegation list.

## Quickstart

Train the tiny deterministic model (seconds on CPU):

```bash
torchzero train configs/tiny.yaml
```

Generate text:

```bash
torchzero generate checkpoints/tiny "The future of computing"
```

Open the execution debugger (text UI when interactive; add `--html out.html`
for a full HTML report):

```bash
torchzero inspect checkpoints/tiny "the sun rises in the east"
torchzero inspect checkpoints/tiny --html debugger_report.html
```

Run the performance + correctness benchmark suite:

```bash
torchzero benchmark            # writes experiments/results/*.json
```

Resume interrupted training:

```bash
torchzero train configs/tiny.yaml --resume checkpoints/tiny/checkpoint.pkl
```

## Configuration

`configs/tiny.yaml` — deterministic, runs in seconds; used by tests.
`configs/research.yaml` — larger configuration for meaningful experiments.

Point `data.text` at any UTF-8 `.txt` file to train on your own corpus; the
byte-level BPE tokenizer is trained from scratch as part of the run and saved
next to the checkpoints (`tokenizer.json`). Large datasets must not be
committed; fetch them yourself and reference them by path.

## Tests

```bash
pytest                          # full suite (~165 tests)
pytest tests/test_gradcheck.py  # finite-difference gradient checks
pytest tests/test_reference_parity.py
```

The suite covers: tensor arithmetic, broadcasting, matmul gradients, shape ops,
autograd semantics (branching, reuse, accumulation, graph cleanup), finite
difference gradient checks for every differentiable primitive, parameter
registration, serialization, optimizer steps vs hand-computed references,
normalization, causal attention, transformer blocks, cross entropy, training
steps, checkpoint resume equivalence, KV-cache vs uncached parity, generation,
CLI end-to-end, and debugger data endpoints.

Static check: no ML-framework import exists in runtime code
(`tests/test_integration.py::TestHardConstraint`).

## Benchmarks & research

```bash
python benchmarks/run.py                        # full comparison vs PyTorch (if installed)
python benchmarks/run.py --quick                # smaller configs
python scripts/experiment_attention_scaling.py  # attention time fraction vs T
python scripts/experiment_training_curve.py     # real training run -> loss curve
python research/generate_figures.py             # rebuild research/figures/*.png
```

PyTorch is used **only** inside `benchmarks/` as an external measurement
reference; it never powers TorchZero runtime execution. Without torch, only
TorchZero-side measurements run. Raw results live in `experiments/results/`;
see `RESEARCH_REPORT.md` for analysis.

## Repository layout

```
torchzero/        tensor engine, autograd, nn modules, optimizers,
                  tokenizer, data pipeline, transformer, runtime, CLI
debugger/         forward/backward tracers + text/HTML reports
tests/            pytest suite incl. gradient checks & reference impls
benchmarks/       PyTorch-comparison benchmark harness
experiments/      raw result JSONs (committed)
configs/          tiny.yaml (fast) and research.yaml (bigger)
scripts/          experiment drivers and data utilities
research/         figure generation + figures
docs/             PLAN / STATUS / DECISIONS / KNOWN_ISSUES
```

## License

MIT. See `LICENSE`.
