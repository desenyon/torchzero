# TorchZero

> **A neural network and transformer stack built without PyTorch, TensorFlow, JAX, or another automatic differentiation framework.**

TorchZero implements the machinery normally hidden by modern machine learning frameworks: tensors, broadcasting, automatic differentiation, optimizers, neural network modules, attention, transformer training, checkpointing, generation, and an interactive execution debugger.

The goal is not to recreate every feature of a mature ML framework.

The goal is to build a compact system whose core mechanics are transparent, correct, measurable, and sufficient to train and run a small decoder only transformer from first principles.

An autonomous coding agent operating from this README must continue until every completion gate passes, every created file is committed, the research report is generated from real experiments, and the completed repository is pushed.

---

## 1. Hard Constraint

Core model execution and training may not use:

```text
PyTorch autograd
TensorFlow
JAX
MLX automatic differentiation
another neural network framework
```

A numerical array library may be used as a low level storage and primitive computation backend if necessary, but TorchZero must implement its own:

```text
tensor abstraction
computation graph
reverse mode automatic differentiation
gradient accumulation
module system
optimizer logic
transformer layers
training loop
checkpointing
generation
```

Document every external numerical primitive used.

The project must make the boundary between TorchZero code and external numerical kernels explicit.

---

## 2. Required User Experience

A user must be able to run:

```bash
torchzero train configs/tiny.yaml
```

and train a small language model.

Then:

```bash
torchzero generate checkpoints/tiny "The future of computing"
```

and generate text.

Then:

```bash
torchzero inspect checkpoints/tiny
```

and open an interactive debugger or launch a browser interface.

Then:

```bash
torchzero benchmark
```

to run the performance and correctness benchmark suite.

The repository must include a tiny deterministic configuration that can run quickly for testing and a larger research configuration for meaningful experiments.

---

## 3. Tensor Engine

Implement a `Tensor` abstraction.

It must support:

1. Data storage
2. Shape
3. Strides or documented memory model
4. Dtype
5. Gradient storage
6. Gradient requirement flag
7. Parent operations
8. Backward function
9. Graph cleanup
10. Detach semantics

Required operations include at minimum:

```text
addition
subtraction
multiplication
division
power
negation
sum
mean
max when needed
matrix multiplication
transpose
reshape
view or equivalent
broadcasting
exp
log
sqrt
tanh or equivalent
softmax primitives
indexing
embedding gather
concatenation when needed
```

Broadcasting gradients must reduce correctly to original operand shapes.

---

## 4. Automatic Differentiation

Implement reverse mode autodiff.

Requirements:

1. Build a dynamic computation graph.
2. Topologically order dependencies.
3. Propagate gradients backward.
4. Accumulate gradients from multiple paths.
5. Handle scalar and tensor outputs.
6. Support explicit gradient arguments for non scalar roots.
7. Avoid retaining graphs unnecessarily.
8. Detect invalid backward operations where practical.

Build finite difference gradient checks.

For every differentiable primitive, compare analytical gradients with numerical estimates over deterministic test cases.

Gradient checking must be part of automated tests.

---

## 5. Neural Network Module System

Implement a small module API.

Support:

```text
Module
Parameter
ModuleList or equivalent
state_dict
load_state_dict
train mode
eval mode
zero_grad
parameter traversal
```

Required modules:

```text
Linear
Embedding
LayerNorm and or RMSNorm
Dropout if used
Sequential when useful
```

Parameter registration must be automatic and testable.

---

## 6. Optimizers

Implement:

1. SGD
2. Adam
3. AdamW

Document equations.

Tests must compare several deterministic optimizer steps against a trusted reference calculation.

Implement:

```text
weight decay
learning rate
beta values
epsilon
gradient clipping
```

where appropriate.

---

## 7. Transformer

Implement a decoder only transformer.

Required components:

1. Token embeddings
2. Positional encoding or RoPE
3. Causal self attention
4. Multi head attention
5. Normalization
6. Feed forward network
7. Residual connections
8. Transformer blocks
9. Final normalization
10. Language modeling head
11. Cross entropy loss
12. Autoregressive generation

The model must support KV caching for inference.

The implementation must be readable enough that the execution path from tokens to logits can be inspected without navigating framework internals.

---

## 8. Tokenization

Use one of two approaches.

### Option A

Implement a compact tokenizer such as byte pair encoding.

### Option B

Use an external tokenizer library but keep tokenizer logic isolated from the neural network runtime.

If an external tokenizer is used, the research report must explicitly state that tokenization is outside TorchZero's from scratch ML engine.

The preferred implementation includes a small byte level BPE tokenizer because it creates a complete training pipeline.

---

## 9. Data Pipeline

Implement:

1. Dataset loading
2. Tokenization
3. Train and validation split
4. Sequence packing
5. Batch creation
6. Deterministic shuffling with seed
7. Configurable context length

The tiny validation dataset must be included or reproducibly downloadable.

Large datasets must not be committed.

Provide scripts to obtain and preprocess larger experiment data.

---

## 10. Training Runtime

Training must include:

```text
forward pass
loss calculation
backward pass
gradient clipping
optimizer step
learning rate schedule
evaluation
checkpointing
metric logging
resume training
```

Checkpoint state must include enough information to continue training reproducibly:

```text
model parameters
optimizer state
training step
configuration
random state where practical
metadata
```

Interrupting and resuming training must be tested.

---

## 11. Execution Debugger

This is a major portfolio component.

Build an interface that exposes the internals of a transformer execution.

A user should be able to inspect:

```text
input tokens
embeddings
each transformer block
attention matrices
Q, K, V shapes
residual stream
MLP activations
normalization statistics
logits
loss
gradient norms
parameter updates
```

Support at least these two modes.

### Forward Trace

Trace one batch through the network.

### Backward Trace

Show the backward graph and gradient flow.

The debugger should identify:

```text
operation
input shapes
output shapes
requires gradient
gradient norm
execution time
memory estimate
```

Metrics must come from real execution.

Do not fabricate traces on the front end.

---

## 12. Numerical Validation

Where possible, implement a reference comparison harness using an established framework only for testing.

The reference framework may not power TorchZero runtime execution.

Compare:

1. Primitive forward results
2. Primitive gradients
3. Linear layer output
4. Normalization
5. Attention
6. Transformer block
7. Loss
8. Optimizer steps
9. Full model logits

Use deterministic parameters and inputs.

Explicitly document tolerances.

Investigate rather than conceal divergence.

---

## 13. Performance Instrumentation

Measure:

```text
forward latency
backward latency
optimizer latency
tokens per second
training steps per second
peak memory when measurable
operation counts
parameter count
activation sizes
```

Add operation level tracing where practical.

Performance is secondary to correctness, but the research report must analyze where the from scratch stack spends time.

---

## 14. Repository Structure

Target structure:

```text
torchzero/
    tensor/
    autograd/
    nn/
    optim/
    tokenizer/
    data/
    transformer/
    runtime/
    cli/
debugger/
tests/
benchmarks/
experiments/
configs/
scripts/
docs/
research/
README.md
ARCHITECTURE.md
RESEARCH_REPORT.md
CHANGELOG.md
LICENSE
```

Document any meaningful deviation.

---

## 15. Testing

Required test categories:

```text
tensor arithmetic
broadcasting
matrix multiplication
shape operations
autograd
finite difference gradient checks
parameter registration
serialization
optimizer correctness
normalization
attention
causal masking
transformer blocks
cross entropy
training step
checkpoint resume
KV cache
generation
CLI
debugger data endpoints
```

Tests must cover edge cases.

At minimum test:

1. Scalar gradients
2. Branched computation graphs
3. Reused tensors
4. Broadcasting
5. Zero sized or invalid dimensions where supported
6. Numerically large softmax inputs
7. Causal mask correctness
8. Variable sequence lengths if supported
9. Gradient clipping
10. Checkpoint restoration

No failing test may be ignored to finish the project.

---

## 16. Benchmark Suite

Create reproducible experiments comparing TorchZero against a reference framework on identical small models.

Measure:

1. Forward latency
2. Backward latency
3. Training step latency
4. Tokens per second
5. Peak memory when available
6. Numerical output error
7. Gradient error
8. Checkpoint size

Benchmark several:

```text
sequence lengths
batch sizes
embedding dimensions
layer counts
head counts
```

Keep configurations small enough to run reliably.

Store raw results under:

```text
experiments/results/
```

Generate figures programmatically.

Never invent results.

---

## 17. Research Report

Create `RESEARCH_REPORT.md`.

Required sections:

### Abstract

### Motivation

### System Architecture

### Automatic Differentiation Design

### Transformer Implementation

### Experimental Method

### Research Questions

At minimum:

1. How closely can a compact autodiff engine reproduce reference framework gradients?
2. Which operations dominate runtime in a small from scratch transformer?
3. How does sequence length change the fraction of time spent in attention?
4. What memory costs arise from retaining the dynamic computation graph?
5. How much does KV caching improve autoregressive generation?
6. What implementation decisions produce the largest correctness or performance tradeoffs?

### Correctness Results

Include gradient error distributions and full model parity measurements.

### Performance Results

Include generated plots and tables.

### Ablations

At minimum evaluate:

```text
KV cache on versus off
different sequence lengths
different batch sizes
different model depths
```

### Failure Analysis

Document important implementation failures discovered during development.

Examples:

```text
incorrect broadcasting gradients
unstable softmax
causal mask bugs
checkpoint mismatch
gradient accumulation bugs
```

### Limitations

State precisely what TorchZero does not implement.

### Reproducibility

Provide exact commands.

### Future Work

Base proposals on observed limitations.

---

## 18. Autonomous Completion Loop

The agent must repeatedly execute this loop.

### 1. Inspect

Read the repository, history, open documentation, test state, and project configuration.

### 2. Maintain State

Create and maintain:

```text
docs/PLAN.md
docs/STATUS.md
docs/DECISIONS.md
docs/KNOWN_ISSUES.md
```

### 3. Prioritize

Work in this order unless dependencies require otherwise:

1. Tensor correctness
2. Autograd correctness
3. Gradient checks
4. Module system
5. Optimizers
6. Transformer correctness
7. Training
8. Checkpointing
9. Inference
10. KV caching
11. Debugger
12. Tests
13. Performance
14. Benchmarks
15. Research
16. Documentation
17. Polish

### 4. Implement Completely

Do not leave placeholders.

A function is incomplete if it contains production behavior equivalent to:

```text
TODO
NotImplemented
pass
fake metrics
mocked results
hardcoded benchmark numbers
disabled user interface paths
```

### 5. Test

Run focused tests after each implementation.

Run full tests periodically and before every milestone.

### 6. Investigate Failures

Fix root causes.

Add regression tests.

Never weaken correctness thresholds without documented numerical justification.

### 7. Commit

Commit every coherent set of changes.

Examples:

```text
feat: implement reverse mode autograd traversal
test: add finite difference broadcast gradient checks
feat: add adamw optimizer state serialization
feat: implement causal multi head attention
feat: add kv cached generation
ui: add backward graph debugger
bench: add sequence length scaling experiment
research: analyze attention runtime scaling
```

All files created by the agent must be committed unless intentionally excluded.

### 8. Research

Once end to end execution works, begin experiments while continuing implementation.

Unexpected benchmark or correctness behavior must become investigation tasks.

### 9. Reevaluate

Update project status.

Check all completion gates.

Repeat automatically while any gate remains false.

---

## 19. Git Protocol

Initialize Git if needed.

Create `.gitignore` immediately.

Never commit:

```text
credentials
API tokens
private keys
large datasets
large model checkpoints unless explicitly small fixtures
virtual environments
temporary caches
build directories
```

Commit:

```text
source
tests
small fixtures
configs
benchmark scripts
raw compact experiment summaries
plots
documentation
research report
```

Use descriptive commits.

Do not batch the entire project into one final commit.

Before push:

1. Run all tests.
2. Run static checks.
3. Train the tiny validation model.
4. Resume it from a checkpoint.
5. Generate text.
6. Run the benchmark suite.
7. Regenerate research figures.
8. Build the debugger.
9. Follow README setup from a clean environment.
10. Scan for secrets.
11. Verify `git status` is clean.

If no remote exists and GitHub CLI is authenticated, create one.

If a remote exists, preserve it.

Push only when all completion gates pass.

---

## 20. Completion Gates

TorchZero is complete only when:

1. Tensor operations work.
2. Broadcasting works.
3. Reverse mode autodiff works.
4. Finite difference gradient checks pass.
5. Matrix multiplication gradients pass.
6. Module parameter registration works.
7. State serialization works.
8. SGD works.
9. Adam works.
10. AdamW works.
11. Tokenization pipeline works.
12. Data batching works.
13. Transformer forward execution works.
14. Causal attention is correct.
15. Cross entropy works.
16. Full model backward execution works.
17. A tiny model trains and loss decreases on a controlled task.
18. Checkpoints save.
19. Checkpoints reload.
20. Training resumes from checkpoint.
21. Autoregressive generation works.
22. KV caching works.
23. KV cached generation matches uncached deterministic generation.
24. Debugger exposes real forward data.
25. Debugger exposes real backward and gradient data.
26. Reference parity tests pass within documented tolerance.
27. Unit tests pass.
28. Integration tests pass.
29. Static checks pass.
30. Benchmark suite completes.
31. Raw benchmark results are stored.
32. Research figures regenerate from raw results.
33. `RESEARCH_REPORT.md` is complete.
34. `ARCHITECTURE.md` is complete.
35. README instructions work from a clean environment.
36. Repository contains no secrets.
37. Git working tree is clean.
38. Final commit exists.
39. Repository has been pushed successfully.

Only then may the autonomous run terminate.

---

## 21. Final Audit

Create a fresh environment and perform:

```text
install
unit tests
gradient checks
tiny model training
checkpoint restore
generation
benchmark
debugger build
research report regeneration
```

Review every research claim against experiment data.

Review all commands in the README.

If anything cannot be reproduced, the project remains incomplete.

---

## 22. Integrity Rule

The purpose of TorchZero is to expose mechanics that frameworks usually hide.

Do not obscure those mechanics with framework calls simply to satisfy completion gates.

When external numerical libraries are used, document the exact primitive delegated to them and what TorchZero implements around it.

Every claim of correctness must be backed by a test.

Every claim of performance must be backed by a benchmark.

Every research conclusion must be backed by stored results.
