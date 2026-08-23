# Changelog

## 0.1.0 — initial complete implementation

### Added
- Tensor engine: strided numpy-backed storage, dtypes, gradient fields,
  full operation set (arithmetic, reductions, matmul, shape ops,
  broadcasting, exp/log/sqrt/tanh, stable softmax/log_softmax, fused cross
  entropy, indexing/gather/scatter-add, concat/split, masked_fill).
- Reverse-mode autodiff: iterative topological traversal, gradient
  accumulation across branches and reused tensors, explicit non-scalar root
  gradients, graph freeing with loud double-backward detection, invalid
  backward validation.
- Module system: Module/Parameter/ModuleList/Sequential, automatic parameter
  registration, state_dict save/load, train/eval, zero_grad; Linear,
  Embedding, LayerNorm, RMSNorm, Dropout, GELU/SiLU.
- Optimizers: SGD (momentum, nesterov, weight decay), Adam, AdamW (decoupled
  decay); global-norm gradient clipping; warmup-cosine LR scheduling.
- Byte-level BPE tokenizer with deterministic training and JSON persistence.
- Data pipeline: split/pack/batch with SHA-256-seeded deterministic shuffling.
- Decoder-only transformer: RoPE causal multi-head attention, RMSNorm pre-norm
  blocks, GELU MLPs, weight-tied LM head, KV cache, autoregressive generation.
- Training runtime: loop, evaluation, metric history, checkpointing with full
  reproducible resume state.
- CLI: torchzero train / generate / inspect / benchmark.
- Execution debugger: real forward/backward traces, text + HTML reports.
- Test suite: 165 tests incl. finite-difference gradient checks for every
  primitive, plain-Python reference parity, PyTorch full-model parity,
  CLI end-to-end, static forbidden-import check.
- Benchmark suite with raw results in experiments/results/ and programmatic
  figures in research/figures/.
- Research report with experiment-backed analysis.
