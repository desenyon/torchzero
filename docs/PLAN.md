# TorchZero Plan

Living plan maintained during the autonomous completion loop. Order follows
README §18.3 unless dependencies force otherwise.

## Milestones

1. [x] Repository bootstrap: git init, .gitignore, venv, pyproject, state docs.
2. [ ] Tensor engine (storage, shape, strides, dtype, grad fields, detach).
3. [ ] Operations: arithmetic, reductions, matmul, shape ops, broadcasting,
       exp/log/sqrt/tanh, softmax, indexing/embedding gather, concat.
4. [ ] Reverse-mode autograd engine (topo sort, accumulation, non-scalar roots).
5. [ ] Finite-difference gradient checks for every differentiable primitive.
6. [ ] Module system: Module/Parameter/ModuleList/Sequential, state_dict,
       train/eval, zero_grad.
7. [ ] Optimizers SGD / Adam / AdamW + gradient clipping + LR schedule.
8. [ ] Byte-level BPE tokenizer (train, encode, decode, save/load).
9. [ ] Data pipeline: load, tokenize, split, pack, batch, seeded shuffle.
10. [ ] Decoder-only transformer: embeddings + RoPE, causal MHA, RMSNorm, FFN,
       blocks, LM head, cross entropy, autoregressive generation, KV cache.
11. [ ] Training runtime: loop, clipping, schedule, eval, logging, checkpoints,
       resume.
12. [ ] CLI: `torchzero train|generate|inspect|benchmark`.
13. [ ] Debugger: forward trace + backward trace from real executions; inspect UI.
14. [ ] Test suite: all README §15 categories incl. edge cases.
15. [ ] Benchmark suite + raw results in experiments/results/.
16. [ ] Research figures + RESEARCH_REPORT.md.
17. [ ] Docs: README, ARCHITECTURE.md, CHANGELOG.md, LICENSE.
18. [ ] Final audit from clean environment; push.

## Completion gate tracking

Tracked in docs/STATUS.md as a checklist of README §20 gates 1-39.

## Key design decisions

See docs/DECISIONS.md. Notably:

- numpy is the single external numerical kernel backend (README §1 permits this);
  the boundary is documented in ARCHITECTURE.md and enforced by convention:
  torchzero code never calls another ML framework.
- No external ML framework even for tests; reference comparisons use independent
  double-precision implementations written in this repo (documented tolerances).
