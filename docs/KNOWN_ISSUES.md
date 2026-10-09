# Known Issues

Investigations in progress or resolved. Resolved items stay here with their fix
and regression test reference (README §18.6: investigate, don't bypass).

| ID | Status | Summary | Resolution |
|----|--------|---------|------------|
| KI-1 | resolved | Cross-entropy produced negative loss: sign error in `nll = -(shifted) - lse` instead of `lse - shifted` | Fixed in tensor.py; regression: tests/test_gradcheck.py::test_cross_entropy_gradient |
| KI-2 | resolved | Split backward returned only the sliced gradient, shape mismatch vs parent | Return zero-filled full-shape gradient with slice set; tests/test_tensor.py::test_split |
| KI-3 | resolved | KV-cached attention mixed new queries against new values only (`probs @ v`), cached values ignored | Use `probs @ v_full`; regression: tests/test_transformer.py::test_cached_matches_uncached |
| KI-4 | resolved | RoPE applied to (B,H,T,D) while helper assumed seq axis at dim 1 -> wrong positions/broadcast errors | Apply RoPE to (B,T,H,D) before transpose; regression: full test_transformer.py suite |
| KI-5 | resolved | Stale KV cache length used as RoPE offset when reset_cache=True on a model with leftover cache | Compute offset after cache reset logic; caught by cached-vs-uncached equivalence test |
| KI-6 | resolved | `tie_weights` config existed but was never applied; lm_head had independent random weights -> full-model parity vs torch failed at O(1) while layer parity passed | Implemented true tying: logits = h @ E^T through an autograd transpose node so both embedding paths accumulate gradient; regression: tests/test_reference_parity.py::test_full_model_logits_parity |
| KI-7 | resolved | Benchmark grad comparison initially showed ~100% "error": torch nn.Linear stores (out,in), TorchZero (in,out); shape-based transpose detection silently fails for square matrices | Explicit per-parameter transpose flags in the mapping table; TorchZero grads were verified correct by finite differences first |


## Version 0.2 reliability regressions

- **Resolved:** min sign, tensor-power derivatives, vector/batched matmul,
  negative transpose axes and zero-exponent gradients. Independent finite
  differences and edge cases: `tests/test_numerical_regressions.py`.
- **Resolved:** gather overwrote duplicate gradients; scatter-add ignored its
  axis and detached the graph. Both operands and forward values are tested.
- **Resolved:** each update restarted sampling; row-end targets wrapped. A
  persistent cursor and lookahead packing now cover real adjacent tokens.
- **Resolved:** SGD momentum restore wrote temporary dictionaries; checkpoints
  saved unrelated RNG and omitted final sampler state. Version-2 complete
  checkpoints pass actual split/resume tests for all optimizers with dropout.
- **Resolved:** dropout configuration was unused; seeds reset on each call;
  generation retained graphs and changed module modes. Persistent streams,
  no_grad generation and exception-safe mode/cache cleanup are tested.
- **Resolved:** file paths were tokenized literally; resume could replace its
  tokenizer. Real UTF-8 CLI train/resume/generate/inspect tests protect this path.
- **Intentional limits:** v1 exact training resume is unsupported; v1 inference
  remains supported. Corpus storage is in memory, pickle requires trust and
  shared model/grad-mode state is not thread-safe. See README.
