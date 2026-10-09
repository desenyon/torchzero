# Numerical and training reliability upgrade

## Intent and scope

Keep TorchZero's NumPy-only educational framework and public entry points,
while making differentiation and the train/resume/generate path trustworthy.
The starting point is commit `70efb3a`; its recent work focused on experiments
and framework-boundary checks. This upgrade targets defects in the underlying
execution path, without introducing another numerical framework.

## Design

1. Use one rank-promotion rule for matmul backward; correctly differentiate
   tensor powers, min, duplicate gathers and axis-aware scatter-add. Prove the
   derivatives with independent central finite differences for both operands.
2. Retain `pack_sequences` as the general packing utility. Add language-model
   packing with one lookahead token per row; share next-token batch extraction
   between evaluation and training. No target may wrap to the row's first token.
3. Give the sampler an explicit epoch/cursor state and a consuming next-batch
   API. Each row is visited once per epoch, including the final partial batch.
4. Separate checkpoint I/O from the trainer. Write versioned checkpoints by
   atomic replacement; save real model RNG, optimizer, scheduler, sampler and
   corpus identity. Resume a fixed training plan exactly, including dropout.
   Read version-1 weights for inference, but reject version-1 exact training
   resume because its batch/target/RNG contract was incomplete.
5. Activate configured dropout using persistent, independently seeded streams.
   Generation uses one RNG per call under no_grad, restores module modes and
   releases its temporary cache even on errors. Context capacity is explicit.
6. Read corpus files as UTF-8 and distinguish them from inline text. Bind saved
   tokenization to resume so a changed corpus or vocabulary cannot silently
   continue a different task. Validate configurations and input boundaries.

## Implementation and verification sequence

- Establish the full pytest baseline with single-threaded BLAS.
- Add failing numerical regressions, implement primitives, run focused checks.
- Add failing sampler, checkpoint, dropout and inference tests; implement the
  state lifecycle and exercise interrupted versus uninterrupted training for
  SGD, Adam and AdamW with dropout and partial batches.
- Exercise a real CLI corpus-file train, resume, generate and inspect workflow.
- Expand README setup, architecture, API, configuration, checkpoint trust,
  migration, verification and limitations; update architecture and change log.
- Run the full suite, static checks, distribution build and installed-wheel
  smoke checks. Commit and push a new branch, then verify remote SHA and CI.

Timing and benchmark artifacts are not regenerated or represented as evidence
for this implementation. No deployment, release or merge is part of this work.
