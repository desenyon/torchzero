# Reliability upgrade verification

Local verification was performed on macOS arm64 with Python 3.13.12,
NumPy 2.5.3 and pytest 9.1.1. BLAS/OpenMP thread limits were set to 1.
The starting commit was `70efb3a717ec75b6dbf843b098e731c375298422`.

| Check | Observed result |
| --- | --- |
| Original suite before implementation | 166 passed |
| New numerical regressions against original code | 14 failures reproduced (3 control cases passed) |
| Initial runtime reliability regressions | 30 failures reproduced |
| Final complete checkout suite | 232 passed |
| Complete suite from the built source archive | 232 passed |
| `python -m ruff check .` | Passed configured correctness rules |
| `python -m compileall -q torchzero debugger benchmarks` | Passed |
| `git diff --check` | Passed |
| `python -m build --no-isolation` | Wheel and source archive built successfully |
| Fresh wheel environment, outside checkout | Imported installed 0.2.0 package; train/pause/resume/generate/inspect passed |
| Greedy and seeded cached vs uncached wheel generation | Complete output matched |
| README tensor/model/data examples | Executed successfully |

The source-archive audit initially found missing test helper modules and config
files. `MANIFEST.in` now includes the inputs needed to run the full suite from
that artifact; the resulting archive was extracted and tested independently.

Exact-resume tests compare every model array, loss, gradient norm, learning
rate, validation loss, sampler position and dropout RNG state after real
interruptions. They cover SGD, Adam and AdamW, both checkpoint filenames,
active dropout, a partial batch and a fresh model initialized with a different
seed. Tests also cover data/plan mismatch rejection, legacy weight loading,
unsupported versions, failed atomic writes and inference exception cleanup.

CI is defined in `.github/workflows/ci.yml` for Python 3.10 and 3.13. It runs
checkout tests, lint, compilation, distribution builds, source-archive tests
and installed-wheel CLI smoke checks. Remote status must be verified against
the pushed commit; this local record does not substitute for that check.

No type checker is configured (annotations are partial). Ruff checks E9,
F63, F7 and F82, rather than imposing a style rewrite on the existing code.
Optional PyTorch comparisons, full research experiments and figure generation
were not run; no benchmark/performance claims or historical results changed.
