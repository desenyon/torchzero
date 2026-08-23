"""Research experiment: KV-cache generation speedup as a function of the
number of generated tokens. Median over repeats. Determinism checked at
every length via greedy equivalence.

Raw -> experiments/results/kv_cache_scaling.json
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero.transformer import Transformer, TransformerConfig


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    cfg_kwargs = dict(vocab_size=256, dim=96, n_layers=4, n_heads=4,
                      block_size=192)
    model = Transformer(TransformerConfig(**cfg_kwargs), seed=42)
    model.eval()

    prompt = np.random.default_rng(0).integers(0, 256, (1, 16)) \
        .astype(np.int64)
    prompt_ids = list(prompt[0])
    repeats = 3
    rows = []
    for n_new in [8, 16, 32, 64, 128]:
        cached_times, uncached_times = [], []
        match = True
        for r in range(repeats):
            t0 = time.perf_counter()
            g_cached = model.generate(list(prompt_ids), n_new,
                                      temperature=0.9, use_kv_cache=True,
                                      seed=r)
            cached_times.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            g_uncached = model.generate(list(prompt_ids), n_new,
                                        temperature=0.9, use_kv_cache=False,
                                        seed=r)
            uncached_times.append(time.perf_counter() - t0)

            # greedy determinism check at this length
            gc_ = model.generate(list(prompt_ids), n_new, temperature=0.0,
                                 use_kv_cache=True)
            gu_ = model.generate(list(prompt_ids), n_new, temperature=0.0,
                                 use_kv_cache=False)
            match = match and gc_[:len(gu_)] == gu_[:len(gu_)]

        cached_s = float(np.median(cached_times))
        uncached_s = float(np.median(uncached_times))
        rows.append({
            "new_tokens": n_new,
            "cached_seconds": cached_s,
            "uncached_seconds": uncached_s,
            "speedup": uncached_s / cached_s,
            "cached_tokens_per_s": n_new / cached_s,
            "uncached_tokens_per_s": n_new / uncached_s,
            "greedy_outputs_match": bool(match),
        })
        print(f"+{n_new:>3} tokens: cached {cached_s:.3f}s "
              f"({n_new/cached_s:7.0f} tok/s)  "
              f"uncached {uncached_s:.3f}s ({n_new/uncached_s:7.0f} tok/s)  "
              f"speedup {uncached_s/cached_s:.2f}x  match={match}")

    out = os.path.join(results_dir, "kv_cache_scaling.json")
    json.dump({"name": "kv_cache_scaling", "prompt_len": len(prompt_ids),
               "model": cfg_kwargs, "repeats": repeats, "rows": rows},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
