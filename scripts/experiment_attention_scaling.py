"""Research experiment: how does sequence length change the fraction of
step time spent inside attention?

Instruments real executions by wrapping every block's attention module with
a wall-clock accumulator, then runs full training steps at increasing
sequence lengths. Raw results -> experiments/results/attention_scaling.json.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero.transformer import Transformer, TransformerConfig
from torchzero.optim import AdamW, clip_grad_norm


def timed_attention_fraction(T, dim=128, n_layers=4, n_heads=4,
                             batch=8, vocab=512, steps=6):
    cfg = TransformerConfig(vocab_size=vocab, dim=dim, n_layers=n_layers,
                            n_heads=n_heads, block_size=max(T, 16))
    model = Transformer(cfg, seed=0)
    opt = AdamW(model.parameters(), lr=1e-3)

    attn_time = [0.0]
    for blk in model.blocks:
        orig = blk.attn.forward

        def wrapped(*args, _orig=orig, **kwargs):
            t0 = time.perf_counter()
            out = _orig(*args, **kwargs)
            attn_time[0] += time.perf_counter() - t0
            return out

        blk.attn.forward = wrapped

    rng = np.random.default_rng(T)
    ids = rng.integers(0, vocab, (batch, T)).astype(np.int64)
    targets = np.roll(ids, -1, axis=1)

    # warmup
    _, loss = model.forward(ids, targets=targets)
    loss.backward()
    for p in model.parameters():
        p.grad = None

    total_time = 0.0
    for _ in range(steps):
        attn_time[0] = 0.0
        t0 = time.perf_counter()
        _, loss = model.forward(ids, targets=targets)
        loss.backward()
        clip_grad_norm(model.parameters(), 1.0)
        opt.step()
        for p in model.parameters():
            p.grad = None
        total_time += time.perf_counter() - t0
        step_attn = attn_time[0]

    frac = step_attn / (total_time / steps)
    return {
        "seq_len": T,
        "step_ms": (total_time / steps) * 1000,
        "attn_ms": step_attn * 1000,
        "attn_fraction": frac,
    }


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for T in [16, 32, 64, 128, 256]:
        row = timed_attention_fraction(T)
        rows.append(row)
        print(f"T={T:>3}: step {row['step_ms']:7.1f}ms  "
              f"attention {row['attn_ms']:7.1f}ms  "
              f"fraction {row['attn_fraction']*100:5.1f}%")

    out = os.path.join(results_dir, "attention_scaling.json")
    json.dump({"name": "attention_scaling", "rows": rows},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
