"""Research experiment: distribution of gradient errors vs PyTorch.

For each configuration, transfers identical weights into a torch mirror,
runs the same forward/backward, and records the per-tensor relative gradient
error (max abs diff / max abs reference) for every parameter, grouped by
parameter type. Also repeats across depths to test whether error accumulates
with depth.

Requires torch (reference framework, benchmarks only).
Raw -> experiments/results/grad_error_distribution.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.run import pair_models


def grad_errors(tz_model, ref, cfg):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, cfg.vocab_size, (2, min(cfg.block_size, 32))) \
        .astype(np.int64)
    targets = np.roll(ids, -1, axis=1)

    _, loss_tz = tz_model.forward(ids, targets=targets)
    loss_tz.backward()
    import torch
    ref.zero_grad()
    logits = ref(torch.tensor(ids))
    loss_pt = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.vocab_size).float(),
        torch.tensor(targets.reshape(-1)))
    loss_pt.backward()

    rows = []
    named = tz_model.named_parameters()
    for name, p in sorted(named.items()):
        parts = name.split(".")
        obj = ref
        try:
            for part in parts:
                obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
        except AttributeError:
            continue
        g_pt = None if obj.grad is None else obj.grad.numpy()
        if g_pt is None or p.grad is None:
            continue
        # torch Linear weight layout is transposed
        if "weight" in name and ("attn." in name or "mlp." in name):
            g_pt = g_pt.T
        denom = max(float(np.abs(g_pt).max()), 1e-30)
        rel = float(np.abs(p.grad - g_pt).max()) / denom
        ptype = ("embedding" if "tok_emb" in name
                 else "norm" if "norm" in name
                 else "attention" if ".attn." in name
                 else "mlp")
        rows.append({"name": name, "type": ptype,
                     "shape": list(p.grad.shape),
                     "rel_err": rel})
    return rows


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    configs = [
        {"label": "depth2_dim32", "vocab_size": 64, "dim": 32,
         "n_layers": 2, "n_heads": 4, "block_size": 16},
        {"label": "depth4_dim48", "vocab_size": 128, "dim": 48,
         "n_layers": 4, "n_heads": 4, "block_size": 24},
        {"label": "depth6_dim48", "vocab_size": 128, "dim": 48,
         "n_layers": 6, "n_heads": 4, "block_size": 24},
        {"label": "depth8_dim64", "vocab_size": 128, "dim": 64,
         "n_layers": 8, "n_heads": 8, "block_size": 24},
    ]

    all_rows = []
    for kw in configs:
        label = kw.pop("label")
        tz_model, ref, torch, cfg = pair_models(seed=0, **kw)
        if ref is None:
            print("torch not available; skipping")
            return
        rows = grad_errors(tz_model, ref, cfg)
        errs = [r["rel_err"] for r in rows]
        by_type = {}
        for r in rows:
            by_type.setdefault(r["type"], []).append(r["rel_err"])
        summary = {
            "config": {k: v for k, v in kw.items()},
            "num_tensors": len(rows),
            "rel_err_median": float(np.median(errs)),
            "rel_err_p95": float(np.percentile(errs, 95)),
            "rel_err_max": float(np.max(errs)),
            "by_type_median": {k: float(np.median(v))
                               for k, v in by_type.items()},
        }
        all_rows.append({"label": label, **summary,
                         "tensors": rows})
        print(f"{label}: median {summary['rel_err_median']:.2e} "
              f"p95 {summary['rel_err_p95']:.2e} "
              f"max {summary['rel_err_max']:.2e} "
              f"(types: " + ", ".join(
                  f"{k}={float(np.median(v)):.1e}"
                  for k, v in by_type.items()) + ")")

    out = os.path.join(results_dir, "grad_error_distribution.json")
    json.dump({"name": "grad_error_distribution", "configs": all_rows},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
