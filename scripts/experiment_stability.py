"""Research experiment: numerical stability of softmax / cross entropy under
extreme logit magnitudes. Measures loss value, max |gradient|, and gradient
finiteness across logit scales. Raw -> experiments/results/stability.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero import Tensor


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    rows = []
    for scale in [1, 10, 100, 1000, 5000, 10000]:
        rng = np.random.default_rng(scale)
        logits = rng.standard_normal((8, 64)).astype(np.float32) * scale
        t = Tensor(logits, requires_grad=True)
        targets = rng.integers(0, 64, 8)
        loss = Tensor.cross_entropy(t, targets)
        loss.backward()
        g = t.grad
        finite_loss = bool(np.isfinite(loss.data))
        finite_grad = bool(np.isfinite(g).all())
        # reference in float64 via stable formula
        x64 = logits.astype(np.float64)
        shifted = x64 - x64.max(-1, keepdims=True)
        lse = np.log(np.exp(shifted).sum(-1))
        ref = float((lse - shifted[np.arange(8), targets]).mean())
        rows.append({
            "logit_scale": scale,
            "loss": float(loss.item()),
            "loss_float64_reference": ref,
            "abs_error": abs(float(loss.item()) - ref),
            "grad_abs_max": float(np.abs(g).max()),
            "finite": bool(finite_loss and finite_grad),
        })
        print(f"scale {scale:>5}: loss {float(loss.item()):12.5f} "
              f"ref {ref:12.5f} err {rows[-1]['abs_error']:.2e} "
              f"|g|max {rows[-1]['grad_abs_max']:.3f} "
              f"finite={rows[-1]['finite']}")

    out = os.path.join(results_dir, "stability.json")
    json.dump({"name": "stability", "rows": rows}, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
