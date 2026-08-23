"""Research experiment: optimizer ablation on a controlled next-token task.

SGD (momentum 0.9) vs Adam vs AdamW, identical model/task/seed set.
3 seeds each; loss curves recorded per seed so mean/std can be computed.
Also compares warmup-cosine vs constant LR schedule for AdamW.

Raw -> experiments/results/optimizer_ablation.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero.transformer import Transformer, TransformerConfig
from torchzero.optim import SGD, Adam, AdamW, LambdaLR, clip_grad_norm


def run(optimizer_name, seed, steps=250, schedule="warmup_cosine",
        lr=3e-3):
    vocab, T = 48, 12
    cfg = TransformerConfig(vocab_size=vocab, dim=24, n_layers=2,
                            n_heads=3, block_size=T)
    model = Transformer(cfg, seed=seed)
    model.train()
    params = model.parameters()

    if optimizer_name == "sgd":
        opt = SGD(params, lr=lr * 10, momentum=0.9, weight_decay=0.01)
    elif optimizer_name == "adam":
        opt = Adam(params, lr=lr)
    else:
        opt = AdamW(params, lr=lr, weight_decay=0.01)

    if schedule == "warmup_cosine":
        sched = LambdaLR(opt, lambda s: min(
            s / 25.0, 1.0) if s < 25 else
            0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * min(1.0, (s - 25)
                                                      / (steps - 25)))))
    else:
        sched = None

    rng = np.random.default_rng(seed + 1000)
    ids = rng.integers(0, vocab, (4, T)).astype(np.int64)
    targets = np.roll(ids, -1, axis=1)

    curve = []
    for step in range(1, steps + 1):
        _, loss = model.forward(ids, targets=targets)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(params, 1.0)
        opt.step()
        if sched is not None:
            sched.step()
        curve.append(float(loss.item()))
    return curve


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    runs = {}
    for name in ["sgd", "adam", "adamw"]:
        for seed in [0, 1, 2]:
            key = f"{name}_seed{seed}"
            runs[key] = {"optimizer": name, "schedule": "warmup_cosine",
                         "curve": run(name, seed)}
            print(f"{key}: first {runs[key]['curve'][0]:.3f} "
                  f"final {runs[key]['curve'][-1]:.4f}")
    # schedule ablation on adamw
    for seed in [0, 1, 2]:
        key = f"adamw_constant_seed{seed}"
        runs[key] = {"optimizer": "adamw", "schedule": "constant",
                     "curve": run("adamw", seed, schedule="constant")}
        print(f"{key}: final {runs[key]['curve'][-1]:.4f}")

    def stats(name):
        finals = [np.mean(runs[k]["curve"][-20:]) for k in runs
                  if k.startswith(name + "_seed")]
        return {"final_loss_mean": float(np.mean(finals)),
                "final_loss_std": float(np.std(finals))}

    summary = {n: stats(n) for n in ["sgd", "adam", "adamw"]}
    const_finals = [np.mean(runs[k]["curve"][-20:])
                    for k in runs if k.startswith("adamw_constant_seed")]
    summary["adamw_constant_lr"] = {
        "final_loss_mean": float(np.mean(const_finals)),
        "final_loss_std": float(np.std(const_finals))}

    out = os.path.join(results_dir, "optimizer_ablation.json")
    json.dump({"name": "optimizer_ablation", "steps": 250,
               "summary_last20_mean": summary, "runs": runs},
              open(out, "w"), indent=1)
    print(json.dumps(summary, indent=1))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
