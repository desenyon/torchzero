"""Regenerate all research figures from raw experiment JSON results.

Reads experiments/results/*.json and writes PNG figures into
research/figures/. Never invents data points: everything plotted comes from
stored results.
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "experiments", "results")
FIGURES = os.path.join(ROOT, "research", "figures")


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        print(f"warning: missing {name}")
        return None
    return json.load(open(path))


def fig_latency(data):
    rows = data["rows"]
    labels = [f"B{r['batch']}\nT{r['seq']} D{r['dim']}\nL{r['layers']} H{r['heads']}"
              for r in rows]
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar([i - width / 2 for i in x], [r["tz_step_ms"] for r in rows],
           width, label="TorchZero", color="#d62728")
    has_torch = any("torch_step_ms" in r for r in rows)
    if has_torch:
        ax.bar([i + width / 2 for i in x],
               [r.get("torch_step_ms", 0) for r in rows],
               width, label="PyTorch (reference)", color="#1f77b4")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("training step latency (ms)")
    ax.set_title("Training step latency: TorchZero vs PyTorch")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "latency_comparison.png"), dpi=150)
    plt.close(fig)


def fig_attention_scaling(data):
    rows = data["rows"]
    ts = [r["seq_len"] for r in rows]
    fracs = [100 * r["attn_fraction"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ts, fracs, marker="o", color="#d62728")
    for t, f in zip(ts, fracs):
        ax.annotate(f"{f:.1f}%", (t, f), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("% of training step inside attention (%)")
    ax.set_title("Attention time fraction vs sequence length "
                 "(dim=128, 4 layers, batch=8)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "attention_scaling.png"), dpi=150)
    plt.close(fig)


def fig_kv_cache(data):
    rows = data["rows"]
    labels = [f"prompt {r['prompt_len']}\n+{r['new_tokens']} tokens"
              for r in rows]
    x = range(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar([i - width / 2 for i in x],
           [r["uncached_tokens_per_s"] for r in rows], width,
           label="no KV cache", color="#7f7f7f")
    ax.bar([i + width / 2 for i in x],
           [r["cached_tokens_per_s"] for r in rows], width,
           label="KV cache", color="#2ca02c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("generation tokens/second")
    ax.set_title(f"KV cache generation speedup "
                 f"({', '.join(str(round(r['speedup'], 2)) + 'x' for r in rows)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "kv_cache_ablation.png"), dpi=150)
    plt.close(fig)


def fig_op_profile(data):
    profile = data["per_op_backward_ms"]
    items = sorted(profile.items(), key=lambda kv: -kv[1])[:10]
    labels = [k.replace("composite", "op") for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = range(len(items))[::-1]
    ax.barh(y, vals, color="#ff7f0e")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("median backward time per node type (ms)")
    ax.set_title("Where backward time goes "
                 f"(forward {data['forward_wall_ms_median']:.1f} ms / "
                 f"backward {data['backward_wall_ms_median']:.1f} ms wall)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "op_profile.png"), dpi=150)
    plt.close(fig)


def fig_training_curve(data):
    rows = data["rows"]
    steps = [r["step"] for r in rows]
    loss = [r["loss"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(steps, loss, label="train loss", color="#1f77b4")
    val = [(r["step"], r["val_loss"]) for r in rows if "val_loss" in r]
    if val:
        ax.plot([s for s, _ in val], [l for _, l in val], marker="o",
                linestyle="--", label="validation loss", color="#2ca02c")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross entropy loss")
    ax.set_ylim(bottom=0)
    ax.set_title(f"Training curve (dim={data['model']['dim']}, "
                 f"L={data['model']['n_layers']}, "
                 f"{data['vocab_size']} BPE vocab)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "training_curve.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    specs = [
        ("latency_scaling.json", fig_latency),
        ("attention_scaling.json", fig_attention_scaling),
        ("kv_cache_ablation.json", fig_kv_cache),
        ("op_profile.json", fig_op_profile),
        ("training_curve.json", fig_training_curve),
    ]
    produced = []
    for name, fn in specs:
        data = load(name)
        if data is None:
            continue
        fn(data)
        produced.append(name)
    print(f"figures written to {FIGURES} from: {', '.join(produced)}")


if __name__ == "__main__":
    main()
