"""Regenerate all research figures from raw experiment JSON results.

Reads experiments/results/*.json and writes PNG figures into
research/figures/. Never invents data points: everything plotted comes from
stored results.
"""

from __future__ import annotations

import json
import os

import numpy as np
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
    profile = data["forward_stage_ms"]
    items = sorted(profile.items(), key=lambda kv: -kv[1])[:10]
    labels = [k for k, _ in items]
    vals = [v for _, v in items]
    total = sum(vals)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y = range(len(items))[::-1]
    bars = ax.barh(y, vals, color="#ff7f0e")
    for rect, v in zip(bars, vals):
        ax.text(rect.get_width() + max(vals) * 0.01, rect.get_y() + 0.35,
                f"{100 * v / total:.0f}%", fontsize=9)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("median forward time per module type (ms)")
    ax.set_title("Forward-pass time by module type "
                 f"(fwd {data['forward_wall_ms_median']:.1f} ms / "
                 f"bwd {data['backward_wall_ms_median']:.1f} ms wall)")
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


def fig_graph_memory(data):
    rows = data["rows"]
    depths = sorted({r["layers"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    colors = {2: "#1f77b4", 4: "#ff7f0e", 8: "#d62728"}
    for L in depths:
        rs = sorted([r for r in rows if r["layers"] == L],
                    key=lambda r: r["seq_len"])
        ts = [r["seq_len"] for r in rs]
        mb = [r["activation_bytes"] / 1e6 for r in rs]
        shares = [100 * r["attn_share_of_activations"] for r in rs]
        c = colors.get(L)
        axes[0].plot(ts, mb, marker="o", color=c, label=f"{L} layers")
        axes[1].plot(ts, shares, marker="s", color=c,
                     label=f"{L} layers")
    axes[0].set_xlabel("sequence length T")
    axes[0].set_ylabel("retained activation memory (MB)")
    axes[0].set_title("Dynamic-graph retained activations (B=8)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set_xlabel("sequence length T")
    axes[1].set_ylabel("attention-probability share (%)")
    axes[1].set_title("Attention tensors as share of retained bytes")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "graph_memory.png"), dpi=150)
    plt.close(fig)


def fig_kv_scaling(data):
    rows = data["rows"]
    ns = [r["new_tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(ns, [r["uncached_tokens_per_s"] for r in rows], marker="o",
            color="#7f7f7f", label="no KV cache")
    ax.plot(ns, [r["cached_tokens_per_s"] for r in rows], marker="o",
            color="#2ca02c", label="KV cache")
    for r in rows:
        ax.annotate(f"{r['speedup']:.1f}x",
                    (r["new_tokens"], r["cached_tokens_per_s"]),
                    textcoords="offset points", xytext=(0, 9), fontsize=9,
                    color="#2ca02c")
    ax.set_xlabel("generated tokens")
    ax.set_ylabel("generation throughput (tokens/s)")
    ax.set_title(f"KV-cache speedup vs generation length "
                 f"(prompt={data['prompt_len']}, median of "
                 f"{data['repeats']} runs)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "kv_cache_scaling.png"), dpi=150)
    plt.close(fig)


def _smooth(xs, k=15):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k // 2)
        hi = min(len(xs), i + k // 2 + 1)
        out.append(sum(xs[lo:hi]) / (hi - lo))
    return out


def fig_optimizer_ablation(data):
    fig, ax = plt.subplots(figsize=(7, 4.4))
    colors = {"sgd": "#7f7f7f", "adam": "#1f77b4", "adamw": "#d62728"}
    steps = range(1, data["steps"] + 1)
    curves = {}
    for name, color in colors.items():
        seed_curves = [runs["curve"] for k, runs in data["runs"].items()
                       if k.startswith(name + "_seed")]
        arr = np.array(seed_curves)
        mean, std = arr.mean(0), arr.std(0)
        curves[name] = float(arr[:, -20:].mean())
        sm = _smooth(list(mean))
        ax.plot(steps, sm, color=color, label=name)
        ax.fill_between(steps, _smooth(list(mean - std)),
                        _smooth(list(mean + std)), color=color, alpha=0.18)
    # constant-LR adamw reference
    const = [runs["curve"] for k, runs in data["runs"].items()
             if k.startswith("adamw_constant_seed")]
    m = np.array(const).mean(0)
    ax.plot(steps, _smooth(list(m)), linestyle="--", color="#d62728",
            alpha=0.65, label="adamw (constant LR)")
    ax.set_xlabel("training step")
    ax.set_ylabel("training loss")
    ax.set_yscale("log")
    ax.set_title("Optimizer ablation on controlled next-token task "
                 "(mean ± std over 3 seeds)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "optimizer_ablation.png"), dpi=150)
    plt.close(fig)


def fig_grad_error_distribution(data):
    labels = [c["label"] for c in data["configs"]]
    medians = [c["rel_err_median"] for c in data["configs"]]
    p95s = [c["rel_err_p95"] for c in data["configs"]]
    maxes = [c["rel_err_max"] for c in data["configs"]]
    x = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - width, medians, width, label="median", color="#1f77b4")
    ax.bar(x, p95s, width, label="p95", color="#ff7f0e")
    ax.bar(x + width, maxes, width, label="max", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yscale("log")
    ax.set_xlabel("configuration")
    ax.set_ylabel("relative gradient error vs PyTorch")
    ax.set_title("Per-tensor gradient error distributions (all params)")
    ax.axhline(1e-6, color="gray", linestyle=":", linewidth=1)
    ax.text(len(labels) - 0.5, 1.05e-6, "1e-6", fontsize=8, color="gray")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "grad_error_distribution.png"), dpi=150)
    plt.close(fig)


def fig_stability(data):
    rows = data["rows"]
    scales = [r["logit_scale"] for r in rows]
    errs = [max(r["abs_error"], 1e-12) for r in rows]
    refs = [r["loss_float64_reference"] for r in rows]
    fig, ax1 = plt.subplots(figsize=(6.8, 4.2))
    ax1.loglog(scales, refs, marker="o", color="#1f77b4",
               label="cross-entropy loss (float64 ref)")
    ax1.set_xlabel("logit magnitude scale")
    ax1.set_ylabel("loss value")
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.loglog(scales, errs, marker="s", color="#d62728",
               label="float32 abs error")
    ax2.set_ylabel("absolute error vs float64")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper left")
    ax1.set_title("Cross entropy under extreme logits (stays finite & accurate)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES, "stability.png"), dpi=150)
    plt.close(fig)


def main():
    os.makedirs(FIGURES, exist_ok=True)
    specs = [
        ("latency_scaling.json", fig_latency),
        ("attention_scaling.json", fig_attention_scaling),
        ("kv_cache_ablation.json", fig_kv_cache),
        ("kv_cache_scaling.json", fig_kv_scaling),
        ("op_profile.json", fig_op_profile),
        ("training_curve.json", fig_training_curve),
        ("graph_memory.json", fig_graph_memory),
        ("optimizer_ablation.json", fig_optimizer_ablation),
        ("grad_error_distribution.json", fig_grad_error_distribution),
        ("stability.json", fig_stability),
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
