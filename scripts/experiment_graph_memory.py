"""Research experiment: memory cost of the retained dynamic computation graph.

For a model holding its graph (after forward, before backward), we walk the
actual graph via topological order and measure:

  * number of graph nodes,
  * total bytes of intermediate activation tensors (parameters excluded),
  * bytes attributable to attention probability tensors,
  * process peak RSS before/after the forward.

Varies sequence length and depth. Raw -> experiments/results/graph_memory.json
"""

from __future__ import annotations

import gc
import json
import os
import resource
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero.transformer import Transformer, TransformerConfig
from torchzero.autograd.engine import topological_order


def measure(T, n_layers, dim=96, n_heads=4, vocab=512, batch=8):
    cfg = TransformerConfig(vocab_size=vocab, dim=dim, n_layers=n_layers,
                            n_heads=n_heads, block_size=max(T, 16))
    model = Transformer(cfg, seed=0)

    gc.collect()
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    ids = np.random.default_rng(T * 7 + n_layers).integers(
        0, vocab, (batch, T)).astype(np.int64)
    targets = np.roll(ids, -1, axis=1)

    _, loss = model.forward(ids, targets=targets)

    # walk the live dynamic graph: every node is an intermediate that must be
    # retained until backward completes
    order = topological_order(loss)
    from torchzero.nn.module import Parameter
    visited = set()
    act_bytes = 0
    node_count = 0
    param_bytes = 0
    for node in order:
        if id(node) in visited:
            continue
        visited.add(id(node))
        if isinstance(node, Parameter):
            param_bytes += node.data.nbytes
            continue
        node_count += 1
        act_bytes += node.data.nbytes

    attn_prob_bytes = batch * cfg.n_heads * T * T * 4 * n_layers

    del loss, order
    gc.collect()
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024

    return {
        "seq_len": T,
        "layers": n_layers,
        "batch": batch,
        "dim": dim,
        "graph_nodes": node_count,
        "activation_bytes": act_bytes,
        "parameter_bytes": param_bytes,
        "attention_prob_bytes_theory": attn_prob_bytes,
        "attn_share_of_activations": attn_prob_bytes / act_bytes,
        "peak_rss_bytes": rss_after,
        "rss_increased": rss_after > rss_before,
    }


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    rows = []
    for L in [2, 4, 8]:
        for T in [16, 32, 64, 128, 256]:
            row = measure(T, L)
            rows.append(row)
            print(f"L={L} T={T:>3}: nodes {row['graph_nodes']:>6} "
                  f"act {row['activation_bytes']/1e6:7.2f} MB "
                  f"(attn share {100*row['attn_share_of_activations']:.0f}%)")

    out = os.path.join(results_dir, "graph_memory.json")
    json.dump({"name": "graph_memory", "rows": rows}, open(out, "w"),
              indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
