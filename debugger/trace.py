"""Execution debugger: real forward and backward traces from live runs.

Nothing here is fabricated: every record comes from instrumenting an actual
forward/backward pass of a TorchZero model.
"""

from __future__ import annotations

import time

import numpy as np

from torchzero.autograd.engine import topological_order


def _nbytes(a):
    return int(np.asarray(a).nbytes)


def forward_trace(model, ids, targets=None):
    """Run one batch through ``model`` and collect per-stage records."""
    records = []
    t0 = time.perf_counter()
    logits, loss = model.forward(ids, targets=targets, trace=records)
    wall_ms = (time.perf_counter() - t0) * 1000
    return {
        "mode": "forward",
        "input_ids": np.asarray(ids).tolist(),
        "records": records,
        "logits_shape": tuple(logits.shape),
        "loss": None if loss is None else float(loss.item()),
        "wall_ms": wall_ms,
        "memory_bytes": sum(_nbytes(r.get("embedding"))
                            + _nbytes(r.get("residual_stream", 0))
                            + _nbytes(r.get("attn_probs", 0))
                            + _nbytes(r.get("hidden", 0))
                            + _nbytes(r.get("logits", 0))
                            for r in records),
    }


def backward_trace(model, ids, targets):
    """Instrumented backward pass.

    Wraps every node's backward function with a timer (real execution times,
    not estimates), then runs the reverse pass and collects gradient norms
    for all parameters plus graph structure from the topological order.
    """
    logits, loss = model.forward(ids, targets=targets)
    order = topological_order(loss)

    timed = []
    for node in order:
        orig_fn = node._backward_fn
        if orig_fn is None:
            continue
        entry = {
            "op": getattr(node, "_name", None) or "composite",
            "out_shape": tuple(node.shape),
            "requires_grad": bool(node.requires_grad),
            "parents": len(node._parents),
            "ms": 0.0,
            "grad_norm": None,
        }

        def wrap(g, _orig=orig_fn, _entry=entry):
            t0 = time.perf_counter()
            out = _orig(g)
            _entry["ms"] = (time.perf_counter() - t0) * 1000
            return out

        node._backward_fn = wrap
        timed.append((node, entry))

    param_norms_before = {id(p): float(np.sqrt((p.data.astype(np.float64)
                                                ** 2).sum()))
                          for p in model.parameters()}

    t0 = time.perf_counter()
    loss.backward()
    wall_ms = (time.perf_counter() - t0) * 1000

    # gradient norms captured after the pass (leaves retain .grad)
    params = model.parameters()
    grad_norms = []
    for i, p in enumerate(params):
        gn = None if p.grad is None else float(
            np.sqrt((p.grad.astype(np.float64) ** 2).sum()))
        grad_norms.append({
            "index": i,
            "shape": tuple(p.shape),
            "grad_norm": gn,
            "param_norm": param_norms_before[id(p)],
        })

    total_grad_sq = sum(g["grad_norm"] ** 2 for g in grad_norms
                        if g["grad_norm"] is not None)
    return {
        "mode": "backward",
        "loss": float(loss.item()),
        "wall_ms": wall_ms,
        "graph_nodes": [e for _, e in timed],
        "backward_ms_by_node_sum": sum(e["ms"] for e, _ in
                                       [(e, n) for n, e in timed]),
        "grad_norms": grad_norms,
        "global_grad_norm": float(np.sqrt(total_grad_sq)),
        "num_parameters": len(params),
        "parameter_count": int(sum(p.data.size for p in params)),
    }


def parameter_summary(model):
    rows = []
    named = model.named_parameters()
    if not named:
        named = {f"param_{i}": p for i, p in enumerate(model.parameters())}
    for name, p in sorted(named.items()):
        rows.append({
            "name": name,
            "shape": tuple(p.shape),
            "count": int(p.data.size),
            "mean": float(p.data.mean()),
            "std": float(p.data.std()),
        })
    return rows
