"""Render debugger traces as text or a self-contained HTML report."""

from __future__ import annotations

import html
import json


def _fmt_bytes(n):
    if n is None:
        return "-"
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}TB"


def render_text(fwd, bwd=None, param_rows=None):
    lines = []
    lines.append("=" * 72)
    lines.append("TorchZero execution trace (real data from live run)")
    lines.append("=" * 72)
    lines.append(f"input tokens : {fwd['input_ids']}")
    lines.append(f"loss         : {fwd['loss']}")
    lines.append(f"logits shape : {fwd['logits_shape']}")
    lines.append(f"forward wall : {fwd['wall_ms']:.2f} ms")
    lines.append(f"act memory   : ~{_fmt_bytes(fwd['memory_bytes'])}")
    lines.append("")
    lines.append(f"{'stage':<18}{'shape':<22}{'note':<30}")
    lines.append("-" * 72)
    for r in fwd["records"]:
        op = r.get("op", "?")
        shape = str(r.get("out_shape", r.get("attn_probs_shape", "-")))
        note = ""
        if "norm_mean" in r:
            note = f"mean={r['norm_mean']:.3f} std={r['norm_std']:.3f}"
        elif r.get("op") == "cross_entropy":
            note = f"loss={r.get('loss'):.4f}"
        lines.append(f"{op:<18}{shape:<22}{note:<30}")
    if bwd:
        lines.append("")
        lines.append("backward graph (timed during real reverse pass)")
        lines.append("-" * 72)
        lines.append(f"loss {bwd['loss']:.4f}  "
                     f"global grad norm {bwd['global_grad_norm']:.4f}  "
                     f"backward wall {bwd['wall_ms']:.2f} ms")
        slowest = sorted(bwd["graph_nodes"],
                         key=lambda e: -e["ms"])[:10]
        lines.append(f"{'op':<14}{'out shape':<22}{'ms':>8}  {'parents':>7}")
        for e in slowest:
            lines.append(f"{e['op']:<14}{str(e['out_shape']):<22}"
                         f"{e['ms']:>8.3f}  {e['parents']:>7}")
        lines.append("")
        lines.append("top gradient norms")
        top = sorted([g for g in bwd["grad_norms"]
                      if g["grad_norm"] is not None],
                     key=lambda g: -g["grad_norm"])[:8]
        for g in top:
            lines.append(f"  shape {str(g['shape']):<16} "
                         f"|g|={g['grad_norm']:.5f} |w|={g['param_norm']:.4f}")
    if param_rows:
        lines.append("")
        lines.append("parameters")
        total = sum(r["count"] for r in param_rows)
        lines.append(f"  {len(param_rows)} tensors, "
                     f"{total:,} parameters")
        for r in param_rows[:12]:
            lines.append(f"  {r['name']:<28}{str(r['shape']):<16}"
                         f"mean={r['mean']:+.4f} std={r['std']:.4f}")
        if len(param_rows) > 12:
            lines.append(f"  ... and {len(param_rows) - 12} more")
    return "\n".join(lines)


def render_html(fwd, bwd=None, param_rows=None):
    """Self-contained HTML page; all numbers come from the trace objects."""

    def table(rows, headers):
        out = "<table><thead><tr>"
        out += "".join(f"<th>{html.escape(str(h))}</th>" for h in headers)
        out += "</tr></thead><tbody>"
        for row in rows:
            out += "<tr>" + "".join(
                f"<td>{html.escape(str(c))}</td>" for c in row) + "</tr>"
        return out + "</tbody></table>"

    fwd_rows = []
    for r in fwd["records"]:
        fwd_rows.append([
            r.get("op"), r.get("out_shape",
                               r.get("attn_probs_shape", "-")),
            f"{r['norm_mean']:.4f}" if "norm_mean" in r else "",
            f"{r['norm_std']:.4f}" if "norm_std" in r else "",
            f"{r['loss']:.4f}" if r.get("op") == "cross_entropy" else "",
        ])
    bwd_rows = []
    if bwd:
        bwd_rows = [[e["op"], e["out_shape"], f"{e['ms']:.3f}",
                     e["parents"], e["requires_grad"]]
                    for e in bwd["graph_nodes"]]
    grad_rows = []
    if bwd:
        grad_rows = [[g["index"], g["shape"],
                      f"{g['grad_norm']:.6f}", f"{g['param_norm']:.4f}"]
                     for g in bwd["grad_norms"]]
    par_rows = [[p["name"], p["shape"], p["count"],
                 f"{p['mean']:+.4f}", f"{p['std']:.4f}"]
                for p in (param_rows or [])]

    parts = ["""<!doctype html><html><head><meta charset="utf-8">
<title>TorchZero Debugger</title><style>
body{font-family:-apple-system,Segoe UI,sans-serif;margin:24px;background:#0d1117;color:#e6edf3}
h1,h2{color:#79c0ff}table{border-collapse:collapse;margin:12px 0;font-size:13px}
td,th{border:1px solid #30363d;padding:4px 10px;text-align:left}
th{background:#161b22}.num{text-align:right}
.kpi{display:inline-block;background:#161b22;border:1px solid #30363d;
border-radius:8px;padding:10px 16px;margin:4px}
.kpi b{color:#79c0ff;font-size:18px}
</style></head><body>"""]
    parts.append("<h1>TorchZero Execution Debugger</h1>")
    parts.append("<div>")
    loss_str = "-" if fwd["loss"] is None else f"{fwd['loss']:.4f}"
    parts.append(f'<span class="kpi">loss<br><b>{loss_str}</b></span>')
    parts.append(f'<span class="kpi">forward<br>'
                 f'<b>{fwd["wall_ms"]:.1f} ms</b></span>')
    parts.append(f'<span class="kpi">activations<br>'
                 f'<b>{_fmt_bytes(fwd["memory_bytes"])}</b></span>')
    if bwd:
        parts.append(f'<span class="kpi">global |grad|<br>'
                     f'<b>{bwd["global_grad_norm"]:.4f}</b></span>')
        parts.append(f'<span class="kpi">params<br>'
                     f'<b>{bwd["parameter_count"]:,}</b></span>')
    parts.append("</div>")
    parts.append(f"<p>input tokens: <code>{fwd['input_ids']}</code></p>")
    parts.append("<h2>Forward stages</h2>")
    parts.append(table(fwd_rows,
                       ["op", "output shape", "mean", "std", "loss"]))
    if bwd:
        parts.append("<h2>Backward graph</h2>")
        parts.append(table(bwd_rows,
                           ["op", "output shape", "ms", "parents",
                            "requires_grad"]))
        parts.append("<h2>Gradient norms</h2>")
        parts.append(table(grad_rows,
                           ["param idx", "shape", "|grad|", "|weight|"]))
    if par_rows:
        parts.append("<h2>Parameters</h2>")
        parts.append(table(par_rows,
                           ["name", "shape", "count", "mean", "std"]))
    parts.append("</body></html>")
    return "\n".join(parts)


def save_html(content, path):
    with open(path, "w") as f:
        f.write(content)
