"""Benchmark suite (README §16).

Compares TorchZero against PyTorch (reference framework used ONLY here for
measurement -- it never powers TorchZero runtime execution) on identical
small decoder-only transformers:

  * forward / backward / training-step latency
  * tokens per second
  * numerical output error + gradient error (parity)
  * checkpoint size
  * KV-cache on/off generation throughput (ablation)
  * scaling across sequence lengths, batch sizes, dims, layers, heads

Raw results are written as JSON under experiments/results/.
If torch is not installed, parity/latency comparisons are skipped and only
TorchZero-side measurements are recorded (documented in the report).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time

import numpy as np

from torchzero.transformer import Transformer, TransformerConfig


def _try_import_torch():
    try:
        import torch  # noqa: F401  (reference framework, tests only)
        import torch.nn as nn
        return torch, nn
    except ImportError:
        return None, None


def timed(fn, repeats=5, warmup=2):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.median(times))


# --------------------------------------------------------------------------
# Reference torch model mirroring the TorchZero architecture exactly
# --------------------------------------------------------------------------

def build_torch_reference(torch, nn, cfg: TransformerConfig, tz_model):
    """Mirror Transformer with identical weights."""
    class RefAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.wq = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.wk = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.wv = nn.Linear(cfg.dim, cfg.dim, bias=False)
            self.proj = nn.Linear(cfg.dim, cfg.dim, bias=False)

        def forward(self, x, cos, sin):
            B, T, C = x.shape
            H, Dh = cfg.n_heads, cfg.head_dim
            q = self.wq(x).view(B, T, H, Dh)
            k = self.wk(x).view(B, T, H, Dh)
            v = self.wv(x).view(B, T, H, Dh)

            def rope(t):
                half = Dh // 2
                c, s = cos[..., :half], sin[..., :half]
                t1, t2 = t[..., :half], t[..., half:]
                return torch.cat([t1 * c - t2 * s,
                                  t1 * s + t2 * c], dim=-1)
            q, k = rope(q), rope(k)
            q, k, v = (t.transpose(1, 2) for t in (q, k, v))
            att = (q @ k.transpose(-2, -1)) / np.sqrt(Dh)
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            att = att.masked_fill(~mask, -1e30)
            att = torch.softmax(att, dim=-1)
            out = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
            return self.proj(out)

    class RefBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm_attn = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
            self.attn = RefAttention()
            self.norm_mlp = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)
            self.fc = nn.Linear(cfg.dim, cfg.ffn_hidden)
            self.proj = nn.Linear(cfg.ffn_hidden, cfg.dim)

        def forward(self, x, cos, sin):
            beta = np.sqrt(2.0 / np.pi)
            kappa = 0.044715
            x = x + self.attn(self.norm_attn(x), cos, sin)
            h = self.fc(self.norm_mlp(x))
            h = 0.5 * h * (1 + torch.tanh(beta * h + beta * kappa * h ** 3))
            x = x + self.proj(h)
            return x

    class RefModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.dim)
            self.blocks = nn.ModuleList([RefBlock() for _ in range(
                cfg.n_layers)])
            self.norm_f = nn.RMSNorm(cfg.dim, eps=cfg.norm_eps)

        def forward(self, idx):
            from torch.nn.functional import cross_entropy
            cos, sin = tz_rope_tables(idx.shape[1], cfg.head_dim)
            x = self.tok_emb(idx)
            for blk in self.blocks:
                x = blk(x, cos, sin)
            x = self.norm_f(x)
            logits = x @ self.tok_emb.weight.T
            return logits

    def tz_rope_tables(T, head_dim):
        from torchzero.transformer.model import rope_cache
        c, s = rope_cache(T, head_dim)  # full width (T, head_dim)
        cc = torch.tensor(c, dtype=torch.float32)
        ss = torch.tensor(s, dtype=torch.float32)
        # broadcast over batch and heads: (1, T, 1, head_dim)
        return cc.unsqueeze(0).unsqueeze(2), ss.unsqueeze(0).unsqueeze(2)

    ref = RefModel()
    # transfer weights. TorchZero Linear stores (in, out); torch nn.Linear
    # stores (out, in), so every Linear weight is transposed on transfer.
    def lin(arr):
        return torch.from_numpy(
            np.ascontiguousarray(arr.astype(np.float32).T))
    sd = {}
    sd["tok_emb.weight"] = torch.from_numpy(
        tz_model.tok_emb.weight.data.astype(np.float32))
    for i, blk in enumerate(tz_model.blocks):
        p = f"blocks.{i}."
        sd[p + "norm_attn.weight"] = torch.from_numpy(
            blk.norm_attn.weight.data.astype(np.float32))
        sd[p + "norm_mlp.weight"] = torch.from_numpy(
            blk.norm_mlp.weight.data.astype(np.float32))
        sd[p + "attn.wq.weight"] = lin(blk.attn.wq.weight.data)
        sd[p + "attn.wk.weight"] = lin(blk.attn.wk.weight.data)
        sd[p + "attn.wv.weight"] = lin(blk.attn.wv.weight.data)
        sd[p + "attn.proj.weight"] = lin(blk.attn.proj.weight.data)
        sd[p + "fc.weight"] = lin(blk.mlp.fc.weight.data)
        sd[p + "fc.bias"] = torch.from_numpy(
            blk.mlp.fc.bias.data.astype(np.float32))
        sd[p + "proj.weight"] = lin(blk.mlp.proj.weight.data)
        sd[p + "proj.bias"] = torch.from_numpy(
            blk.mlp.proj.bias.data.astype(np.float32))
    sd["norm_f.weight"] = torch.from_numpy(
        tz_model.norm_f.weight.data.astype(np.float32))
    ref.load_state_dict(sd)
    return ref


def pair_models(seed, **cfg_kwargs):
    """Build TorchZero model and its torch twin with identical weights."""
    cfg = TransformerConfig(**cfg_kwargs)
    tz_model = Transformer(cfg, seed=seed)
    tz_model.eval()
    torch, nn = _try_import_torch()
    if torch is None:
        return tz_model, None, None, cfg
    torch.manual_seed(seed)
    ref = build_torch_reference(torch, nn, cfg, tz_model)
    return tz_model, ref, torch, cfg


def parity_experiment(results_dir):
    """Numerical output error + gradient error vs torch."""
    torch, _ = _try_import_torch()
    entry = {"name": "parity_vs_torch", "torch_available": torch is not None,
             "rows": []}
    if torch is None:
        results_dir_file = os.path.join(results_dir, "parity_vs_torch.json")
        json.dump(entry, open(results_dir_file, "w"), indent=1)
        return entry

    rng = np.random.default_rng(0)
    cases = [
        dict(vocab_size=64, dim=32, n_layers=2, n_heads=4, block_size=16),
        dict(vocab_size=128, dim=48, n_layers=3, n_heads=4, block_size=32),
        dict(vocab_size=256, dim=64, n_layers=2, n_heads=8, block_size=48),
    ]
    for ci, kw in enumerate(cases):
        tz_model, ref, _, cfg = pair_models(seed=ci, **kw)
        B = 2
        ids = rng.integers(0, cfg.vocab_size, (B, cfg.block_size)) \
            .astype(np.int64)
        targets = rng.integers(0, cfg.vocab_size, (B, cfg.block_size)) \
            .astype(np.int64)

        # TorchZero forward + backward
        logits_tz, loss_tz = tz_model.forward(ids, targets=targets)
        loss_tz.backward()
        tz_grads = {n: p.grad.copy() for n, p in
                    tz_model.named_parameters().items()}
        gnorm_tz = float(np.sqrt(sum((g ** 2).sum() for g in tz_grads.values())))

        # torch side
        tt_ids = torch.tensor(ids)
        tt_targets = torch.tensor(targets.reshape(-1))
        ref.zero_grad()
        logits_pt = ref(tt_ids)
        loss_pt = torch.nn.functional.cross_entropy(
            logits_pt.reshape(-1, cfg.vocab_size).float(),
            tt_targets)
        loss_pt.backward()

        fwd_err = float(np.abs(logits_tz.data
                               - logits_pt.detach().numpy()).max())
        rel_fwd = fwd_err / float(np.abs(logits_pt.detach().numpy()).max())
        loss_err = abs(float(loss_tz.item()) - float(loss_pt.item()))

        # gradient errors on matched names.
        # value is (ref_name, needs_transpose): torch nn.Linear stores
        # (out, in); TorchZero stores (in, out).
        grad_errs = []
        mapping = {
            "tok_emb.weight": ("tok_emb.weight", False),
            "norm_f.weight": ("norm_f.weight", False),
        }
        for i in range(cfg.n_layers):
            p = f"blocks.{i}."
            mapping[f"{p}norm_attn.weight"] = (f"{p}norm_attn.weight", False)
            mapping[f"{p}norm_mlp.weight"] = (f"{p}norm_mlp.weight", False)
            for sub in ["wq", "wk", "wv", "proj"]:
                mapping[f"{p}attn.{sub}.weight"] = \
                    (f"{p}attn.{sub}.weight", True)
            mapping[f"{p}mlp.fc.weight"] = (f"{p}fc.weight", True)
            mapping[f"{p}mlp.fc.bias"] = (f"{p}fc.bias", False)
            mapping[f"{p}mlp.proj.weight"] = (f"{p}proj.weight", True)
            mapping[f"{p}mlp.proj.bias"] = (f"{p}proj.bias", False)

        for tz_name, (pt_name, do_transpose) in mapping.items():
            if tz_name not in tz_grads:
                continue
            pt_grad = get_ref_grad(ref, pt_name)
            if pt_grad is None:
                continue
            if do_transpose:
                pt_grad = pt_grad.T
            if tz_grads[tz_name].shape != pt_grad.shape:
                continue
            denom = max(float(np.abs(pt_grad).max()), 1e-8)
            grad_errs.append(float(np.abs(tz_grads[tz_name] - pt_grad).max())
                             / denom)
        gnorm_pt = float(torch.sqrt(sum(
            (p.grad ** 2).sum() for p in ref.parameters())).item())

        row = {
            "case": ci, "config": kw,
            "logits_max_abs_error": fwd_err,
            "logits_rel_error": rel_fwd,
            "loss_abs_error": loss_err,
            "grad_rel_max_error": float(max(grad_errs)),
            "grad_norm_torchzero": gnorm_tz,
            "grad_norm_torch": gnorm_pt,
        }
        print(f"[parity] case {ci}: fwd_err={fwd_err:.2e} "
              f"rel={rel_fwd:.2e} loss_err={loss_err:.2e} "
              f"grad_rel={row['grad_rel_max_error']:.2e}")
        entry["rows"].append(row)

    out = os.path.join(results_dir, "parity_vs_torch.json")
    json.dump(entry, open(out, "w"), indent=1)
    return entry


def get_ref_grad(ref, name):
    parts = name.split(".")
    obj = ref
    for p in parts:
        if p.isdigit():
            obj = obj[int(p)]
        else:
            obj = getattr(obj, p)
    return None if obj.grad is None else obj.grad.numpy()


def latency_experiment(results_dir, quick=False):
    """Forward/backward/train-step latency + tokens/s for both frameworks."""
    torch, _ = _try_import_torch()
    configs = [
        # (batch, seq, dim, layers, heads)
        (4, 32, 64, 2, 4),
        (8, 64, 128, 4, 4),
        (16, 64, 128, 4, 8),
        (8, 128, 128, 4, 4),
        (16, 128, 256, 6, 8),
    ]
    if quick:
        configs = configs[:3]
    rows = []
    for bi, (B, T, D, L, H) in enumerate(configs):
        vocab = 512
        kw = dict(vocab_size=vocab, dim=D, n_layers=L, n_heads=H,
                  block_size=max(T, 128))
        tz_model, ref, _, cfg = pair_models(seed=bi, **kw)
        rng = np.random.default_rng(bi)
        ids = rng.integers(0, vocab, (B, T)).astype(np.int64)

        def step_tz():
            logits, loss = tz_model.forward(ids, targets=np.roll(ids, -1, 1))
            loss.backward(retain_graph=False)
            # free grads
            for p in tz_model.parameters():
                p.grad = None
        f_ms = timed(lambda: tz_model.forward(ids)[0].data, repeats=3 if quick else 5)
        b_ms = timed(lambda: _backward_only(tz_model, ids), repeats=3 if quick else 5)
        step_ms = timed(step_tz, repeats=3 if quick else 5)
        tokens_per_s = B * T / (step_ms / 1000.0)

        row = {
            "batch": B, "seq": T, "dim": D, "layers": L, "heads": H,
            "params": tz_model.num_parameters(),
            "tz_forward_ms": f_ms * 1000,
            "tz_backward_ms": b_ms * 1000,
            "tz_step_ms": step_ms * 1000,
            "tz_tokens_per_s": tokens_per_s,
        }

        if torch is not None:
            tt_ids = torch.tensor(ids)

            def step_pt():
                ref.zero_grad()
                logits = ref(tt_ids)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, vocab).float(),
                    torch.roll(tt_ids, -1, 1).reshape(-1))
                loss.backward()
            pf = timed(lambda: ref(tt_ids), repeats=3 if quick else 5)
            pb = timed(lambda: _pt_backward(ref, tt_ids, vocab),
                       repeats=3 if quick else 5)
            ps = timed(step_pt, repeats=3 if quick else 5)
            row["torch_forward_ms"] = pf * 1000
            row["torch_backward_ms"] = pb * 1000
            row["torch_step_ms"] = ps * 1000
            row["torch_tokens_per_s"] = B * T / ps
        rows.append(row)
        print(f"[latency] B{B} T{T} D{D} L{L} H{H}: "
              f"tz_step={row['tz_step_ms']:.1f}ms"
              + (f" torch_step={row['torch_step_ms']:.1f}ms"
                 if torch is not None else ""))

    entry = {"name": "latency_scaling", "rows": rows}
    json.dump(entry, open(os.path.join(results_dir,
                                       "latency_scaling.json"), "w"), indent=1)
    return entry


def _backward_only(model, ids):
    _, loss = model.forward(ids, targets=np.roll(ids, -1, axis=1))
    loss.backward()
    for p in model.parameters():
        p.grad = None


def _pt_backward(ref, tt_ids, vocab):
    import torch
    ref.zero_grad()
    logits = ref(tt_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab).float(),
        torch.roll(tt_ids, -1, 1).reshape(-1))
    loss.backward()


def kv_cache_ablation(results_dir):
    """Generation throughput with and without the KV cache."""
    rows = []
    for gi, (n_prompt, n_new) in enumerate([(8, 24), (16, 48)]):
        kw = dict(vocab_size=256, dim=96, n_layers=4, n_heads=4,
                  block_size=128)
        tz_model, _, _, cfg = pair_models(seed=42 + gi, **kw)
        rng = np.random.default_rng(gi)
        prompt = rng.integers(0, 256, (1, n_prompt)).astype(np.int64)

        t0 = time.perf_counter()
        tz_model.generate(list(prompt[0]), n_new, temperature=0.9,
                          use_kv_cache=True, seed=gi)
        cached_s = time.perf_counter() - t0

        t0 = time.perf_counter()
        tz_model.generate(list(prompt[0]), n_new, temperature=0.9,
                          use_kv_cache=False, seed=gi)
        uncached_s = time.perf_counter() - t0

        # determinism check at temperature 0 (greedy must match exactly)
        g_cached = tz_model.generate(list(prompt[0]), n_new, temperature=0.0,
                                     use_kv_cache=True)
        g_uncached = tz_model.generate(list(prompt[0]), n_new,
                                       temperature=0.0, use_kv_cache=False)
        greedy_match = bool(g_cached[:len(g_uncached)] == g_uncached[:len(g_cached)])

        rows.append({
            "prompt_len": n_prompt, "new_tokens": n_new,
            "cached_seconds": cached_s,
            "uncached_seconds": uncached_s,
            "speedup": uncached_s / cached_s,
            "cached_tokens_per_s": n_new / cached_s,
            "uncached_tokens_per_s": n_new / uncached_s,
            "greedy_outputs_match": greedy_match,
        })
        print(f"[kv] prompt {n_prompt} new {n_new}: "
              f"speedup {uncached_s / cached_s:.2f}x "
              f"greedy_match={greedy_match}")

    entry = {"name": "kv_cache_ablation", "rows": rows}
    json.dump(entry, open(os.path.join(results_dir,
                                       "kv_cache_ablation.json"), "w"),
              indent=1)
    return entry


def op_profile_experiment(results_dir):
    """Where does time go inside one training step? Instruments real
    executions by wrapping modules with wall-clock accumulators."""
    tz_model, _, _, cfg = pair_models(
        seed=7, vocab_size=512, dim=128, n_layers=4, n_heads=4,
        block_size=64)
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 512, (8, 64)).astype(np.int64)

    acc = {}
    originals = []

    def wrap_module(module, label):
        orig = module.forward

        def wrapped(*args, **kwargs):
            t0 = time.perf_counter()
            out = orig(*args, **kwargs)
            acc[label] = acc.get(label, 0.0) + time.perf_counter() - t0
            return out

        originals.append((module, orig))
        module.forward = wrapped

    for blk in tz_model.blocks:
        wrap_module(blk.attn, "attention")
        wrap_module(blk.mlp, "mlp")
        wrap_module(blk.norm_attn, "rmsnorm")
        wrap_module(blk.norm_mlp, "rmsnorm")

    from torchzero.runtime.trainer import clip_grad_norm
    from torchzero.optim import AdamW
    opt = AdamW(tz_model.parameters(), lr=1e-3)

    def step():
        acc.clear()
        t0 = time.perf_counter()
        _, loss = tz_model.forward(ids, targets=np.roll(ids, -1, axis=1))
        fwd_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        loss.backward()
        bwd_ms = (time.perf_counter() - t0) * 1000
        # attention fraction of the step (forward+backward inside wrappers is
        # included via autograd executing the same closures? No -- backward
        # does not call module.forward). Measure forward-side breakdown.
        opt.zero_grad()
        clip_grad_norm(tz_model.parameters(), 1.0)
        opt.step()
        return fwd_ms, bwd_ms, dict(acc)

    repeats = 5
    fwds, bwds = [], []
    merged = {}
    for _ in range(repeats):
        f, b, a = step()
        fwds.append(f)
        bwds.append(b)
        for k, v in a.items():
            merged[k] = merged.get(k, []) + [v]

    import statistics
    profile = {k: float(statistics.median(v)) for k, v in merged.items()}
    entry = {
        "name": "op_profile",
        "config": {"batch": 8, "seq": 64, "dim": 128, "layers": 4},
        "forward_wall_ms_median": float(statistics.median(fwds)),
        "backward_wall_ms_median": float(statistics.median(bwds)),
        "forward_stage_ms": profile,
        "note": ("backward time is dominated by the same primitives "
                 "(matmul, softmax, elementwise) executed on saved "
                 "activations; module wrappers only capture the forward "
                 "pass, so stage numbers refer to forward-side cost."),
    }
    json.dump(entry, open(os.path.join(results_dir, "op_profile.json"),
                          "w"), indent=1)
    print(f"[profile] fwd {entry['forward_wall_ms_median']:.1f}ms "
          f"bwd {entry['backward_wall_ms_median']:.1f}ms stages={profile}")
    return entry


def checkpoint_size_experiment(results_dir):
    """Checkpoint size for equal architectures."""
    import pickle
    from torchzero.runtime.trainer import save_checkpoint
    rows = []
    tmp = os.path.join(results_dir, "_tmp_ckpt")
    for dims in [(64, 2), (128, 4)]:
        d, l = dims
        tz_model, _, _, cfg = pair_models(seed=0, vocab_size=512, dim=d,
                                          n_layers=l, n_heads=4,
                                          block_size=64)
        path = os.path.join(tmp, f"m{d}.pkl")
        save_checkpoint(path, tz_model, None, step=0,
                        config_dict={"model": vars(cfg)})
        size = os.path.getsize(path)
        params = tz_model.num_parameters()
        rows.append({"dim": d, "layers": l, "params": params,
                     "bytes": size, "bytes_per_param": size / params})
        print(f"[ckpt] dim{d} L{l}: {size/1024:.1f}KB "
              f"({size/params:.1f} B/param)")
    entry = {"name": "checkpoint_size", "rows": rows}
    json.dump(entry, open(os.path.join(results_dir, "checkpoint_size.json"),
                          "w"), indent=1)
    return entry


def peak_memory_note():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def main(quick=False, out=None):
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    summary = {
        "platform": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
        },
        "peak_rss_bytes_process_level": peak_memory_note(),
    }
    summary.update({
        "parity": parity_experiment(results_dir)["rows"],
        "latency": latency_experiment(results_dir, quick=quick)["rows"],
        "kv_cache": kv_cache_ablation(results_dir)["rows"],
        "op_profile": op_profile_experiment(results_dir),
        "checkpoint_size": checkpoint_size_experiment(results_dir)["rows"],
    })
    out_path = out or os.path.join(results_dir, "benchmark.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"benchmark complete -> {out_path}")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    main(quick=args.quick, out=args.out)
