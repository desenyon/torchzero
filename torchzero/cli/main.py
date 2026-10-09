"""torchzero command line interface.

Commands:
    torchzero train <config.yaml> [--resume]
    torchzero generate <checkpoint_dir> "<prompt>" [--tokens N]
                       [--temperature T] [--top-k K] [--seed S]
    torchzero inspect <checkpoint_dir> ["<prompt>"] [--html report.html]
    torchzero benchmark [--quick] [--out results.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys

import numpy as np
import yaml


def _load_config(path):
    with open(path, encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("config must be a YAML mapping")
    for key in ("model", "train"):
        if not isinstance(cfg.get(key), dict):
            raise ValueError(f"config requires a {key} mapping")
    cfg.setdefault("data", {})
    if not isinstance(cfg["data"], dict):
        raise ValueError("data must be a mapping")
    # Corpus and external tokenizer paths are relative to the YAML file.
    for key in ("text", "tokenizer_path"):
        value = cfg["data"].get(key)
        if value is not None:
            cfg["data"][key] = str((Path(path).resolve().parent / value).resolve())
    return cfg


def _corpus_text(cfg):
    from ..data.dataset import CharStreamDataset, load_text
    data = cfg.get("data", {})
    if data.get("text") is not None and data.get("inline_text") is not None:
        raise ValueError("choose data.text (file) or data.inline_text, not both")
    if data.get("text") is not None:
        return load_text(data["text"])
    text = data.get("inline_text", CharStreamDataset.TEXT)
    if not isinstance(text, str) or not text:
        raise ValueError("corpus must be non-empty UTF-8 text")
    return text


def _tokenizer_signature(tok):
    return hashlib.sha256(json.dumps(tok.merges, separators=(",", ":")).encode()).hexdigest()


def _build_tokenizer(cfg, out_dir, *, text=None, resume_from=None):
    """Resume uses the checkpoint's tokenizer without retraining/overwriting it."""
    from ..tokenizer import BPETokenizer
    if resume_from is not None:
        path = Path(resume_from).parent / "tokenizer.json"
        if not path.is_file():
            raise ValueError("resume requires tokenizer.json beside the checkpoint")
        return BPETokenizer.load(path)
    tok_path = cfg.get("data", {}).get("tokenizer_path")
    if tok_path is not None:
        tok = BPETokenizer.load(tok_path)
    else:
        vocab = cfg["model"].get("vocab_size", "auto")
        vocab_size = 288 if vocab in (None, "auto") else int(vocab)
        tok = BPETokenizer().train(_corpus_text(cfg) if text is None else text, vocab_size=vocab_size)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tok.save(os.path.join(out_dir, "tokenizer.json"))
    return tok


def cmd_train(args):
    from ..runtime.trainer import Trainer, evaluate
    from ..transformer.model import Transformer, TransformerConfig
    from ..data.dataset import pack_lm_sequences, train_val_split

    cfg = _load_config(args.config)
    out_dir = cfg.get("out_dir", "checkpoints/run")
    text = _corpus_text(cfg)
    tok = _build_tokenizer(cfg, None, text=text, resume_from=args.resume)
    ids = tok.encode(text)
    tr_ids, va_ids = train_val_split(ids, float(cfg["data"].get("val_fraction", .1)))
    ctx = int(cfg["model"].get("block_size", 64))
    train_rows = pack_lm_sequences(tr_ids, ctx)
    val_rows = pack_lm_sequences(va_ids, ctx)
    if not len(train_rows) or not len(val_rows):
        raise ValueError("training/validation data split produced no complete next-token rows; "
                         "increase corpus size, adjust val_fraction or reduce block_size")
    model_cfg = dict(cfg["model"])
    # The requested vocabulary is a BPE training ceiling. Use only actual
    # learned token IDs so generation can always decode the model's outputs.
    model_cfg["vocab_size"] = tok.vocab_size
    full_cfg = {**cfg["train"], "model": model_cfg,
                "tokenizer_signature": _tokenizer_signature(tok),
                "corpus_signature": hashlib.sha256(text.encode("utf-8")).hexdigest()}
    seed = int(full_cfg.get("seed", 0))
    model = Transformer(TransformerConfig.from_dict(model_cfg), seed=seed)
    trainer = Trainer(model, train_rows, val_rows, full_cfg, out_dir=out_dir)
    if args.resume is None:
        os.makedirs(out_dir, exist_ok=True)
        tok.save(os.path.join(out_dir, "tokenizer.json"))
    final = trainer.fit(resume_from=args.resume, stop_after=args.stop_after)
    # Only copy the resume tokenizer after state and data have been validated.
    tok.save(os.path.join(out_dir, "tokenizer.json"))
    vloss = evaluate(model, val_rows, full_cfg["batch_size"])
    print(f"training stopped at step {trainer.step}: {final}")
    print(f"final validation loss: {vloss:.4f} (perplexity {np.exp(vloss):.2f})")
    print(f"parameters: {model.num_parameters():,}")


def _load_model_and_tokenizer(ckpt_dir):
    from ..runtime.checkpoint import load_checkpoint, build_model_from_config
    from ..tokenizer import BPETokenizer

    # checkpoint file or directory both accepted
    path = ckpt_dir
    if os.path.isdir(ckpt_dir):
        for name in ("final.pkl", "checkpoint.pkl"):
            candidate = os.path.join(ckpt_dir, name)
            if os.path.exists(candidate):
                path = candidate
                break
    payload = load_checkpoint(path)
    config_dict = payload["config"]
    model = build_model_from_config(config_dict)
    model.load_state_dict(payload["model_state"])
    model.eval()

    tok = None
    tok_path = os.path.join(os.path.dirname(path) or ".", "tokenizer.json")
    if os.path.exists(tok_path):
        tok = BPETokenizer.load(tok_path)
        signature = config_dict.get("tokenizer_signature")
        if signature is not None and signature != _tokenizer_signature(tok):
            raise ValueError("checkpoint/tokenizer identity mismatch")
        if tok.vocab_size != model.config.vocab_size:
            raise ValueError("checkpoint/tokenizer vocabulary size mismatch")
    elif config_dict.get("tokenizer_signature") is not None:
        raise ValueError("checkpoint requires tokenizer.json beside it")
    return model, tok


def cmd_generate(args):
    model, tok = _load_model_and_tokenizer(args.checkpoint)
    prompt = args.prompt
    ids = tok.encode(prompt) if tok else [int(t) for t in prompt.split()]
    if not ids:
        ids = [0]
    out_ids = model.generate(ids, args.tokens,
                             temperature=args.temperature,
                             top_k=args.top_k,
                             seed=args.seed,
                             use_kv_cache=not args.no_cache)
    new_ids = out_ids[len(ids):]
    text = tok.decode(out_ids) if tok else str(new_ids)
    print(text)


def cmd_inspect(args):
    from debugger.trace import forward_trace, backward_trace, \
        parameter_summary
    from debugger.report import render_text, render_html

    model, tok = _load_model_and_tokenizer(args.checkpoint)
    prompt = args.prompt or "the model"
    ids = tok.encode(prompt) if tok else list(range(1, 9))

    ctx = model.config.block_size
    rng = np.random.default_rng(int(args.seed))
    if len(ids) < 4:
        ids = ids + rng.integers(0, model.config.vocab_size,
                                 8 - len(ids)).tolist()
    ids = ids[:ctx - 1]
    arr = np.array([ids], dtype=np.int64)
    targets = np.roll(arr, -1, axis=1)

    fwd = forward_trace(model, arr, targets)
    bwd = backward_trace(model, arr, targets)
    params = parameter_summary(model)

    if args.html:
        from debugger.report import save_html
        save_html(render_html(fwd, bwd, params), args.html)
        print(f"debugger report written to {args.html}")
    else:
        print(render_text(fwd, bwd, params))
        if sys.stdin.isatty():
            _interactive_loop(fwd, bwd)


def _interactive_loop(fwd, bwd):
    print("\nInteractive debugger. Commands: stages | attn | grads | graph | "
          "params | quit")
    while True:
        try:
            cmd = input("debug> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            break
        if cmd in ("q", "quit", "exit"):
            break
        if cmd == "stages":
            for r in fwd["records"]:
                shape = r.get("out_shape", r.get("attn_probs_shape"))
                extra = ""
                if "norm_mean" in r:
                    extra = (f"mean={r['norm_mean']:.4f} "
                             f"std={r['norm_std']:.4f}")
                print(f"  {r.get('op'):<20}{str(shape):<24}{extra}")
        elif cmd == "attn":
            for r in fwd["records"]:
                if r.get("op") == "attention":
                    p = r["attn_probs"]
                    row_mass = p.sum(-1)[0]
                    print(f"  probs {p.shape}, row sums (should be ~1): "
                          f"{np.round(row_mass, 3)}")
                    causal_ok = all(
                        abs(p[0, h].sum()) <= 1 + 1e-3 for h in range(p.shape[1]))
                    print(f"  finite: {np.isfinite(p).all()}, "
                          f"normalized rows: {causal_ok}")
        elif cmd == "grads":
            top = sorted([g for g in bwd["grad_norms"]
                          if g["grad_norm"] is not None],
                         key=lambda g: -g["grad_norm"])[:10]
            for g in top:
                print(f"  idx {g['index']:>3} shape {str(g['shape']):<16}"
                      f"|g|={g['grad_norm']:.5f} |w|={g['param_norm']:.4f}")
        elif cmd == "graph":
            for e in sorted(bwd["graph_nodes"], key=lambda e: -e["ms"]):
                print(f"  {e['op']:<16}{str(e['out_shape']):<22}"
                      f"{e['ms']:.3f}ms parents={e['parents']} "
                      f"rg={e['requires_grad']}")
        elif cmd == "params":
            print(f"  {bwd['num_parameters']} tensors, "
                  f"{bwd['parameter_count']:,} parameters")
        elif cmd:
            print(f"  unknown command: {cmd}")
    print("bye")


def cmd_benchmark(args):
    from benchmarks.run import main as bench_main
    bench_main(quick=args.quick, out=args.out)


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="torchzero",
        description="From-scratch neural network / transformer stack")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="train a model from a YAML config")
    p_train.add_argument("config")
    p_train.add_argument("--resume", default=None,
                         help="checkpoint path to resume from")
    p_train.add_argument("--stop-after", type=int, default=None,
                         help="pause at this absolute step, preserving the max_steps schedule")
    p_train.set_defaults(fn=cmd_train)

    p_gen = sub.add_parser("generate",
                           help="generate text from a checkpoint")
    p_gen.add_argument("checkpoint")
    p_gen.add_argument("prompt")
    p_gen.add_argument("--tokens", type=int, default=40)
    p_gen.add_argument("--temperature", type=float, default=0.7)
    p_gen.add_argument("--top-k", type=int, default=None)
    p_gen.add_argument("--seed", type=int, default=None)
    p_gen.add_argument("--no-cache", action="store_true",
                       help="disable KV cache during generation")
    p_gen.set_defaults(fn=cmd_generate)

    p_ins = sub.add_parser("inspect", help="open the execution debugger")
    p_ins.add_argument("checkpoint")
    p_ins.add_argument("prompt", nargs="?", default=None)
    p_ins.add_argument("--html", default=None,
                       help="write an HTML report to this path")
    p_ins.add_argument("--seed", type=int, default=0)
    p_ins.set_defaults(fn=cmd_inspect)

    p_bench = sub.add_parser("benchmark",
                             help="run performance/correctness benchmarks")
    p_bench.add_argument("--quick", action="store_true")
    p_bench.add_argument("--out", default="experiments/results/benchmark.json")
    p_bench.set_defaults(fn=cmd_benchmark)

    args = parser.parse_args(argv)
    try:
        return args.fn(args)
    except (ValueError, TypeError, OSError, KeyError, IndexError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    sys.exit(main())
