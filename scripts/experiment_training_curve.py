"""Research experiment: end-to-end training run on the built-in deterministic
corpus. Records the loss/eval curve -> experiments/results/training_curve.json.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchzero.transformer import Transformer, TransformerConfig
from torchzero.tokenizer import BPETokenizer
from torchzero.data.dataset import CharStreamDataset, pack_lm_sequences, \
    train_val_split
from torchzero.runtime.trainer import Trainer


def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "experiments", "results")
    os.makedirs(results_dir, exist_ok=True)

    text = CharStreamDataset.TEXT * 6          # ~18k chars, still tiny
    tok = BPETokenizer().train(text, vocab_size=288)
    ids = tok.encode(text)
    tr_ids, va_ids = train_val_split(ids, 0.05)
    tr_rows = pack_lm_sequences(tr_ids, 64)
    va_rows = pack_lm_sequences(va_ids, 64)
    if not len(tr_rows) or not len(va_rows):
        raise ValueError("corpus splits are too small for the configured context")

    model_cfg = {"vocab_size": tok.vocab_size, "dim": 96, "n_layers": 3,
                 "n_heads": 4, "block_size": 64}
    cfg = {
        "optimizer": "adamw", "lr": 3e-3, "weight_decay": 0.01,
        "grad_clip": 1.0, "batch_size": 16, "max_steps": 400,
        "eval_every": 50, "ckpt_every": 100000, "log_every": 100,
        "seed": 0, "warmup_steps": 40, "min_lr_ratio": 0.1,
        "model": model_cfg,
    }
    model = Transformer(TransformerConfig.from_dict(model_cfg), seed=0)
    trainer = Trainer(model, tr_rows, va_rows, cfg,
                      out_dir=os.path.join(results_dir, "_tmp_train"),
                      log_fn=print)
    trainer.fit()
    history = []
    for h in trainer.history:
        row = {"step": h["step"], "loss": h["loss"], "lr": h["lr"]}
        if "val_loss" in h:
            row["val_loss"] = h["val_loss"]
        history.append(row)
    out = os.path.join(results_dir, "training_curve.json")
    json.dump({
        "name": "training_curve",
        "model": model_cfg,
        "vocab_size": tok.vocab_size,
        "train_rows": len(tr_rows),
        "final_val_loss": history[-1].get("val_loss"),
        "rows": history,
    }, open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
