"""Training runtime: loop, evaluation, checkpointing, resume (README §10)."""

from __future__ import annotations

import json
import os
import pickle
import time

import numpy as np

from ..optim.optim import SGD, Adam, AdamW, clip_grad_norm, LambdaLR
from ..data.dataset import BatchSampler


def make_optimizer(name, params, lr, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def evaluate(model, val_rows, batch_size=8, max_batches=None):
    """Mean validation loss over packed rows."""
    model.eval()
    from ..autograd.engine import no_grad
    losses = []
    sampler = BatchSampler(len(val_rows), batch_size, seed=1234)
    batches = list(sampler)
    if max_batches:
        batches = batches[:max_batches]
    with no_grad():
        for idx in batches:
            x, y = x_y_from_rows(val_rows, idx)
            _, loss = model.forward(x, targets=y)
            losses.append(float(loss.item()))
    model.train()
    return float(np.mean(losses)) if losses else float("nan")


def x_y_from_rows(rows, indices):
    x = rows[indices]
    y = np.concatenate([x[:, 1:], x[:, :1]], axis=1)
    return x.astype(np.int64), y.astype(np.int64)


def save_checkpoint(path, model, optimizer, step, config_dict,
                    rng_state=None, metrics=None):
    """Checkpoint contains everything needed to resume reproducibly."""
    payload = {
        "format": "torchzero.checkpoint",
        "version": 1,
        "step": int(step),
        "config": config_dict,
        "model_state": {k: v for k, v in model.state_dict().items()},
        "optimizer_state": optimizer.state_dict() if optimizer is not None
        else None,
        "rng_state": rng_state
        if rng_state is not None
        else np.random.default_rng().bit_generator.state,
        "metrics": metrics or {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    return payload


def load_checkpoint(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("format") != "torchzero.checkpoint":
        raise ValueError(f"not a torchzero checkpoint: {path}")
    return payload


class Trainer:
    def __init__(self, model, train_rows, val_rows, cfg, out_dir,
                 log_fn=print):
        self.model = model
        self.train_rows = train_rows
        self.val_rows = val_rows
        self.cfg = cfg
        self.out_dir = out_dir
        self.log = log_fn

        self.optimizer = make_optimizer(cfg["optimizer"], model.parameters(),
                                        lr=float(cfg["lr"]),
                                        weight_decay=float(
                                            cfg.get("weight_decay", 0.01)))
        total_steps = int(cfg["max_steps"])
        warmup = int(cfg.get("warmup_steps", 20))
        self.scheduler = LambdaLR(
            self.optimizer,
            lambda s: warmup_cosine_ratio(s, warmup, total_steps,
                                          float(cfg.get("min_lr_ratio", 0.1))))
        self.sampler = BatchSampler(len(train_rows),
                                    int(cfg["batch_size"]),
                                    seed=int(cfg.get("seed", 0)))
        self.history = []

    def train_step(self, step):
        self.model.train()
        batch_indices = next(iter(self.sampler))
        x, y = x_y_from_rows(self.train_rows, batch_indices)

        t0 = time.perf_counter()
        _, loss = self.model.forward(x, targets=y)
        forward_ms = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        backward_ms = (time.perf_counter() - t0) * 1000

        grad_norm = clip_grad_norm(self.model.parameters(),
                                   float(self.cfg.get("grad_clip", 1.0)))

        t0 = time.perf_counter()
        self.optimizer.step()
        self.scheduler.step()
        opt_ms = (time.perf_counter() - t0) * 1000

        return {
            "step": step,
            "loss": float(loss.item()),
            "grad_norm": grad_norm,
            "lr": self.optimizer.lr,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": opt_ms,
            "tokens_per_s": x.size / ((forward_ms + backward_ms + opt_ms)
                                      / 1000.0),
        }

    def fit(self, resume_from=None):
        cfg = self.cfg
        start_step = 0
        os.makedirs(self.out_dir, exist_ok=True)

        if resume_from is not None:
            from .checkpoint import load_checkpoint_into
            state = load_checkpoint_into(self.model, self.optimizer,
                                         self.scheduler, resume_from)
            start_step = state["step"]
            self.history = state.get("metrics", {}).get("history", [])
            # restore deterministic batch order position
            self.sampler.epoch = int(
                state.get("metrics", {}).get("sampler_epoch",
                                             self.sampler.epoch))
            self.log(f"resumed from {resume_from} at step {start_step}")

        max_steps = int(cfg["max_steps"])
        eval_every = int(cfg.get("eval_every", 50))
        ckpt_every = int(cfg.get("ckpt_every", 100))
        log_every = int(cfg.get("log_every", 10))

        for step in range(start_step + 1, max_steps + 1):
            stats = self.train_step(step)
            self.history.append(stats)
            self.sampler.advance_epoch()

            if step % log_every == 0 or step == max_steps:
                self.log(
                    f"step {step}/{max_steps} "
                    f"loss {stats['loss']:.4f} "
                    f"ppl {np.exp(stats['loss']):.2f} "
                    f"gnorm {stats['grad_norm']:.3f} "
                    f"lr {stats['lr']:.2e} "
                    f"tok/s {stats['tokens_per_s']:.0f}")
            if step % eval_every == 0 or step == max_steps:
                vloss = evaluate(self.model, self.val_rows,
                                 int(cfg["batch_size"]))
                self.log(f"  eval: val_loss {vloss:.4f}")
                stats["val_loss"] = vloss
            if step % ckpt_every == 0 or step == max_steps:
                path = os.path.join(self.out_dir, "checkpoint.pkl")
                save_checkpoint(path, self.model, self.optimizer, step,
                                self.cfg,
                                metrics={"history": self.history,
                                         "sampler_epoch": self.sampler.epoch})
                self.log(f"  checkpoint saved -> {path}")

        final = os.path.join(self.out_dir, "final.pkl")
        save_checkpoint(final, self.model, self.optimizer, max_steps,
                        self.cfg, metrics={"history": self.history})
        with open(os.path.join(self.out_dir, "history.json"), "w") as f:
            json.dump(self.history, f, indent=1)
        return final


def warmup_cosine_ratio(step, warmup_steps, total_steps, min_ratio):
    import math
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_ratio + (1 - min_ratio) * 0.5 * (
        1 + math.cos(math.pi * min(1.0, progress)))
