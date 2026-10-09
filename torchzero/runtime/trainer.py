"""Deterministic training, evaluation and complete training-state ownership."""

from __future__ import annotations

import hashlib
import json
import os
import time

import numpy as np

from ..autograd.engine import no_grad
from ..optim.optim import SGD, Adam, AdamW, clip_grad_norm, LambdaLR, warmup_cosine
from ..data.dataset import BatchSampler, get_batch
# Keep the original imports from trainer.py available to callers.
from .checkpoint import save_checkpoint, load_checkpoint, load_checkpoint_into


def make_optimizer(name, params, lr, weight_decay=0.0):
    name = name.lower()
    if name == "sgd":
        return SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer: {name}")


def evaluate(model, val_rows, batch_size=8, max_batches=None):
    """Token-weighted validation loss, preserving every module's prior mode."""
    if max_batches is not None and max_batches <= 0:
        raise ValueError("max_batches must be positive")
    sampler = BatchSampler(len(val_rows), batch_size, seed=1234)
    total_loss, tokens = 0., 0
    with model.evaluating(), no_grad():
        for batch, idx in enumerate(sampler):
            if max_batches is not None and batch >= max_batches:
                break
            x, y = get_batch(val_rows, idx)
            _, loss = model.forward(x, targets=y)
            total_loss += float(loss.item()) * y.size
            tokens += y.size
    return total_loss / tokens


def x_y_from_rows(rows, indices):
    """Compatibility alias for the shared next-token extraction routine."""
    return get_batch(rows, indices)


def _data_signature(train_rows, val_rows):
    digest = hashlib.sha256()
    for rows in (train_rows, val_rows):
        digest.update(str((rows.shape, rows.dtype.str)).encode())
        digest.update(np.ascontiguousarray(rows).tobytes())
    return digest.hexdigest()


class Trainer:
    def __init__(self, model, train_rows, val_rows, cfg, out_dir, log_fn=print):
        self.model = model
        self.train_rows = np.asarray(train_rows).copy()
        self.val_rows = np.asarray(val_rows).copy()
        self.cfg = dict(cfg)
        self.cfg["model"] = model.config.to_dict()
        cfg = self.cfg
        self.out_dir = out_dir
        self.log = log_fn
        for name, rows in (("training", self.train_rows), ("validation", self.val_rows)):
            if (rows.ndim != 2 or len(rows) == 0 or rows.dtype.kind not in "iu"
                    or not 2 <= rows.shape[1] <= model.config.block_size + 1):
                raise ValueError(f"{name} data must contain integer rows of 2..block_size+1 tokens")
            if rows.min() < 0 or rows.max() >= model.config.vocab_size:
                raise ValueError(f"{name} data contains token IDs outside the vocabulary")
        for key, default in (("max_steps", None), ("batch_size", None),
                             ("eval_every", 50), ("ckpt_every", 100), ("log_every", 10)):
            value = cfg.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{key} must be a positive integer")
        if not np.isfinite(float(cfg["lr"])) or float(cfg["lr"]) <= 0:
            raise ValueError("lr must be finite and positive")
        if int(cfg.get("warmup_steps", 20)) < 0:
            raise ValueError("warmup_steps cannot be negative")
        if not 0 <= float(cfg.get("min_lr_ratio", .1)) <= 1:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if not np.isfinite(float(cfg.get("grad_clip", 1.))) or float(cfg.get("grad_clip", 1.)) <= 0:
            raise ValueError("grad_clip must be finite and positive")
        self.optimizer = make_optimizer(cfg["optimizer"], model.parameters(),
                                        lr=float(cfg["lr"]),
                                        weight_decay=float(cfg.get("weight_decay", .01)))
        total_steps, warmup = cfg["max_steps"], int(cfg.get("warmup_steps", 20))
        self.scheduler = LambdaLR(self.optimizer, lambda s: warmup_cosine(
            s, warmup, total_steps, float(cfg.get("min_lr_ratio", .1))))
        self.sampler = BatchSampler(len(self.train_rows), cfg["batch_size"],
                                    seed=int(cfg.get("seed", 0)))
        self.history = []
        self.step = 0

    def train_step(self, step):
        if step != self.step + 1:
            raise ValueError("training steps must be consecutive")
        self.model.train()
        x, y = get_batch(self.train_rows, self.sampler.next_batch())
        t0 = time.perf_counter()
        _, loss = self.model.forward(x, targets=y)
        forward_ms = (time.perf_counter() - t0) * 1000
        if not np.isfinite(loss.item()):
            raise FloatingPointError("non-finite training loss")
        t0 = time.perf_counter()
        self.optimizer.zero_grad()
        loss.backward()
        backward_ms = (time.perf_counter() - t0) * 1000
        grad_norm = clip_grad_norm(self.model.parameters(), float(self.cfg.get("grad_clip", 1.)))
        if not np.isfinite(grad_norm):
            raise FloatingPointError("non-finite training gradient norm")
        t0 = time.perf_counter()
        # Schedule the update about to run, including the first warmup step.
        self.scheduler.step()
        self.optimizer.step()
        opt_ms = (time.perf_counter() - t0) * 1000
        self.step = step
        return {"step": step, "loss": float(loss.item()), "grad_norm": grad_norm,
                "lr": self.optimizer.lr, "forward_ms": forward_ms,
                "backward_ms": backward_ms, "optimizer_ms": opt_ms,
                "tokens_per_s": x.size / max((forward_ms + backward_ms + opt_ms) / 1000., 1e-12)}

    def save(self, path):
        """Save the same complete state to periodic and final checkpoints."""
        return save_checkpoint(path, self.model, self.optimizer, self.step, self.cfg,
                               scheduler=self.scheduler, metrics={"history": self.history},
                               training_state={"sampler": self.sampler.state_dict(),
                                               "data_signature": _data_signature(self.train_rows, self.val_rows)})

    def _resume(self, path):
        payload = load_checkpoint(path)
        if payload["version"] != 2:
            raise ValueError("legacy checkpoint version cannot resume exactly; start a new training run")
        for key in ("corpus_signature", "tokenizer_signature"):
            if payload["config"].get(key) != self.cfg.get(key):
                raise ValueError("resume data/tokenizer identity mismatch")
        # Logging frequency can change without changing training numerics.
        ignored = {"eval_every", "ckpt_every", "log_every"}
        plan = lambda cfg: {k: v for k, v in cfg.items() if k not in ignored}
        if plan(payload["config"]) != plan(self.cfg):
            raise ValueError("resume training config/plan mismatch (including max_steps)")
        state = payload.get("training_state")
        if state is None or state.get("data_signature") != _data_signature(self.train_rows, self.val_rows):
            raise ValueError("resume data mismatch or missing training state")
        if (payload.get("scheduler_state") or {}).get("last_step") != payload["step"]:
            raise ValueError("checkpoint step and scheduler state disagree")
        self.sampler.load_state_dict(state["sampler"])
        load_checkpoint_into(self.model, self.optimizer, self.scheduler, path)
        self.step = payload["step"]
        self.history = payload.get("metrics", {}).get("history", [])
        self.log(f"resumed from {path} at step {self.step}")

    def fit(self, resume_from=None, *, stop_after=None):
        """Train to max_steps, or pause at an absolute step without replanning LR."""
        if resume_from is not None:
            self._resume(resume_from)
        cfg = self.cfg
        max_steps = cfg["max_steps"]
        end = max_steps if stop_after is None else stop_after
        if not isinstance(end, int) or not self.step <= end <= max_steps:
            raise ValueError("stop_after must be between the current step and max_steps")
        os.makedirs(self.out_dir, exist_ok=True)
        for step in range(self.step + 1, end + 1):
            stats = self.train_step(step)
            self.history.append(stats)
            if step % cfg.get("log_every", 10) == 0 or step == end:
                self.log(f"step {step}/{max_steps} loss {stats['loss']:.4f} "
                         f"gnorm {stats['grad_norm']:.3f} lr {stats['lr']:.2e} "
                         f"tok/s {stats['tokens_per_s']:.0f}")
            if step % cfg.get("eval_every", 50) == 0 or step == max_steps:
                stats["val_loss"] = evaluate(self.model, self.val_rows, cfg["batch_size"])
                self.log(f"  eval: val_loss {stats['val_loss']:.4f}")
            if step % cfg.get("ckpt_every", 100) == 0 or step == end:
                self.save(os.path.join(self.out_dir, "checkpoint.pkl"))
        final = os.path.join(self.out_dir, "final.pkl")
        self.save(final)
        with open(os.path.join(self.out_dir, "history.json"), "w") as stream:
            json.dump(self.history, stream, indent=1)
        return final


def warmup_cosine_ratio(step, warmup_steps, total_steps, min_ratio):
    """Backward-compatible name for the shared schedule implementation."""
    return warmup_cosine(step, warmup_steps, total_steps, min_ratio)
