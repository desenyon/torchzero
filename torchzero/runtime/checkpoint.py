"""Versioned, atomically written checkpoints. Load only trusted pickle files."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import pickle
import tempfile
import time

from ..transformer.model import Transformer, TransformerConfig


CHECKPOINT_VERSION = 2


def save_checkpoint(path, model, optimizer, step, config_dict,
                    rng_state=None, metrics=None, *, scheduler=None,
                    training_state=None):
    """Save weights and supplied training state via atomic file replacement.

    Pickle is a trusted-local-artifact format, not a safe interchange format.
    A model-only checkpoint is valid for inference; Trainer supplies the
    scheduler, sampler and data identity required for exact training resume.
    """
    payload = {
        "format": "torchzero.checkpoint", "version": CHECKPOINT_VERSION,
        "step": int(step), "config": copy.deepcopy(config_dict),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "rng_state": copy.deepcopy(rng_state) if rng_state is not None else model.rng_state_dict(),
        "training_state": copy.deepcopy(training_state),
        "metrics": copy.deepcopy(metrics) if metrics is not None else {},
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return payload


def load_checkpoint(path):
    """Read a trusted checkpoint; reject unsupported versions explicitly."""
    with open(path, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict) or payload.get("format") != "torchzero.checkpoint":
        raise ValueError(f"not a torchzero checkpoint: {path}")
    if payload.get("version") not in (1, CHECKPOINT_VERSION):
        raise ValueError(f"unsupported checkpoint version: {payload.get('version')}")
    required = {"step", "config", "model_state"}
    if not required <= payload.keys():
        raise ValueError(f"checkpoint missing fields: {sorted(required - payload.keys())}")
    if not isinstance(payload["step"], int) or payload["step"] < 0:
        raise ValueError("invalid checkpoint step")
    return payload


def build_model_from_config(config_dict, seed=None):
    cfg = TransformerConfig.from_dict(config_dict.get("model", config_dict))
    return Transformer(cfg, seed=seed)


def load_checkpoint_into(model, optimizer, scheduler, path):
    """Restore weights and requested state; returns the validated payload."""
    payload = load_checkpoint(path)
    if optimizer is not None or scheduler is not None:
        if payload["version"] != CHECKPOINT_VERSION:
            raise ValueError("legacy checkpoint version cannot resume training exactly; load weights instead")
        if optimizer is not None and payload.get("optimizer_state") is None:
            raise ValueError("checkpoint has no optimizer state")
        if scheduler is not None and payload.get("scheduler_state") is None:
            raise ValueError("checkpoint has no scheduler state")
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None:
        scheduler.load_state_dict(payload["scheduler_state"])
    if payload["version"] == CHECKPOINT_VERSION:
        model.load_rng_state_dict(payload["rng_state"])
    if hasattr(model, "clear_cache"):
        model.clear_cache()
    return payload
