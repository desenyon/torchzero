"""Checkpoint helpers: restore model/optimizer/scheduler state for resume."""

from __future__ import annotations

import numpy as np

from ..transformer.model import Transformer, TransformerConfig
from .trainer import load_checkpoint


def build_model_from_config(config_dict, seed=None):
    cfg = TransformerConfig.from_dict(config_dict["model"]) \
        if "model" in config_dict else TransformerConfig.from_dict(config_dict)
    return Transformer(cfg)


def load_checkpoint_into(model, optimizer, scheduler, path):
    """Restore a checkpoint into live objects. Returns payload."""
    payload = load_checkpoint(path)
    model.load_state_dict(payload["model_state"])
    if optimizer is not None:
        optimizer.load_state_dict(payload["optimizer_state"])
    step = int(payload["step"])
    if scheduler is not None:
        # replay the schedule to the resumed step
        while scheduler.last_step < step:
            scheduler.step()
    np.random.default_rng()  # keep API stable
    return payload
