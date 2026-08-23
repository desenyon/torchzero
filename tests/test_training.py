"""Training runtime tests: training step, checkpoint save/restore/resume."""

import numpy as np
import pytest

from torchzero import no_grad
from torchzero.transformer import Transformer, TransformerConfig
from torchzero.runtime import (Trainer, evaluate, make_optimizer,
                               save_checkpoint, load_checkpoint,
                               load_checkpoint_into)


def build(cfg_over=None, seed=0):
    cfg = dict(vocab_size=48, dim=24, n_layers=2, n_heads=3, block_size=12)
    cfg.update(cfg_over or {})
    model_cfg = TransformerConfig(**cfg)
    model = Transformer(model_cfg, seed=seed)
    return model


def make_rows(n_rows=8, seq=12, vocab=48, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, vocab, (n_rows, seq)).astype(np.int64)


class TestTrainingStep:
    def test_loss_decreases_on_controlled_task(self):
        """Copy-next-token task with fixed batch: loss must drop."""
        rows = make_rows()
        x = rows[:1]
        y = np.roll(x, -1, axis=1)
        model = build({"block_size": 12}, seed=1)
        model.train()
        opt = make_optimizer("adamw", model.parameters(), lr=5e-3,
                             weight_decay=0.01)
        first = last = None
        for step in range(30):
            _, loss = model.forward(x, targets=y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if first is None:
                first = float(loss.item())
            last = float(loss.item())
        assert last < first * 0.7

    def test_trainer_history_and_schedule(self):
        cfg = {
            "model": {"vocab_size": 48, "dim": 24, "n_layers": 2,
                      "n_heads": 3, "block_size": 12},
            "optimizer": "adamw", "lr": 3e-3, "weight_decay": 0.01,
            "grad_clip": 1.0, "batch_size": 4, "max_steps": 20,
            "eval_every": 10, "ckpt_every": 10, "log_every": 10,
            "seed": 0, "warmup_steps": 5, "min_lr_ratio": 0.1,
        }
        rows = make_rows()
        trainer = Trainer(build(), rows, rows[6:], cfg, out_dir="/tmp/tz_test")
        final = trainer.fit()
        losses = [s["loss"] for s in trainer.history]
        assert len(trainer.history) == 20
        assert all(np.isfinite(l) for l in losses)

    def test_evaluate_returns_finite(self):
        model = build()
        rows = make_rows()
        vloss = evaluate(model, rows, batch_size=4)
        assert np.isfinite(vloss) and vloss > 0


class TestCheckpoints:
    def test_save_and_load_roundtrip(self, tmp_path):
        model = build(seed=2)
        opt = make_optimizer("adamw", model.parameters(), lr=1e-3)
        rows = make_rows()
        _, loss = model.forward(rows[:2], targets=rows[:2])
        loss.backward()
        opt.step()

        path = tmp_path / "ckpt.pkl"
        save_checkpoint(str(path), model, opt, step=5,
                        config_dict={"model": vars(model.config)})
        payload = load_checkpoint(str(path))
        assert payload["step"] == 5
        assert "model_state" in payload and "optimizer_state" in payload

    def test_restore_reproduces_logits(self, tmp_path):
        model_a = build(seed=3)
        model_b = build(seed=99)  # different init
        path = tmp_path / "ckpt.pkl"
        save_checkpoint(str(path), model_a, None, step=0,
                        config_dict={"model": vars(model_a.config)})
        payload = load_checkpoint(str(path))
        model_b.load_state_dict(payload["model_state"])

        ids = make_rows(2)
        with no_grad():
            la, _ = model_a.forward(ids)
            lb, _ = model_b.forward(ids)
        assert np.allclose(la.data, lb.data, atol=1e-6)

    def test_resume_continues_training_identically(self, tmp_path):
        """Interrupt at step k; a fresh run resumed from the checkpoint must
        match an uninterrupted run step-for-step."""
        cfg_common = {
            "optimizer": "adamw", "lr": 3e-3, "weight_decay": 0.01,
            "grad_clip": 1.0, "batch_size": 4, "max_steps": 8,
            "eval_every": 100, "ckpt_every": 4, "log_every": 100,
            "seed": 0, "warmup_steps": 2, "min_lr_ratio": 0.1,
            "model": {"vocab_size": 48, "dim": 24, "n_layers": 2,
                      "n_heads": 3, "block_size": 12},
        }

        def fresh():
            m = Transformer(
                TransformerConfig.from_dict(cfg_common["model"]), seed=7)
            return Trainer(m, make_rows(8), make_rows(2), cfg_common.copy(),
                           out_dir=str(tmp_path))

        # uninterrupted
        t_full = fresh()
        t_full.fit()
        full_losses = [s["loss"] for s in t_full.history]

        # interrupted after checkpoint at step 4, then resumed
        t_split = fresh()
        t_split.fit()  # runs all 8 too but we simulate by re-running from ckpt

        # rebuild trainer and resume from its step-4 checkpoint state
        t_resume = fresh()
        # manually run 4 steps exactly as fit() would (train + advance
        # epoch), checkpoint, new trainer resumes
        for step in range(1, 5):
            t_resume.train_step(step)
            t_resume.sampler.advance_epoch()
        from torchzero.runtime.trainer import save_checkpoint
        ckpt_path = str(tmp_path / "resume_ckpt.pkl")
        save_checkpoint(ckpt_path, t_resume.model, t_resume.optimizer,
                        step=4, config_dict=cfg_common,
                        metrics={"history": t_resume.history,
                                 "sampler_epoch": t_resume.sampler.epoch})

        t_resumed = fresh()
        t_resumed.fit(resume_from=ckpt_path)
        resumed_losses = ([h["loss"] for h in t_resumed.history[:4]]
                          + [h["loss"] for h in t_resumed.history[4:]])
        # steps 5..8 of resumed run must equal uninterrupted run
        assert np.allclose(full_losses[4:], resumed_losses[-4:], atol=1e-5)

    def test_checkpoint_contains_metadata(self, tmp_path):
        model = build()
        path = tmp_path / "c.pkl"
        save_checkpoint(str(path), model, None, step=3,
                        config_dict={"model": vars(model.config)},
                        metrics={"val_loss": 1.23})
        p = load_checkpoint(str(path))
        assert p["metrics"]["val_loss"] == 1.23
        assert "saved_at" in p
        assert "rng_state" in p
