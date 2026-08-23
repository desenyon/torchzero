"""Integration tests: CLI end-to-end (train -> resume -> generate ->
inspect), debugger data endpoints, forbidden-import static check."""

import os
import subprocess
import sys
import tempfile
import pytest

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TORCHZERO = [sys.executable, "-m", "torchzero.cli.main"]


def run_cli(args, expect_ok=True):
    proc = subprocess.run(TORCHZERO + args, capture_output=True, text=True,
                          cwd=REPO)
    if expect_ok and proc.returncode != 0:
        raise AssertionError(
            f"CLI failed: {args}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc


@pytest.fixture(scope="module")
def trained_checkpoint(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ckpt")
    cfg_path = tmp / "tiny.yaml"
    base_cfg = open(os.path.join(REPO, "configs", "tiny.yaml")).read()
    cfg_path.write_text(base_cfg.replace("checkpoints/tiny",
                                         str(tmp / "tiny")))
    run_cli(["train", str(cfg_path)])
    return str(tmp / "tiny"), str(cfg_path)


class TestCLIEndToEnd:
    def test_generate(self, trained_checkpoint):
        ckpt_dir, _ = trained_checkpoint
        out = run_cli(["generate", ckpt_dir,
                       "the sun rises in the east",
                       "--tokens", "10", "--temperature", "0.0"])
        assert len(out.stdout.strip()) > 0

    def test_resume_from_checkpoint(self, trained_checkpoint):
        _, cfg_path = trained_checkpoint
        proc = run_cli(["train", cfg_path,
                        "--resume", os.path.join(
                            os.path.dirname(cfg_path), "tiny",
                            "checkpoint.pkl")])
        assert "resumed" in proc.stdout

    def test_inspect_html(self, trained_checkpoint, tmp_path):
        ckpt_dir, _ = trained_checkpoint
        html_path = str(tmp_path / "report.html")
        run_cli(["inspect", ckpt_dir, "--html", html_path])
        content = open(html_path).read()
        assert "<html" in content.lower()
        assert "grad" in content.lower()


class TestDebuggerData:
    def _model(self):
        from torchzero.transformer import Transformer, TransformerConfig
        return Transformer(TransformerConfig(vocab_size=32, dim=16,
                                             n_layers=2, n_heads=2,
                                             block_size=8), seed=0)

    def test_forward_trace_real_data(self):
        from debugger.trace import forward_trace
        model = self._model()
        ids = np.random.default_rng(0).integers(0, 32, (1, 6)).astype(np.int64)
        fwd = forward_trace(model, ids, targets=np.roll(ids, -1, axis=1))
        ops = [r["op"] for r in fwd["records"]]
        assert "token_embedding" in ops
        assert ops.count("attention") == 2
        assert ops.count("mlp") == 2
        assert "lm_head" in ops and "cross_entropy" in ops
        # attention matrices are real softmax rows summing to 1
        for r in fwd["records"]:
            if r["op"] == "attention":
                probs = r["attn_probs"]
                assert np.allclose(probs.sum(-1), 1.0, atol=1e-3)
                assert np.isfinite(probs).all()
        assert fwd["memory_bytes"] > 0
        assert fwd["loss"] is not None

    def test_backward_trace_real_data(self):
        from debugger.trace import backward_trace
        model = self._model()
        ids = np.random.default_rng(1).integers(0, 32, (1, 5)).astype(np.int64)
        bwd = backward_trace(model, ids, targets=np.roll(ids, -1, axis=1))
        assert len(bwd["graph_nodes"]) > 10
        assert all(e["ms"] >= 0 for e in bwd["graph_nodes"])
        grads_with_values = [g for g in bwd["grad_norms"]
                             if g["grad_norm"] is not None]
        assert len(grads_with_values) == len(bwd["grad_norms"])
        assert any(g["grad_norm"] > 0 for g in grads_with_values)
        assert bwd["global_grad_norm"] > 0
        assert bwd["parameter_count"] == model.num_parameters()

    def test_report_renders_both_modes(self):
        from debugger.trace import forward_trace, backward_trace, \
            parameter_summary
        from debugger.report import render_text, render_html
        model = self._model()
        ids = np.random.default_rng(2).integers(0, 32, (1, 4)).astype(np.int64)
        fwd = forward_trace(model, ids, targets=np.roll(ids, -1, axis=1))
        bwd = backward_trace(model, ids, targets=np.roll(ids, -1, axis=1))
        text = render_text(fwd, bwd, parameter_summary(model))
        page = render_html(fwd, bwd, parameter_summary(model))
        assert "backward graph" in text and "<table>" in page


class TestHardConstraint:
    def test_no_forbidden_framework_imports(self):
        """README §1: no PyTorch/TF/JAX/MLX in RUNTIME code. The benchmark
        suite may import torch strictly as an optional external reference
        for measurement only (README §12, §16) -- never under torchzero/."""
        forbidden = ("torch", "tensorflow", "jax", "mlx", "keras")
        runtime_dirs = ("torchzero", "debugger", "scripts", "research")
        bad = []
        for root, dirs, files in os.walk(REPO):
            dirs[:] = [d for d in dirs
                       if d not in (".venv", ".git", "__pycache__",
                                    ".pytest_cache", "node_modules",
                                    "benchmarks", "tests")]
            rel = os.path.relpath(root, REPO)
            top = rel.split(os.sep)[0]
            if top not in runtime_dirs and root != REPO:
                continue
            for f in files:
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                src = open(path, encoding="utf-8").read()
                for line in src.splitlines():
                    s = line.strip()
                    if s.startswith(("import ", "from ")):
                        module = s.split()[1].split(".")[0]
                        if module in forbidden and module != "torchzero":
                            # 'torch' must not match 'torchzero'
                            if module == "torch" and \
                                    s.split()[1].startswith("torchzero"):
                                continue
                            bad.append((path, line))
        assert not bad, f"forbidden framework imports in runtime code: {bad}"
