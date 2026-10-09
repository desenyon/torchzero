"""Real state-lifecycle and CLI regressions; no optional ML framework needed."""

import json
import pickle
import subprocess
import sys

import numpy as np
import pytest
import yaml

from torchzero import Tensor
from torchzero.data import dataset
from torchzero.nn.layers import Dropout
from torchzero.optim import SGD, Adam, AdamW
from torchzero.runtime import Trainer, evaluate, save_checkpoint, load_checkpoint
from torchzero.transformer import Transformer, TransformerConfig


def model(seed=4, dropout=0.25):
    return Transformer(TransformerConfig(vocab_size=16, dim=8, n_layers=1,
                                         n_heads=2, block_size=6, dropout=dropout), seed=seed)


def training_config():
    return dict(model=model().config.to_dict(), optimizer="adamw", lr=0.01,
                batch_size=2, max_steps=6, warmup_steps=2, seed=19,
                eval_every=2, ckpt_every=2, log_every=20, grad_clip=1.)


def trainer(path, optimizer="adamw", seed=4):
    cfg = training_config()
    cfg["optimizer"] = optimizer
    rows = (np.arange(35).reshape(5, 7) % 16).astype(np.int64)
    return Trainer(model(seed=seed), rows, rows[:3], cfg, str(path), log_fn=lambda _: None)


def test_next_token_batches_never_wrap():
    rows = dataset.pack_lm_sequences(np.arange(11), 4)
    np.testing.assert_array_equal(rows, [[0, 1, 2, 3, 4], [4, 5, 6, 7, 8]])
    x, y = dataset.get_batch(rows, [0, 1])
    np.testing.assert_array_equal(y, x + 1)
    assert x.shape == (2, 4)


@pytest.mark.parametrize("num_rows,batch_size", [(5, 2), (4, 4), (3, 8)])
def test_consuming_sampler_covers_epoch_and_restores_cursor(num_rows, batch_size):
    sampler = dataset.BatchSampler(num_rows, batch_size, seed=5)
    first = sampler.next_batch()
    saved = sampler.state_dict()
    restored = dataset.BatchSampler(num_rows, batch_size, seed=5)
    restored.load_state_dict(saved)
    rest = []
    for _ in range((num_rows + batch_size - 1) // batch_size - 1):
        got = sampler.next_batch()
        np.testing.assert_array_equal(got, restored.next_batch())
        rest.extend(got.tolist())
    assert sorted(first.tolist() + rest) == list(range(num_rows))
    np.testing.assert_array_equal(sampler.next_batch(), restored.next_batch())
    assert sampler.epoch == 1


def test_sampler_rejects_mismatched_or_invalid_state():
    sampler = dataset.BatchSampler(5, 2)
    state = sampler.state_dict()
    for update in ({"num_rows": 6}, {"cursor": 9}, {"epoch": -1}, {"batch_size": 3}):
        with pytest.raises(ValueError):
            sampler.load_state_dict({**state, **update})
        assert sampler.state_dict() == state


def test_dropout_stream_advances_reproduces_and_restores():
    x = Tensor(np.ones(128))
    a, b = Dropout(0.5, seed=3), Dropout(0.5, seed=3)
    first = a(x).data
    np.testing.assert_array_equal(first, b(x).data)
    saved = a.rng_state_dict()
    second = a(x).data
    assert not np.array_equal(first, second)
    b.load_rng_state_dict(saved)
    np.testing.assert_array_equal(second, b(x).data)


def test_transformer_dropout_is_active_only_in_training():
    a, b = model(), model()
    ids = np.array([[1, 2, 3]])
    first = a(ids)[0].data
    np.testing.assert_array_equal(first, b(ids)[0].data)
    assert not np.array_equal(first, a(ids)[0].data)
    a.eval()
    np.testing.assert_array_equal(a(ids)[0].data, a(ids)[0].data)


@pytest.mark.parametrize("optimizer", [SGD, Adam, AdamW])
def test_optimizer_restores_buffers_hyperparameters_and_next_update(optimizer):
    a = Tensor(np.array([1., 2.]), True)
    kwargs = {"momentum": .8} if optimizer is SGD else {}
    opt = optimizer([a], lr=.03, **kwargs)
    a.grad = np.array([.2, -.1])
    opt.step()
    b = Tensor(a.data.copy(), True)
    state = opt.state_dict()
    fresh = optimizer([b], lr=.9)
    fresh.load_state_dict(state)
    b.grad = a.grad.copy()
    opt.step()
    fresh.step()
    np.testing.assert_array_equal(a.data, b.data)
    assert fresh.lr == .03


@pytest.mark.parametrize("optimizer", ["sgd", "adam", "adamw"])
@pytest.mark.parametrize("filename", ["checkpoint.pkl", "final.pkl"])
def test_real_interruption_resumes_exactly_with_dropout_and_partial_batches(tmp_path, optimizer, filename):
    full = trainer(tmp_path / "full", optimizer)
    full.fit()
    partial = trainer(tmp_path / "partial", optimizer)
    partial.fit(stop_after=2)
    assert len(partial.history) == 2
    path = tmp_path / "partial" / filename
    saved = load_checkpoint(path)
    assert saved["step"] == 2
    resumed = trainer(tmp_path / "resumed", optimizer, seed=999)
    resumed.fit(resume_from=path)
    for name, value in full.model.state_dict().items():
        np.testing.assert_array_equal(value, resumed.model.state_dict()[name])
    for key in ("loss", "lr", "grad_norm", "val_loss"):
        assert [h.get(key) for h in full.history] == [h.get(key) for h in resumed.history]
    assert full.sampler.state_dict() == resumed.sampler.state_dict()
    assert full.model.rng_state_dict() == resumed.model.rng_state_dict()


def test_resume_rejects_changed_data_and_training_plan(tmp_path):
    original = trainer(tmp_path / "original")
    original.fit(stop_after=2)
    path = tmp_path / "original" / "final.pkl"
    changed = trainer(tmp_path / "changed")
    changed.train_rows[0, 0] = 15
    with pytest.raises(ValueError, match="data"):
        changed.fit(resume_from=path)
    changed = trainer(tmp_path / "changed")
    changed.cfg["max_steps"] += 1
    with pytest.raises(ValueError, match="config|plan"):
        changed.fit(resume_from=path)


def test_legacy_checkpoint_loadable_but_not_exactly_resumable(tmp_path):
    t = trainer(tmp_path)
    path = tmp_path / "legacy.pkl"
    payload = save_checkpoint(path, t.model, t.optimizer, 0, t.cfg)
    payload["version"] = 1
    path.write_bytes(pickle.dumps(payload))
    assert load_checkpoint(path)["version"] == 1
    with pytest.raises(ValueError, match="version|legacy"):
        t.fit(resume_from=path)


def test_unknown_checkpoint_version_rejected(tmp_path):
    path = tmp_path / "future.pkl"
    path.write_bytes(pickle.dumps({"format": "torchzero.checkpoint", "version": 999}))
    with pytest.raises(ValueError, match="version"):
        load_checkpoint(path)


def test_atomic_checkpoint_preserves_previous_on_serialization_failure(tmp_path, monkeypatch):
    path = tmp_path / "checkpoint.pkl"
    m = model()
    save_checkpoint(path, m, None, 0, {"model": m.config.to_dict()})
    before = path.read_bytes()

    def broken_dump(*args, **kwargs):
        args[1].write(b"partial write")
        raise OSError("disk write failed")

    monkeypatch.setattr(pickle, "dump", broken_dump)
    with pytest.raises(OSError, match="disk write"):
        save_checkpoint(path, m, None, 1, {})
    assert path.read_bytes() == before
    assert list(tmp_path.iterdir()) == [path]


def test_evaluation_weighted_by_tokens_restores_all_modes():
    m = model()
    m.blocks[0].mlp.eval()
    rows = (np.arange(21).reshape(3, 7) % 16).astype(np.int64)
    a = evaluate(m, rows, batch_size=2)
    assert m.training and not m.blocks[0].mlp.training
    b = evaluate(m, rows, batch_size=3)
    assert a == pytest.approx(b, abs=2e-7)
    m.eval()
    evaluate(m, rows, batch_size=2)
    assert not m.training


@pytest.mark.parametrize("use_cache", [True, False])
def test_generation_uses_one_random_stream_no_graph_and_restores_modes(use_cache):
    m = model()
    # Zero weights produce uniform logits. Independent NumPy draws are an
    # exact oracle and expose reseeding the sampler on every token.
    for p in m.parameters():
        p.data.fill(0.)
    expected = np.random.default_rng(23).choice(16, size=4, p=np.ones(16) / 16).tolist()
    original = m.forward
    outputs = []

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        outputs.append(result[0])
        return result

    m.forward = capture
    m.blocks[0].mlp.eval()
    generated = m.generate([1], 4, seed=23, use_kv_cache=use_cache)
    assert generated == [1] + expected
    assert all(not out.requires_grad and not out._parents for out in outputs)
    assert m.training and not m.blocks[0].mlp.training
    assert m.kv_caches is None


def test_generation_capacity_and_zero_tokens():
    m = model()
    prompt = list(range(6))
    assert m.generate(prompt, 3) == prompt
    assert m.generate([1], 0) == [1]
    assert m.kv_caches is None
    for ids, n in [([], 1), ([1], -1), ([16], 1)]:
        with pytest.raises((ValueError, IndexError)):
            m.generate(ids, n)


@pytest.mark.parametrize("kwargs", [{"dim": 6, "n_heads": 2}, {"n_heads": 0},
                                     {"dropout": 1.}, {"block_size": 0}, {"n_layers": 0}])
def test_invalid_transformer_config_is_rejected(kwargs):
    values = dict(vocab_size=16, dim=8, n_heads=2, n_layers=1)
    values.update(kwargs)
    with pytest.raises(ValueError):
        TransformerConfig(**values)


def test_cli_file_corpus_train_resume_generate_inspect(tmp_path):
    text = "A real UTF-8 corpus: café, rivers and stars. " * 80
    (tmp_path / "corpus.txt").write_text(text, encoding="utf-8")
    cfg = dict(out_dir=str(tmp_path / "run"),
               model=dict(vocab_size="auto", dim=8, n_layers=1, n_heads=2, block_size=8, dropout=.2),
               data=dict(text="corpus.txt", val_fraction=.2),
               train={**training_config(), "max_steps": 4})
    del cfg["train"]["model"]
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump(cfg))

    def run(*args):
        p = subprocess.run([sys.executable, "-m", "torchzero.cli.main", *map(str, args)],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stdout + p.stderr
        return p.stdout

    run("train", config, "--stop-after", 2)
    path = tmp_path / "run" / "final.pkl"
    assert load_checkpoint(path)["step"] == 2
    run("train", config, "--resume", path)
    assert len(json.loads((tmp_path / "run" / "history.json").read_text())) == 4
    cached = run("generate", path, "A real", "--tokens", 3, "--seed", 8)
    uncached = run("generate", path, "A real", "--tokens", 3, "--seed", 8, "--no-cache")
    assert cached == uncached and cached.startswith("A real")
    report = tmp_path / "report.html"
    run("inspect", path, "A real", "--html", report)
    assert "<html" in report.read_text().lower()
    # Resuming cannot overwrite the tokenizer or silently train changed text.
    tok_path = tmp_path / "run" / "tokenizer.json"
    before = tok_path.read_bytes()
    (tmp_path / "corpus.txt").write_text("changed corpus " * 100)
    p = subprocess.run([sys.executable, "-m", "torchzero.cli.main", "train", str(config),
                        "--resume", str(path)], capture_output=True, text=True)
    assert p.returncode != 0 and "data" in p.stderr.lower()
    assert tok_path.read_bytes() == before


def test_generation_exception_restores_modes_and_clears_cache():
    from torchzero.autograd.engine import get_grad_mode
    m = model()
    m.blocks[0].attn.eval()
    original = m.forward

    def fail_after_forward(*args, **kwargs):
        original(*args, **kwargs)
        raise RuntimeError("injected inference error")

    m.forward = fail_after_forward
    with pytest.raises(RuntimeError, match="injected"):
        m.generate([1, 2], 2)
    assert m.training and not m.blocks[0].attn.training
    assert m.kv_caches is None and get_grad_mode()


def test_top_k_keeps_exactly_k_tied_tokens_and_rejects_invalid_sampling():
    rng = np.random.default_rng(9)
    outputs = [Transformer.sample_next(np.zeros(6), top_k=2, rng=rng) for _ in range(50)]
    assert set(outputs) == {0, 1}
    for kwargs in ({"temperature": -1}, {"temperature": float("nan")}, {"top_k": 0}):
        with pytest.raises(ValueError):
            Transformer.sample_next(np.ones(6), **kwargs)
    assert Transformer.sample_next(np.array([1e308, -1e308]), temperature=1e-308) == 0


def test_cached_chunked_logits_match_and_overflow_rejected():
    from torchzero import no_grad
    m = model().eval()
    ids = np.array([[1, 3, 2, 5, 4, 6]])
    with no_grad():
        full = m(ids)[0].data
        first = m(ids[:, :2], use_cache=True)[0].data
        second = m(ids[:, 2:], use_cache=True, reset_cache=False)[0].data
    np.testing.assert_allclose(np.concatenate([first, second], axis=1), full, atol=1e-6)
    with pytest.raises(ValueError, match="block size"):
        m([[1]], use_cache=True, reset_cache=False)
    assert m.kv_caches[0]["k"].shape[2] == 6


def test_first_update_uses_warmup_schedule(tmp_path):
    t = trainer(tmp_path)
    t.fit(stop_after=1)
    assert t.history[0]["lr"] == t.cfg["lr"] / t.cfg["warmup_steps"]


@pytest.mark.parametrize("optimizer", [SGD, Adam, AdamW])
def test_optimizer_invalid_shapes_rejected_before_mutation(optimizer):
    p = Tensor(np.ones(2), True)
    opt = optimizer([p], lr=.1)
    state = opt.state_dict()
    key = next(iter(state["param_state"][0]))
    state["param_state"][0][key] = np.zeros(3)
    with pytest.raises(ValueError, match="shape"):
        opt.load_state_dict(state)
    assert opt.lr == .1


def test_adamw_does_not_decay_parameters_without_gradients():
    p = Tensor(np.array([1., 2.]), True)
    opt = AdamW([p], lr=.1, weight_decay=.2)
    opt.step()
    np.testing.assert_array_equal(p.data, [1., 2.])


def test_gradient_clipping_accepts_parameter_iterators():
    from torchzero.optim import clip_grad_norm
    p = Tensor(np.array([1., 2.]), True)
    p.grad = np.array([3., 4.])
    assert clip_grad_norm(iter([p]), 1.) == 5.
    np.testing.assert_allclose(p.grad, [.6, .8])


def test_corrupt_scheduler_state_rejected_before_resume(tmp_path):
    t = trainer(tmp_path)
    path = t.fit(stop_after=2)
    payload = load_checkpoint(path)
    payload["scheduler_state"]["last_step"] = 1
    with open(path, "wb") as f:
        pickle.dump(payload, f)
    fresh = trainer(tmp_path / "fresh")
    with pytest.raises(ValueError, match="scheduler"):
        fresh.fit(resume_from=path)
    assert fresh.step == 0


def test_checkpoint_helpers_support_non_transformer_modules(tmp_path):
    from torchzero.nn.layers import Linear
    from torchzero.runtime import load_checkpoint_into
    source, restored = Linear(2, 3, seed=1), Linear(2, 3, seed=2)
    path = tmp_path / "linear.pkl"
    save_checkpoint(path, source, None, 0, {})
    load_checkpoint_into(restored, None, None, path)
    for name, value in source.state_dict().items():
        np.testing.assert_array_equal(value, restored.state_dict()[name])


def test_unsupported_amsgrad_is_explicit():
    with pytest.raises(ValueError, match="amsgrad"):
        Adam([Tensor(np.ones(2), True)], amsgrad=True)


def test_corpus_source_contract_and_saved_external_tokenizer(tmp_path):
    from torchzero.cli.main import _load_config, _corpus_text, _build_tokenizer
    from torchzero.tokenizer import BPETokenizer
    corpus = "UTF-8 café " * 100
    source = tmp_path / "text.txt"
    source.write_text(corpus, encoding="utf-8")
    tok_path = tmp_path / "existing.json"
    BPETokenizer().train(corpus, 270).save(tok_path)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(dict(model={"vocab_size": "auto"}, train={},
                                       data={"text": "text.txt", "tokenizer_path": "existing.json"})))
    cfg = _load_config(path)
    assert _corpus_text(cfg) == corpus
    out = tmp_path / "output"
    tok = _build_tokenizer(cfg, out)
    assert BPETokenizer.load(out / "tokenizer.json").merges == tok.merges
    assert _corpus_text({"data": {"inline_text": corpus}}) == corpus
    with pytest.raises(ValueError, match="choose"):
        _corpus_text({"data": {"text": str(source), "inline_text": "literal"}})
    with pytest.raises(FileNotFoundError):
        _corpus_text({"data": {"text": str(tmp_path / "missing.txt")}})


def test_legacy_model_weights_can_generate(tmp_path):
    from torchzero.cli.main import _load_model_and_tokenizer
    m = model(dropout=0.)
    path = tmp_path / "legacy.pkl"
    payload = save_checkpoint(path, m, None, 0, {"model": m.config.to_dict()})
    payload["version"] = 1
    path.write_bytes(pickle.dumps(payload))
    restored, tok = _load_model_and_tokenizer(str(path))
    assert tok is None
    assert m.generate([1, 2], 3, temperature=0) == restored.generate([1, 2], 3, temperature=0)
