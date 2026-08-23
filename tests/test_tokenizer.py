"""Byte-level BPE tokenizer tests."""

import pytest

from torchzero.tokenizer import BPETokenizer

CORPUS = (
    "the quick brown fox jumps over the lazy dog. "
    "the dog barks; the fox runs. over and over, the end. "
) * 20


class TestBPE:
    def test_lossless_roundtrip(self):
        tok = BPETokenizer().train(CORPUS, vocab_size=280)
        for text in ["the quick brown fox", "over and over!", "the end."]:
            assert tok.decode(tok.encode(text)) == text

    def test_unicode_roundtrip(self):
        tok = BPETokenizer().train("hello wörld naïve café", vocab_size=260)
        text = "naïve café wörld — ünïcode!"
        assert tok.decode(tok.encode(text)) == text

    def test_vocab_size_and_ids(self):
        tok = BPETokenizer().train(CORPUS, vocab_size=300)
        assert tok.vocab_size == 300
        ids = tok.encode("the")
        assert all(0 <= i < 300 for i in ids)
        # merges should actually compress frequent words below byte length
        byte_len = len("the over".encode())
        assert len(tok.encode("the over")) <= byte_len

    def test_deterministic_training(self):
        a = BPETokenizer().train(CORPUS, vocab_size=290)
        b = BPETokenizer().train(CORPUS, vocab_size=290)
        assert a.merges == b.merges

    def test_save_load(self, tmp_path):
        tok = BPETokenizer().train(CORPUS, vocab_size=288)
        path = tmp_path / "tok.json"
        tok.save(path)
        loaded = BPETokenizer.load(path)
        assert loaded.merges == tok.merges
        assert loaded.encode("the lazy dog") == tok.encode("the lazy dog")

    def test_min_vocab_size(self):
        with pytest.raises(ValueError):
            BPETokenizer().train("abc", vocab_size=100)

    def test_empty_input(self):
        tok = BPETokenizer().train(CORPUS, vocab_size=270)
        assert tok.encode("") == []
        assert tok.decode([]) == ""

    def test_compression_ratio(self):
        tok = BPETokenizer().train(CORPUS, vocab_size=320)
        sample = "the quick brown fox jumps over the lazy dog."
        n_bytes = len(sample.encode())
        n_tokens = len(tok.encode(sample))
        assert n_tokens < n_bytes  # learned something useful
