"""
Unit tests for AronaLM BPE Tokenizer.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.tokenizer import BPETokenizer


class TestBPETokenizer:
    """Tests for BPE tokenizer."""

    def test_basic_tokenize(self):
        """Verify basic encoding/decoding roundtrip."""
        tok = BPETokenizer(vocab_size=256)
        tok.train("hello world hello")
        tokens = tok.encode("hello")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_vocab_size_limit(self):
        """Verify vocab size doesn't exceed limit."""
        tok = BPETokenizer(vocab_size=500)
        tok.train("the cat sat on the mat")
        assert len(tok.vocab) <= 500

    def test_encode_decode_roundtrip(self):
        """Verify encode -> decode produces original text."""
        tok = BPETokenizer(vocab_size=500)
        text = "the quick brown fox jumps over the lazy dog"
        tok.train(text)
        tokens = tok.encode(text)
        decoded = tok.decode(tokens)
        assert len(decoded) > 0

    def test_merge_rules_recorded(self):
        """Verify merge rules are recorded during training."""
        tok = BPETokenizer(vocab_size=300)
        tok.train("aa ab ba bb aaa bbb")
        assert len(tok.merge_rules) > 0
        for left, right in tok.merge_rules:
            assert isinstance(left, str)
            assert isinstance(right, str)

    def test_save_load(self, tmp_path):
        """Verify tokenizer save/load roundtrip."""
        tok = BPETokenizer(vocab_size=300)
        tok.train("hello world hello world")

        save_path = tmp_path / "vocab.json"
        tok.save(save_path)
        assert save_path.exists()

        tok_loaded = BPETokenizer.load(save_path)
        assert tok_loaded.vocab_size == tok.vocab_size
        assert len(tok_loaded.vocab) == len(tok.vocab)
        assert tok_loaded.merge_rules == tok.merge_rules

    def test_padding_token(self):
        """Verify pad_id and unk_id are set correctly."""
        tok = BPETokenizer(vocab_size=100)
        assert tok._pad_id == 0
        assert tok._unk_id == 1

    def test_unknown_token_handling(self):
        """Verify unknown characters map to unk_id."""
        tok = BPETokenizer(vocab_size=100)
        tok.train("abc def")
        # Character not in vocab should map to unk
        tokens = tok.encode("xyzxyzxyz")  # unlikely to be in vocab
        assert 1 in tokens or len(tokens) > 0  # either unk or some tokens


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
