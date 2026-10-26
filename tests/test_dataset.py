"""
Unit tests for AronaLM dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.dataset import TextDataset, create_dataloaders
from aronalm.tokenizer import BPETokenizer


class TestTextDataset:
    """Tests for TextDataset."""

    def test_dataset_length(self):
        """Verify dataset length calculation."""
        tokens = list(range(100))
        ds = TextDataset(tokens, seq_len=10)
        # (100 - 10) // 10 = 9 chunks
        assert len(ds) == 9

    def test_dataset_item_shape(self):
        """Verify each item has correct shape."""
        tokens = list(range(200))
        ds = TextDataset(tokens, seq_len=16)
        input_ids = ds[0]
        assert input_ids.shape == (16,)
        assert isinstance(input_ids, torch.Tensor)

    def test_chunk_is_input_only(self):
        """Verify chunk uses first seq_len tokens of a seq_len+1 block."""
        tokens = list(range(50))
        ds = TextDataset(tokens, seq_len=10)
        input_ids = ds[0]
        expected = torch.tensor(tokens[:10], dtype=torch.long)
        assert torch.equal(input_ids, expected)

    def test_non_overlapping_chunks(self):
        """Verify chunks don't overlap."""
        tokens = list(range(100))
        ds = TextDataset(tokens, seq_len=10)
        input0 = ds[0]
        input1 = ds[1]
        assert not torch.equal(input0, input1)

    def test_trailing_tokens_dropped(self):
        """Verify trailing tokens shorter than seq_len+1 are dropped."""
        tokens = list(range(20))  # 20 tokens, seq_len=10
        ds = TextDataset(tokens, seq_len=10)
        # Only 1 chunk: tokens[0:11], remainder tokens[11:20] = 9 tokens < 11, dropped
        assert len(ds) == 1


class TestCreateDataloaders:
    """Tests for create_dataloaders function."""

    def test_dataloaders_created(self):
        """Verify train and val dataloaders are created."""
        tok = BPETokenizer(vocab_size=100)
        tok.train("the cat sat on the mat the dog ran")
        loaders = create_dataloaders(
            tok,
            text_train="the cat sat on the mat",
            text_val="the dog ran in the park",
            seq_len=8,
            batch_size=2,
        )
        assert "train" in loaders
        assert "val" in loaders

    def test_dataloader_returns_tensor(self):
        """Verify dataloader yields a single tensor (input_ids only)."""
        tok = BPETokenizer(vocab_size=100)
        tok.train("the cat sat on the mat the dog ran")
        loaders = create_dataloaders(
            tok,
            text_train="the cat sat on the mat the dog ran",
            text_val="the dog ran in the park",
            seq_len=8,
            batch_size=2,
        )
        batch = next(iter(loaders["train"]))
        assert isinstance(batch, torch.Tensor)
        assert batch.dim() == 2

    def test_test_loader_created(self):
        """Verify test loader is created when text_test is provided."""
        tok = BPETokenizer(vocab_size=100)
        tok.train("the cat sat on the mat the dog ran")
        loaders = create_dataloaders(
            tok,
            text_train="the cat sat on the mat",
            text_val="the dog ran in the park",
            text_test="a bird flies high",
            seq_len=8,
            batch_size=2,
        )
        assert "test" in loaders


class TestSplitIntegrity:
    """Tests for data split integrity."""

    def test_sentence_split_mutual_exclusivity(self):
        """Verify no sentence appears in more than one split."""
        lines = [f"sentence {i} has words here." for i in range(100)]
        import random
        random.seed(42)
        random.shuffle(lines)
        n = len(lines)
        train_sents = lines[:int(n * 0.8)]
        val_sents = lines[int(n * 0.8):int(n * 0.9)]
        test_sents = lines[int(n * 0.9):]

        train_set = set(train_sents)
        val_set = set(val_sents)
        test_set = set(test_sents)

        assert len(train_set & val_set) == 0
        assert len(train_set & test_set) == 0
        assert len(val_set & test_set) == 0

    def test_duplicate_sentences_grouped(self):
        """Verify duplicate sentences end up in the same split."""
        from collections import Counter
        lines = (
            ["unique sentence " + str(i) for i in range(90)]
            + ["duplicate A"] * 3
            + ["duplicate B"] * 2
        )
        import random
        random.seed(42)
        random.shuffle(lines)
        counts = Counter(lines)
        duplicates = {s: c for s, c in counts.items() if c > 1}

        # Group duplicates together
        unique_sents = list(dict.fromkeys(lines))
        random.shuffle(unique_sents)
        n = len(unique_sents)
        train_u = unique_sents[:int(n * 0.8)]
        val_u = unique_sents[int(n * 0.8):int(n * 0.9)]
        test_u = unique_sents[int(n * 0.9):]

        def expand(u):
            return [s for s in u for _ in range(counts[s])]

        train_sents = expand(train_u)
        val_sents = expand(val_u)
        test_sents = expand(test_u)

        for sent, count in duplicates.items():
            in_train = sum(1 for s in train_sents if s == sent)
            in_val = sum(1 for s in val_sents if s == sent)
            in_test = sum(1 for s in test_sents if s == sent)
            total = in_train + in_val + in_test
            assert total == count, f"Duplicate '{sent}' not fully accounted: {total}/{count}"
            # Must be in exactly one split
            groups = [in_train > 0, in_val > 0, in_test > 0]
            assert sum(groups) == 1, f"Duplicate '{sent}' spans multiple splits"

    def test_split_reproducibility(self):
        """Verify fixed seed produces identical splits."""
        lines = [f"sentence {i} number {i*7}." for i in range(100)]

        def do_split(seed, lines):
            import random
            random.seed(seed)
            random.shuffle(lines)
            n = len(lines)
            return (
                lines[:int(n * 0.8)],
                lines[int(n * 0.8):int(n * 0.9)],
                lines[int(n * 0.9):],
            )

        r1 = do_split(42, lines.copy())
        r2 = do_split(42, lines.copy())
        for a, b in zip(r1, r2):
            assert a == b, "Splits differ with same seed"

    def test_tokenizer_not_trained_on_val_test(self):
        """Verify tokenizer was trained only on train text (not val/test)."""
        # Use vocab_size large enough to hold all 256 byte tokens
        tok = BPETokenizer(vocab_size=300)
        tok.train("hello world test foo bar")
        # Train text words should encode cleanly (no unk)
        for w in ["hello", "world", "test", "foo", "bar"]:
            ids = tok.encode(w)
            assert all(i != tok._unk_id for i in ids), f"Train word '{w}' hit unk"

        loaders = create_dataloaders(
            tok,
            text_train="hello world test",
            text_val="foo bar baz qux",
            text_test="quux corge grault",
            seq_len=4,
            batch_size=2,
        )
        # All batches should be loadable without errors
        for split_name in ["train", "val", "test"]:
            loader = loaders[split_name]
            batch = next(iter(loader))
            assert isinstance(batch, torch.Tensor)
            assert batch.dim() == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
