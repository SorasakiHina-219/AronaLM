"""
AronaLM BPE Tokenizer — built from scratch, no external dependencies.

Trains a byte-level BPE tokenizer from raw text corpus.
Supports merge rules learned via greedy frequency-based merging.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class BPETokenizer:
    """
    Byte-level Byte-Pair-Encoding tokenizer.

    Vocabulary consists of:
      - All individual bytes (0-255) as initial symbols
      - Merged symbols learned from the corpus
    """

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab: dict[str, int] = {}
        self.reverse_vocab: dict[int, str] = {}
        self.merge_rules: list[tuple[str, str]] = []
        self._pad_id = 0
        self._unk_id = 1

    def _get_initial_symbols(self) -> set[str]:
        """Byte-level initialization: all 256 byte values."""
        return {chr(i) for i in range(256)}

    def _word_to_bytes(self, word: str) -> tuple[str, ...]:
        """Encode a word as a tuple of byte symbols."""
        return tuple(chr(b) for b in word.encode("utf-8"))

    def train(self, text: str, vocab_size: int | None = None):
        """
        Train BPE tokenizer on raw text.

        Args:
            text: raw training corpus
            vocab_size: target vocabulary size (default: self.vocab_size)
        """
        target_vocab = vocab_size or self.vocab_size

        # Build initial vocabulary from bytes
        words = text.split()
        word_freqs = Counter()
        for w in words:
            word_freqs[self._word_to_bytes(w)] += 1

        # Initial vocab: all bytes + whitespace token
        symbols = self._get_initial_symbols()
        symbols.add(" ")
        for sym in sorted(symbols):
            self.vocab[sym] = len(self.vocab)
            self.reverse_vocab[self.vocab[sym]] = sym

        # Iteratively merge most frequent pairs
        for _ in range(target_vocab - len(self.vocab)):
            if len(word_freqs) == 0:
                break
            # Count adjacent pairs
            pair_counts = Counter()
            for word, freq in word_freqs.items():
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += freq

            if not pair_counts:
                break

            # Find most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            if pair_counts[best_pair] == 0:
                break

            # Apply merge
            left, right = best_pair
            merged = left + right
            self.merge_rules.append((left, right))

            # Add merged symbol to vocab
            if merged not in self.vocab:
                self.vocab[merged] = len(self.vocab)
                self.reverse_vocab[len(self.reverse_vocab)] = merged

            # Apply merge to all words
            new_word_freqs = Counter()
            for word, freq in word_freqs.items():
                new_word = self._apply_merge(word, left, right)
                new_word_freqs[new_word] += freq
            word_freqs = new_word_freqs

            if len(self.vocab) >= target_vocab:
                break

    @staticmethod
    def _apply_merge(word: tuple[str, ...], left: str, right: str) -> tuple[str, ...]:
        """Apply a single merge rule to a word tuple."""
        result = []
        i = 0
        while i < len(word):
            if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                result.append(left + right)
                i += 2
            else:
                result.append(word[i])
                i += 1
        return tuple(result)

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        bytes_tuple = self._word_to_bytes(text)
        tokens = []
        i = 0
        while i < len(bytes_tuple):
            matched = False
            for length in range(min(10, len(bytes_tuple) - i), 0, -1):
                candidate = "".join(bytes_tuple[i:i + length])
                if candidate in self.vocab:
                    tokens.append(self.vocab[candidate])
                    i += length
                    matched = True
                    break
            if not matched:
                tokens.append(self.vocab.get(bytes_tuple[i], self._unk_id))
                i += 1
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs to text."""
        chars = []
        for tid in token_ids:
            sym = self.reverse_vocab.get(tid, "")
            if sym:
                chars.append(sym)
        try:
            return "".join(chars).encode("latin-1").decode("utf-8", errors="replace")
        except Exception:
            return "".join(chars)

    def save(self, path: Path):
        """Save tokenizer to JSON file."""
        data = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "merge_rules": [list(rule) for rule in self.merge_rules],
            "pad_id": self._pad_id,
            "unk_id": self._unk_id,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        """Load tokenizer from JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        tok = cls(vocab_size=data["vocab_size"])
        tok.vocab = data["vocab"]
        tok.reverse_vocab = {int(v): k for k, v in data["vocab"].items()}
        tok.merge_rules = [tuple(rule) for rule in data.get("merge_rules", [])]
        tok._pad_id = data.get("pad_id", 0)
        tok._unk_id = data.get("unk_id", 1)
        return tok
