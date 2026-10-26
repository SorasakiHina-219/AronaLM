"""
Dataset and DataLoader for AronaLM training.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

from aronalm.tokenizer import BPETokenizer


class TextDataset(Dataset):
    """
    Causal language modeling dataset.
    Splits text into chunks of seq_len+1 tokens; each chunk provides
    input_ids of length seq_len. Labels are computed internally by
    AronaLM.compute_loss().
    """

    def __init__(self, token_ids: list[int], seq_len: int = 64):
        self.seq_len = seq_len
        self.chunks = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[i: i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.chunks.append(chunk)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> torch.Tensor:
        chunk = self.chunks[idx]
        return torch.tensor(chunk[: self.seq_len], dtype=torch.long)


def create_dataloaders(
    tokenizer: BPETokenizer,
    text_train: str,
    text_val: str,
    text_test: str | None = None,
    seq_len: int = 64,
    batch_size: int = 16,
    seed: int = 42,
) -> dict[str, DataLoader]:
    """
    Create train/val/test DataLoaders from text strings.
    """
    import random
    random.seed(seed)

    def text_to_tokens(text: str) -> list[int]:
        tokens = []
        for word in text.split():
            tokens.extend(tokenizer.encode(word))
            tokens.append(tokenizer.vocab.get(" ", tokenizer._unk_id))
        return tokens

    train_tokens = text_to_tokens(text_train)
    val_tokens = text_to_tokens(text_val)

    train_ds = TextDataset(train_tokens, seq_len=seq_len)
    val_ds = TextDataset(val_tokens, seq_len=seq_len)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False
    )

    result = {"train": train_loader, "val": val_loader}

    if text_test is not None:
        test_tokens = text_to_tokens(text_test)
        test_ds = TextDataset(test_tokens, seq_len=seq_len)
        test_loader = DataLoader(
            test_ds, batch_size=batch_size, shuffle=False
        )
        result["test"] = test_loader

    return result


def load_text_from_file(path: Path) -> str:
    """Load text from a .txt file."""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
