"""
Unit tests for AronaLM checkpoint functionality.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer
from aronalm.checkpoint import save_checkpoint, load_checkpoint


@pytest.fixture
def model_and_config():
    config = TransformerConfig(vocab_size=100, d_model=32, n_layers=1, n_heads=2, ffn_dim=64)
    torch.manual_seed(42)
    model = AronaLM(config)
    return model, config


@pytest.fixture
def tokenizer():
    tok = BPETokenizer(vocab_size=100)
    tok.train("hello world test")
    return tok


class TestCheckpoint:
    """Tests for checkpoint save/load."""

    def test_save_and_load(self, model_and_config, tokenizer, tmp_path):
        """Verify basic save/load roundtrip."""
        model, config = model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        ckpt_path = save_checkpoint(model, optimizer, step=10, config=config,
                                    tokenizer=tokenizer, output_dir=tmp_path)
        assert ckpt_path.exists()

        loaded_model, opt_state, loaded_config, metadata = load_checkpoint(ckpt_path, torch.device("cpu"))

        assert loaded_config.vocab_size == config.vocab_size
        assert metadata["step"] == 10
        assert metadata["seed"] == 42

    def test_weights_preserved(self, model_and_config, tokenizer, tmp_path):
        """Verify weights are preserved after save/load."""
        model, config = model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        save_checkpoint(model, optimizer, step=5, config=config,
                        tokenizer=tokenizer, output_dir=tmp_path)

        loaded_model, _, _, _ = load_checkpoint(
            list(tmp_path.glob("*.pt"))[0], torch.device("cpu")
        )

        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded_model.state_dict()[key]), \
                f"Weight mismatch for {key}"

    def test_optimizer_state_preserved(self, model_and_config, tokenizer, tmp_path):
        """Verify optimizer state is preserved after save/load."""
        model, config = model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        # Do one optimization step
        input_ids = torch.randint(0, 100, (2, 8))
        loss = model.compute_loss(input_ids)
        loss.backward()
        optimizer.step()

        save_checkpoint(model, optimizer, step=1, config=config,
                        tokenizer=tokenizer, output_dir=tmp_path)

        _, opt_state, _, _ = load_checkpoint(
            list(tmp_path.glob("*.pt"))[0], torch.device("cpu")
        )
        assert opt_state is not None
        assert "state" in opt_state
        assert "param_groups" in opt_state

    def test_metadata_contains_seed(self, model_and_config, tokenizer, tmp_path):
        """Verify metadata contains the correct seed."""
        model, config = model_and_config
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        save_checkpoint(model, optimizer, step=0, config=config,
                        tokenizer=tokenizer, output_dir=tmp_path)

        _, _, _, metadata = load_checkpoint(
            list(tmp_path.glob("*.pt"))[0], torch.device("cpu")
        )
        assert metadata["seed"] == 42


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
