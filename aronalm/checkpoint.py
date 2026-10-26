"""
Checkpoint management for AronaLM.
"""

from __future__ import annotations

from pathlib import Path

import torch

from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer


def save_checkpoint(
    model: AronaLM,
    optimizer,
    step: int,
    config: TransformerConfig,
    tokenizer: BPETokenizer,
    output_dir: Path,
) -> Path:
    """Save model checkpoint with full metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_step{step}.pt"

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "config": {
            "vocab_size": config.vocab_size,
            "max_seq_len": config.max_seq_len,
            "d_model": config.d_model,
            "n_heads": config.n_heads,
            "n_layers": config.n_layers,
            "ffn_dim": config.ffn_dim,
            "dropout": config.dropout,
            "use_bias_in_proj": config.use_bias_in_proj,
            "use_bias_in_ffn": config.use_bias_in_ffn,
            "use_bias_in_output": config.use_bias_in_output,
            "use_bias_in_layernorm": config.use_bias_in_layernorm,
        },
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "tokenizer_merge_rules": tokenizer.merge_rules,
        "seed": 42,
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AronaLM, object | None, TransformerConfig, dict]:
    """Load model checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    cfg_data = checkpoint["config"]
    config = TransformerConfig(
        vocab_size=cfg_data["vocab_size"],
        max_seq_len=cfg_data["max_seq_len"],
        d_model=cfg_data["d_model"],
        n_heads=cfg_data["n_heads"],
        n_layers=cfg_data["n_layers"],
        ffn_dim=cfg_data["ffn_dim"],
        dropout=cfg_data.get("dropout", 0.1),
        use_bias_in_proj=cfg_data.get("use_bias_in_proj", False),
        use_bias_in_ffn=cfg_data.get("use_bias_in_ffn", True),
        use_bias_in_output=cfg_data.get("use_bias_in_output", True),
        use_bias_in_layernorm=cfg_data.get("use_bias_in_layernorm", True),
    )

    model = AronaLM(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    optimizer_state = checkpoint.get("optimizer_state_dict")
    metadata = {
        "step": checkpoint.get("step", 0),
        "seed": checkpoint.get("seed", 42),
        "tokenizer_vocab_size": checkpoint.get("tokenizer_vocab_size"),
        "tokenizer_merge_rules": checkpoint.get("tokenizer_merge_rules", []),
    }

    return model, optimizer_state, config, metadata
