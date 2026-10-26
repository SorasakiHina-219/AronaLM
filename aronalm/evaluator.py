"""
Evaluator for AronaLM.
"""

from __future__ import annotations

import math

import torch

from aronalm.model import AronaLM


@torch.no_grad()
def evaluate(
    model: AronaLM,
    dataloader,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model on a dataloader. Returns dict with 'loss' and 'perplexity'."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids in dataloader:
        input_ids = input_ids.to(device)
        loss = model.compute_loss(input_ids)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

    return {"loss": avg_loss, "perplexity": perplexity}
