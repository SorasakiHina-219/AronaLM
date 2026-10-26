"""
Evaluator for AronaLM.

Returns a structured `EvalResult` (B5 / K4) so that NaN / Inf / empty /
failed / format-error outcomes can be distinguished from valid observations.

Backward compatibility: `EvalResult` supports `[]` item access, so existing
pre-B5 callers that do `metrics["loss"]` keep working without modification.
New B5 code should use `result.status` explicitly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch

from aronalm.model import AronaLM


@dataclass
class EvalResult:
    """Structured evaluation outcome (K4)."""

    status: str  # "ok" | "nan" | "inf" | "failed" | "empty" | "format_error"
    loss: float | None = None
    perplexity: float | None = None
    error_message: str | None = None

    @property
    def is_valid(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status}
        if self.loss is not None:
            d["loss"] = self.loss
        if self.perplexity is not None:
            d["perplexity"] = self.perplexity
        if self.error_message is not None:
            d["error_message"] = self.error_message
        return d

    # --- backward-compat: allow dict-style access (legacy callers) ---
    def __getitem__(self, key: str) -> Any:
        return self.as_dict()[key]

    def __contains__(self, key: str) -> bool:
        return key in self.as_dict()

    def get(self, key: str, default: Any = None) -> Any:
        return self.as_dict().get(key, default)

    def items(self):
        return self.as_dict().items()


@torch.no_grad()
def evaluate(
    model: AronaLM,
    dataloader,
    device: torch.device,
) -> EvalResult:
    """Evaluate model on a dataloader.

    Returns an EvalResult with one of:
      - status="ok": valid observation (loss + perplexity populated)
      - status="empty": dataloader yielded zero batches
      - status="nan": any batch produced a NaN loss
      - status="inf": any batch produced an Inf loss
      - status="failed": an exception was raised inside the eval loop
      - status="format_error": the model / dataloader produced an
        inconsistent result type (defensive)

    All non-"ok" statuses represent *invalid* observations that must NOT
    update the consecutive counters (K5).
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0

    try:
        for input_ids in dataloader:
            input_ids = input_ids.to(device)
            loss = model.compute_loss(input_ids)
            v = loss.item()
            if math.isnan(v):
                return EvalResult(status="nan", error_message="NaN loss during evaluation")
            if math.isinf(v):
                return EvalResult(status="inf", error_message="+/-Inf loss during evaluation")
            total_loss += v
            n_batches += 1
    except Exception as e:  # K4: any failure -> "failed", no auto-retry
        return EvalResult(status="failed", error_message=f"{type(e).__name__}: {e}")

    if n_batches == 0:
        return EvalResult(status="empty", error_message="validation dataloader yielded 0 batches")

    avg_loss = total_loss / n_batches
    if math.isnan(avg_loss):
        return EvalResult(status="nan", error_message="mean loss is NaN")
    if math.isinf(avg_loss):
        return EvalResult(status="inf", error_message="mean loss is Inf")

    perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")
    return EvalResult(status="ok", loss=avg_loss, perplexity=perplexity)
