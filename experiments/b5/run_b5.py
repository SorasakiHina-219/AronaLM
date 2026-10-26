"""
B5 early-stop / anomaly-pause / strict safe-recovery experiment.

Thin wrapper over aronalm.Trainer (K1-K6 + decisions A/B + P1-P8):
  - starts from the original step-350 checkpoint (read-only source)
  - isolated output directory per run (never overwrites existing data)
  - records source-checkpoint hash, config, seed, environment, timing
  - writes the 9-field pause-state JSON only when a B5 event occurs
  - never writes checkpoint / best checkpoint on a hard-stop or anomaly

Usage:
  python experiments/b5/run_b5.py --mode smoke          # 50 steps
  python experiments/b5/run_b5.py --mode main           # 100 steps
  python experiments/b5/run_b5.py --mode main --b5-off  # legacy comparison
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer
from aronalm.dataset import create_dataloaders, load_text_from_file
from aronalm.trainer import Trainer

# ----------------------------------------------------------------------
# fixed B5 parameters (K3 baseline / thresholds -- do NOT modify)
# ----------------------------------------------------------------------
BASELINE_VAL_LOSS = 1.2278622048241752
WARNING_THRESHOLD = 1.234001515848296
HARD_STOP_THRESHOLD = 1.240140826872417
CONSECUTIVE_COUNT = 2
EXEMPTION_WINDOW = 10

SOURCE_CKPT = ROOT / "checkpoints_v2c_500steps" / "checkpoint_step350.pt"
SEQ_LEN = 64
BATCH_SIZE = 16
LR = 1e-4
WEIGHT_DECAY = 0.01
EVAL_INTERVAL = 5
SEED = 42
MAX_STEPS = {"smoke": 50, "main": 100}

OUTPUT_BASE = ROOT / "experiments" / "b5"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_output_dir(tag: str) -> Path:
    """Isolated output dir; if the target exists and is non-empty, pick a
    fresh non-conflicting one (never overwrite existing experiment data)."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = OUTPUT_BASE / f"{tag}_{stamp}"
    while candidate.exists() and any(candidate.iterdir()):
        i = 1
        while True:
            candidate = OUTPUT_BASE / f"{tag}_{stamp}_{i}"
            if not candidate.exists() or not any(candidate.iterdir()):
                break
            i += 1
        # re-check the base name after bumping suffix
        break
    candidate.mkdir(parents=True, exist_ok=True)
    return candidate


def main() -> None:
    ap = argparse.ArgumentParser(description="B5 early-stop experiment")
    ap.add_argument("--mode", choices=["smoke", "main"], required=True)
    ap.add_argument("--b5-off", action="store_true",
                    help="run WITHOUT B5 early-stop (legacy comparison)")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cpu")

    # ---- source checkpoint (read-only) ----
    assert SOURCE_CKPT.exists(), f"source checkpoint missing: {SOURCE_CKPT}"
    src_hash = sha256_file(SOURCE_CKPT)
    ckpt = torch.load(SOURCE_CKPT, map_location=device, weights_only=True)
    cfg_data = ckpt["config"]
    config = TransformerConfig(**cfg_data)
    model = AronaLM(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    log(f"Source checkpoint: {SOURCE_CKPT.name} step={ckpt['step']} "
        f"sha256={src_hash[:12]}...")
    log(f"Config: vocab={config.vocab_size} seq_len={config.max_seq_len} "
        f"d_model={config.d_model} heads={config.n_heads} layers={config.n_layers}")

    # ---- tokenizer / data (same as B4-5) ----
    tok = BPETokenizer.load(ROOT / "experiments" / "b4" / "vocab_expand" / "vocab_extended.json")
    mixed_text = (ROOT / "experiments" / "b4" / "b4_5" / "mixed_train_text.txt").read_text(encoding="utf-8")
    val_text = load_text_from_file(ROOT / "data" / "splits_v2" / "val.txt")
    loaders = create_dataloaders(tok, mixed_text, val_text, seq_len=SEQ_LEN, batch_size=BATCH_SIZE, seed=SEED)
    log(f"Data: train={len(loaders['train'])} batches, val={len(loaders['val'])} batches")

    # ---- output dir (isolated) ----
    tag = f"b5_{args.mode}" + ("_legacy" if args.b5_off else "")
    out_dir = make_output_dir(tag)
    log(f"Output dir: {out_dir}")

    # ---- run metadata ----
    meta = {
        "experiment": "B5 early-stop validation",
        "mode": args.mode,
        "b5_enabled": not args.b5_off,
        "source_checkpoint": {
            "path": str(SOURCE_CKPT.relative_to(ROOT)),
            "sha256": src_hash,
            "step": int(ckpt["step"]),
            "size_bytes": SOURCE_CKPT.stat().st_size,
        },
        "config": {k: (v if not isinstance(v, bool) else v) for k, v in cfg_data.items()},
        "seed": SEED,
        "lr": LR,
        "batch_size": BATCH_SIZE,
        "seq_len": SEQ_LEN,
        "max_steps": MAX_STEPS[args.mode],
        "eval_interval": EVAL_INTERVAL,
        "consecutive_count": CONSECUTIVE_COUNT,
        "exemption_window": EXEMPTION_WINDOW,
        "baseline_val_loss": BASELINE_VAL_LOSS,
        "warning_threshold": WARNING_THRESHOLD,
        "hard_stop_threshold": HARD_STOP_THRESHOLD,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    # ---- trainer ----
    pause_state_path = out_dir / "pause_state.json"
    b5_kwargs = {}
    if not args.b5_off:
        b5_kwargs = dict(
            baseline_val_loss=BASELINE_VAL_LOSS,
            warning_threshold=WARNING_THRESHOLD,
            hard_stop_threshold=HARD_STOP_THRESHOLD,
            consecutive_count=CONSECUTIVE_COUNT,
            exemption_window=EXEMPTION_WINDOW,
            pause_state_path=pause_state_path,
        )
    t = Trainer(
        model=model,
        train_loader=loaders["train"],
        val_loader=loaders["val"],
        config=config,
        output_dir=out_dir,
        device=device,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        log_interval=10,
        eval_interval=EVAL_INTERVAL,
        max_steps=MAX_STEPS[args.mode],
        **b5_kwargs,
    )

    t0 = time.time()
    summary = t.train()
    elapsed = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = round(elapsed, 2)
    meta["summary"] = summary

    # ---- write run metadata + training log ----
    (out_dir / "run_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (out_dir / "training_log.json").write_text(
        json.dumps(t.training_log, indent=2, ensure_ascii=False), encoding="utf-8")

    # ---- outcome report ----
    log("=" * 60)
    log(f"B5 {args.mode.upper()} {'(legacy, B5 off)' if args.b5_off else ''}complete in {elapsed:.1f}s")
    log(f"  steps executed: {summary['steps_executed']}/{MAX_STEPS[args.mode]}")
    log(f"  stopped: {summary['stopped']}")
    log(f"  warn_consec={summary['warn_consec']} hard_consec={summary['hard_consec']}")
    log(f"  final val_loss: {summary.get('final_val_loss')}")
    log(f"  output dir: {out_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
