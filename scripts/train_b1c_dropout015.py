"""
AronaLM Phase B1c: Dropout=0.15 Controlled Experiment.

Independent wrapper that replicates the Phase A v2 training pipeline with:
  - model.dropout = 0.15  (single-variable change vs baseline 0.1)
  - Independent checkpoint directory: checkpoints_v2c_dropout015/
  - Independent log file: logs/b1c_dropout015_training_log.txt
  - --no-data-gen flag enforced (no data regeneration)
  - No modification of scripts/train_smoke.py or any baseline artifacts

Usage:
    .venv\\Scripts\\python.exe scripts\\train_b1c_dropout015.py --no-data-gen
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

# ---------------------------------------------------------------------------
# Project-root resolution (same as train_smoke.py)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.checkpoint import load_checkpoint
from aronalm.dataset import create_dataloaders
from aronalm.evaluator import evaluate
from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer
from aronalm.trainer import Trainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_write(log_file: Path, line: str) -> None:
    """Append one line to the experiment log file (UTF-8)."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def run_generate_test(
    model: AronaLM,
    tokenizer: BPETokenizer,
    prompts: list[str],
    temps: list[float],
    max_new_tokens: int,
    label: str,
    log_file: Path,
) -> None:
    """Run generation tests and write results to both console and log."""
    print(f"\n  --- {label} generation ---")
    log_write(log_file, f"\n  --- {label} generation ---")
    for temp in temps:
        for prompt in prompts:
            model.eval()
            pid = tokenizer.encode(prompt)
            with torch.no_grad():
                gen = model.generate(torch.tensor([pid]), max_new_tokens=max_new_tokens, temperature=temp)
            text = tokenizer.decode(gen[0].tolist())
            safe = text.encode("ascii", errors="replace").decode("ascii")
            msg = f"    prompt='{prompt}'  temp={temp}  ->  '{safe}'"
            print(f"  {msg}")
            log_write(log_file, msg)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="AronaLM Phase B1c: Dropout=0.15 Controlled Experiment"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "smoke_b1c_dropout015.yaml",
        help="Path to B1c experiment config YAML",
    )
    parser.add_argument(
        "--no-data-gen",
        action="store_true",
        help="Skip synthetic data generation; use existing corpus.txt",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Resolve paths
    # ------------------------------------------------------------------
    log_file = PROJECT_ROOT / "logs" / "b1c_dropout015_training_log.txt"
    checkpoint_dir = PROJECT_ROOT / "checkpoints_v2c_dropout015"

    # ------------------------------------------------------------------
    # 2. Safety checks (must pass before any work begins)
    # ------------------------------------------------------------------
    errors = []

    # 2a. Checkpoint directory must not exist or must be empty
    if checkpoint_dir.exists():
        existing = list(checkpoint_dir.iterdir())
        if existing:
            errors.append(
                f"BLOCKED: {checkpoint_dir} already exists with "
                f"{len(existing)} file(s). Aborting to prevent overwriting."
            )

    # 2b. Log file must not already exist
    if log_file.exists():
        errors.append(
            f"BLOCKED: {log_file} already exists. "
            "Aborting to prevent overwriting previous experiment log."
        )

    # 2c. Disk space
    free_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
    if free_gb < 1.0:
        errors.append(
            f"BLOCKED: Insufficient disk space ({free_gb:.2f} GB free). "
            "Need >= 1 GB. Do not delete files to make room."
        )

    # 2d. Config file must exist
    if not args.config.exists():
        errors.append(
            f"BLOCKED: Config file not found: {args.config}"
        )

    # 2e. --no-data-gen must be set
    if not args.no_data_gen:
        errors.append(
            "BLOCKED: --no-data-gen flag is required. "
            "Passing it prevents accidental data regeneration."
        )

    if errors:
        print("\n" + "=" * 60)
        print("B1c EXPERIMENT BLOCKED -- PRE-FLIGHT CHECK FAILED")
        print("=" * 60)
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    # Safe to proceed -- ensure output directories exist
    log_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Deterministic seeds
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)

    # ------------------------------------------------------------------
    # 4. Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)
    max_steps = config["training"]["max_steps"]
    device = torch.device("cpu")

    print("=" * 60)
    print("ARONALM PHASE B1c -- DROPOUT=0.15 EXPERIMENT")
    print("=" * 60)
    log_write(log_file, "=== B1c Experiment Started ===")
    log_write(log_file, f"Config: {args.config}")
    log_write(log_file, f"  dropout: {config['model']['dropout']}")
    log_write(log_file, f"  batch_size: {config['training']['batch_size']}")
    log_write(log_file, f"  lr: {config['training']['lr']}")
    log_write(log_file, f"  weight_decay: {config['training']['weight_decay']}")
    log_write(log_file, f"  max_steps: {max_steps}")
    log_write(log_file, f"  seed: {config['training']['seed']}")
    log_write(log_file, f"  checkpoint_dir: {checkpoint_dir}")
    log_write(log_file, f"  log_file: {log_file}")

    # ------------------------------------------------------------------
    # 5. Data preparation (identical to train_smoke.py, skip gen)
    # ------------------------------------------------------------------
    print("\n[1/5] Data preparation (skipping generation)...")
    log_write(log_file, "\n[1/5] Data preparation (skipping generation)...")

    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    raw_path = data_dir / "raw" / "corpus.txt"
    splits_dir = data_dir / "splits_v2"

    if not raw_path.exists():
        print("  ERROR: data/raw/corpus.txt not found.")
        sys.exit(1)

    # Read existing data
    with open(raw_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Replicate the exact split logic from train_smoke.py
    random.seed(42)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    sentence_counts = Counter(lines)
    duplicates = {s: c for s, c in sentence_counts.items() if c > 1}

    unique_sents = list(dict.fromkeys(lines))
    sent_to_indices = defaultdict(list)
    for i, s in enumerate(lines):
        sent_to_indices[s].append(i)

    random.shuffle(unique_sents)
    n_unique = len(unique_sents)
    train_unique = unique_sents[: int(n_unique * 0.8)]
    val_unique = unique_sents[int(n_unique * 0.8) : int(n_unique * 0.9)]
    test_unique = unique_sents[int(n_unique * 0.9) :]

    def expand_with_duplicates(unique_list: list[str]) -> list[str]:
        return [s for s in unique_list for _ in range(sentence_counts[s])]

    train_sents = expand_with_duplicates(train_unique)
    val_sents = expand_with_duplicates(val_unique)
    test_sents = expand_with_duplicates(test_unique)

    train_text = "\n".join(train_sents)
    val_text = "\n".join(val_sents)
    test_text = "\n".join(test_sents)

    train_tokens = len(train_text.split())
    val_tokens = len(val_text.split())
    test_tokens = len(test_text.split())

    print(f"  Train: {len(train_sents)} sentences, {train_tokens} tokens")
    print(f"  Val:   {len(val_sents)} sentences, {val_tokens} tokens")
    print(f"  Test:  {len(test_sents)} sentences, {test_tokens} tokens")
    log_write(log_file, f"  Train: {len(train_sents)} sents, {train_tokens} tokens")
    log_write(log_file, f"  Val:   {len(val_sents)} sents, {val_tokens} tokens")
    log_write(log_file, f"  Test:  {len(test_sents)} sents, {test_tokens} tokens")

    # ------------------------------------------------------------------
    # 6. Tokenizer
    # ------------------------------------------------------------------
    print("\n[2/5] Loading tokenizer...")
    log_write(log_file, "\n[2/5] Loading tokenizer...")
    tokenizer_path = PROJECT_ROOT / config["paths"]["tokenizer_path"]
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Merge rules: {len(tokenizer.merge_rules)}")
    log_write(log_file, f"  Vocab: {len(tokenizer.vocab)}, Merge rules: {len(tokenizer.merge_rules)}")

    # ------------------------------------------------------------------
    # 7. Model creation (dropout=0.15)
    # ------------------------------------------------------------------
    print("\n[3/5] Creating model (dropout=0.15, random init)...")
    log_write(log_file, "\n[3/5] Creating model (dropout=0.15, random init)...")
    model_config = TransformerConfig(
        vocab_size=config["model"]["vocab_size"],
        max_seq_len=config["model"]["max_seq_len"],
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        n_layers=config["model"]["n_layers"],
        ffn_dim=config["model"]["ffn_dim"],
        dropout=config["model"]["dropout"],
    )
    model = AronaLM(model_config)
    n_params = model.count_parameters()
    print(f"  Parameters: {n_params:,}")
    assert n_params == model_config.total_parameters(), (
        f"Parameter count mismatch: {n_params} != {model_config.total_parameters()}"
    )
    print(f"  Dropout: {model_config.dropout}")
    first_weight = next(model.parameters()).mean().item()
    print(f"  First weight mean: {first_weight:.6f} (random init)")
    log_write(log_file, f"  Params: {n_params:,}, Dropout: {model_config.dropout}")
    log_write(log_file, f"  First weight mean: {first_weight:.6f}")

    # ------------------------------------------------------------------
    # 8. Dataloaders
    # ------------------------------------------------------------------
    print("\n[4/5] Building dataloaders and training...")
    log_write(log_file, "\n[4/5] Building dataloaders and training...")
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        text_train=train_text,
        text_val=val_text,
        text_test=test_text,
        seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
    )
    print(f"  Train chunks: {len(dataloaders['train'].dataset)}")
    print(f"  Val   chunks: {len(dataloaders['val'].dataset)}")
    print(f"  Test  chunks: {len(dataloaders['test'].dataset)}")
    log_write(log_file, f"  Train chunks: {len(dataloaders['train'].dataset)}")
    log_write(log_file, f"  Val   chunks: {len(dataloaders['val'].dataset)}")
    log_write(log_file, f"  Test  chunks: {len(dataloaders['test'].dataset)}")

    # ------------------------------------------------------------------
    # 9. Training
    # ------------------------------------------------------------------
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=model_config,
        output_dir=checkpoint_dir,
        device=device,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
        log_interval=config["training"]["log_interval"],
        eval_interval=config["training"]["eval_interval"],
        max_steps=max_steps,
    )

    start_time = time.time()
    summary = trainer.train()
    elapsed = time.time() - start_time

    # Write per-step log entries
    for entry in trainer.training_log:
        parts = [
            f"  step={entry['step']:>3}",
            f"train_loss={entry['train_loss']:.6f}",
            f"elapsed={entry['elapsed_sec']:.1f}s",
        ]
        if "val_loss" in entry:
            parts.append(f"val_loss={entry['val_loss']:.6f}")
        log_write(log_file, "  ".join(parts))

    print(f"\n  Training complete:")
    print(f"    Steps:        {summary['total_steps']}")
    print(f"    Time:         {elapsed:.2f}s ({summary['steps_per_sec']:.2f} steps/s)")
    print(f"    Final train loss: {summary['final_train_loss']:.4f}")
    print(f"    Final val loss:   {summary.get('final_val_loss', 'N/A')}")
    log_write(log_file, f"\nTraining complete in {elapsed:.2f}s")
    log_write(log_file, f"  Final train loss: {summary['final_train_loss']:.6f}")
    log_write(log_file, f"  Final val loss: {summary.get('final_val_loss', 'N/A'):.6f}")
    log_write(log_file, f"  Params: {summary['params']:,}")

    # ------------------------------------------------------------------
    # 10. Final evaluation -- A-group re-eval + B1c-group test
    # ------------------------------------------------------------------
    print("\n[5/5] Final evaluation (A-group re-eval + B1c-group test)...")
    log_write(log_file, "\n[5/5] Final evaluation...")

    # --- A-group: re-evaluate step200 checkpoint on current test loader ---
    a_ckpt = PROJECT_ROOT / "checkpoints_v2" / "checkpoint_step200.pt"
    m_a, _, _, meta_a = load_checkpoint(a_ckpt, device)
    val_a = evaluate(m_a, dataloaders["val"], device)
    test_a = evaluate(m_a, dataloaders["test"], device)
    print(f"  A-group (step200): val_loss={val_a['loss']:.6f}  "
          f"test_loss={test_a['loss']:.6f}  test_ppl={test_a['perplexity']:.4f}")
    log_write(log_file, f"  A-group step200: val={val_a['loss']:.6f}  test={test_a['loss']:.6f}  test_ppl={test_a['perplexity']:.4f}")

    # --- B1c-group: find best checkpoint by val_loss, then test ---
    ckpt_files = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    best_c_step = None
    best_c_val_loss = float("inf")
    best_c_ckpt_path = None

    for ckpt in ckpt_files:
        m_tmp, _, _, meta_tmp = load_checkpoint(ckpt, device)
        val_tmp = evaluate(m_tmp, dataloaders["val"], device)
        step = meta_tmp["step"]
        print(f"    B1c step{step}: val_loss={val_tmp['loss']:.6f}")
        log_write(log_file, f"    B1c step{step}: val_loss={val_tmp['loss']:.6f}")
        if val_tmp["loss"] < best_c_val_loss:
            best_c_val_loss = val_tmp["loss"]
            best_c_step = step
            best_c_ckpt_path = ckpt

    # Evaluate B1c-group best checkpoint on test set
    m_c, _, _, meta_c = load_checkpoint(best_c_ckpt_path, device)
    val_c = evaluate(m_c, dataloaders["val"], device)
    test_c = evaluate(m_c, dataloaders["test"], device)
    print(f"  B1c-group (step{best_c_step}): val_loss={val_c['loss']:.6f}  "
          f"test_loss={test_c['loss']:.6f}  test_ppl={test_c['perplexity']:.4f}")
    log_write(log_file, f"  B1c-group step{best_c_step}: val={val_c['loss']:.6f}  test={test_c['loss']:.6f}  test_ppl={test_c['perplexity']:.4f}")

    # --- Generation tests (all three groups) ---
    prompts = ["the cat", "the light", "a small bird"]
    run_generate_test(m_a, tokenizer, prompts, [0.0, 1.0],
                      max_new_tokens=20,
                      label="A-group (step200, dropout=0.1)",
                      log_file=log_file)
    run_generate_test(m_c, tokenizer, prompts, [0.0, 1.0],
                      max_new_tokens=20,
                      label=f"B1c-group (step{best_c_step}, dropout=0.15)",
                      log_file=log_file)

    # ------------------------------------------------------------------
    # 11. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("B1c EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Config:               smoke_b1c_dropout015.yaml")
    print(f"  Dropout (B1c):        {model_config.dropout}")
    print(f"  Checkpoints dir:      {checkpoint_dir}")
    print(f"  Log file:             {log_file}")
    print(f"  Training time:        {elapsed:.2f}s")
    print(f"  Best B1c step:        {best_c_step}  (val_loss={best_c_val_loss:.6f})")
    print(f"  A-group test_loss:    {test_a['loss']:.6f}")
    print(f"  B1c-group test_loss:  {test_c['loss']:.6f}")
    diff = test_c['loss'] - test_a['loss']
    print(f"  Delta (B1c - A):      {diff:+.6f}")
    print("=" * 60)
    log_write(log_file, "\n" + "=" * 60)
    log_write(log_file, "B1c EXPERIMENT COMPLETE")
    log_write(log_file, f"  Best B1c step: {best_c_step}, val_loss={best_c_val_loss:.6f}")
    log_write(log_file, f"  A-group test_loss: {test_a['loss']:.6f}")
    log_write(log_file, f"  B1c-group test_loss: {test_c['loss']:.6f}")
    log_write(log_file, f"  Delta (B1c - A): {diff:+.6f}")
    log_write(log_file, "=" * 60)


if __name__ == "__main__":
    main()
