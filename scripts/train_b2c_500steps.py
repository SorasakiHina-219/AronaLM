"""
AronaLM Phase B2c: Extended Training (500 steps) Controlled Experiment.

Same configuration as A v2 (dropout=0.1) but with max_steps=500 instead of 200.
Tests whether longer training improves validation performance.

Independent wrapper with:
  - Independent checkpoint directory: checkpoints_v2c_500steps/
  - Independent log file: logs/b2c_500steps_training_log.txt
  - --no-data-gen flag enforced
  - Reports step=200 metrics for direct comparison with A v2
  - No modification of scripts/train_smoke.py or any baseline artifacts

Usage:
    .venv\\Scripts\\python.exe scripts\\train_b2c_500steps.py --no-data-gen
"""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch

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
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def log_write(log_file: Path, line: str) -> None:
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()


def run_generate_test(
    model, tokenizer, prompts, temps, max_new_tokens, label, log_file
):
    print(f"\n  --- {label} generation ---")
    log_write(log_file, f"\n  --- {label} generation ---")
    for temp in temps:
        for prompt in prompts:
            model.eval()
            pid = tokenizer.encode(prompt)
            with torch.no_grad():
                gen = model.generate(
                    torch.tensor([pid]),
                    max_new_tokens=max_new_tokens,
                    temperature=temp,
                )
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
        description="AronaLM Phase B2c: Extended Training (500 steps)"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "smoke_b2c_500steps.yaml",
    )
    parser.add_argument(
        "--no-data-gen",
        action="store_true",
        help="Skip synthetic data generation; use existing corpus.txt",
    )
    args = parser.parse_args()

    log_file = PROJECT_ROOT / "logs" / "b2c_500steps_training_log.txt"
    checkpoint_dir = PROJECT_ROOT / "checkpoints_v2c_500steps"

    # ------------------------------------------------------------------
    # Safety checks
    # ------------------------------------------------------------------
    errors = []

    if checkpoint_dir.exists():
        existing = list(checkpoint_dir.iterdir())
        if existing:
            errors.append(
                f"BLOCKED: {checkpoint_dir} already exists with "
                f"{len(existing)} file(s)."
            )

    if log_file.exists():
        errors.append(
            f"BLOCKED: {log_file} already exists."
        )

    free_gb = shutil.disk_usage("C:\\").free / (1024 ** 3)
    if free_gb < 1.0:
        errors.append(f"BLOCKED: Insufficient disk space ({free_gb:.2f} GB).")

    if not args.config.exists():
        errors.append(f"BLOCKED: Config not found: {args.config}")

    if not args.no_data_gen:
        errors.append("BLOCKED: --no-data-gen is required.")

    if errors:
        print("\n" + "=" * 60)
        print("B2c EXPERIMENT BLOCKED -- PRE-FLIGHT CHECK FAILED")
        print("=" * 60)
        for e in errors:
            print(f"  ERROR: {e}")
        sys.exit(1)

    log_file.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Deterministic seeds
    # ------------------------------------------------------------------
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)

    # ------------------------------------------------------------------
    # Load config
    # ------------------------------------------------------------------
    config = load_config(args.config)
    max_steps = config["training"]["max_steps"]
    device = torch.device("cpu")

    print("=" * 60)
    print("ARONALM PHASE B2c -- EXTENDED TRAINING (500 STEPS)")
    print("=" * 60)
    log_write(log_file, "=== B2c Experiment Started ===")
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
    # Data preparation (skip generation, use existing splits_v2)
    # ------------------------------------------------------------------
    print(f"\n[1/{max_steps//100+1}] Data preparation (skipping generation)...")
    log_write(log_file, "\n[1] Data preparation (skipping generation)...")

    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    raw_path = data_dir / "raw" / "corpus.txt"

    if not raw_path.exists():
        print("  ERROR: data/raw/corpus.txt not found.")
        sys.exit(1)

    with open(raw_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    random.seed(42)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    sentence_counts = Counter(lines)

    unique_sents = list(dict.fromkeys(lines))
    random.shuffle(unique_sents)
    n_unique = len(unique_sents)
    train_unique = unique_sents[: int(n_unique * 0.8)]
    val_unique = unique_sents[int(n_unique * 0.8) : int(n_unique * 0.9)]
    test_unique = unique_sents[int(n_unique * 0.9) :]

    def expand_with_duplicates(unique_list):
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
    # Tokenizer
    # ------------------------------------------------------------------
    print(f"\n[2] Loading tokenizer...")
    log_write(log_file, "\n[2] Loading tokenizer...")
    tokenizer_path = PROJECT_ROOT / config["paths"]["tokenizer_path"]
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Merge rules: {len(tokenizer.merge_rules)}")
    log_write(
        log_file,
        f"  Vocab: {len(tokenizer.vocab)}, Merge rules: {len(tokenizer.merge_rules)}",
    )

    # ------------------------------------------------------------------
    # Model creation (dropout=0.1, same as A v2)
    # ------------------------------------------------------------------
    print(f"\n[3] Creating model (dropout=0.1, random init)...")
    log_write(log_file, "\n[3] Creating model (dropout=0.1, random init)...")
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
    # Dataloaders
    # ------------------------------------------------------------------
    print(f"\n[4] Building dataloaders and training...")
    log_write(log_file, "\n[4] Building dataloaders and training...")
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
    log_write(
        log_file,
        f"  Train chunks: {len(dataloaders['train'].dataset)}",
    )
    log_write(
        log_file, f"  Val   chunks: {len(dataloaders['val'].dataset)}"
    )
    log_write(
        log_file,
        f"  Test  chunks: {len(dataloaders['test'].dataset)}",
    )

    # ------------------------------------------------------------------
    # Training
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
            f"  step={entry['step']:>4}",
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
    log_write(
        log_file, f"  Final val loss: {summary.get('final_val_loss', 'N/A'):.6f}"
    )
    log_write(log_file, f"  Params: {summary['params']:,}")

    # ------------------------------------------------------------------
    # Final evaluation
    # ------------------------------------------------------------------
    print(f"\n[5] Final evaluation (A-group re-eval + B2c-group test)...")
    log_write(log_file, "\n[5] Final evaluation...")

    # --- A-group: re-evaluate step200 checkpoint ---
    a_ckpt = PROJECT_ROOT / "checkpoints_v2" / "checkpoint_step200.pt"
    m_a, _, _, meta_a = load_checkpoint(a_ckpt, device)
    val_a = evaluate(m_a, dataloaders["val"], device)
    test_a = evaluate(m_a, dataloaders["test"], device)
    print(
        f"  A-group (step200): val_loss={val_a['loss']:.6f}  "
        f"test_loss={test_a['loss']:.6f}  test_ppl={test_a['perplexity']:.4f}"
    )
    log_write(
        log_file,
        f"  A-group step200: val={val_a['loss']:.6f}  test={test_a['loss']:.6f}  test_ppl={test_a['perplexity']:.4f}",
    )

    # --- B2c-group: find best checkpoint by val_loss ---
    ckpt_files = sorted(checkpoint_dir.glob("checkpoint_step*.pt"))
    best_step = None
    best_val_loss = float("inf")
    best_ckpt_path = None

    for ckpt in ckpt_files:
        m_tmp, _, _, meta_tmp = load_checkpoint(ckpt, device)
        val_tmp = evaluate(m_tmp, dataloaders["val"], device)
        step = meta_tmp["step"]
        print(f"    B2c step{step}: val_loss={val_tmp['loss']:.6f}")
        log_write(log_file, f"    B2c step{step}: val_loss={val_tmp['loss']:.6f}")
        if val_tmp["loss"] < best_val_loss:
            best_val_loss = val_tmp["loss"]
            best_step = step
            best_ckpt_path = ckpt

    # Evaluate B2c best checkpoint on test set
    m_b, _, _, meta_b = load_checkpoint(best_ckpt_path, device)
    val_b = evaluate(m_b, dataloaders["val"], device)
    test_b = evaluate(m_b, dataloaders["test"], device)
    print(
        f"  B2c-group (step{best_step}): val_loss={val_b['loss']:.6f}  "
        f"test_loss={test_b['loss']:.6f}  test_ppl={test_b['perplexity']:.4f}"
    )
    log_write(
        log_file,
        f"  B2c-group step{best_step}: val={val_b['loss']:.6f}  test={test_b['loss']:.6f}  test_ppl={test_b['perplexity']:.4f}",
    )

    # Also record step=200 specific metrics for direct comparison with A v2
    ckpt_200 = checkpoint_dir / "checkpoint_step200.pt"
    if ckpt_200.exists():
        m_200, _, _, meta_200 = load_checkpoint(ckpt_200, device)
        val_200 = evaluate(m_200, dataloaders["val"], device)
        test_200 = evaluate(m_200, dataloaders["test"], device)
        print(
            f"  B2c step200 (intermediate): val_loss={val_200['loss']:.6f}  "
            f"test_loss={test_200['loss']:.6f}  test_ppl={test_200['perplexity']:.4f}"
        )
        log_write(
            log_file,
            f"  B2c step200 intermediate: val={val_200['loss']:.6f}  test={test_200['loss']:.6f}  test_ppl={test_200['perplexity']:.4f}",
        )

    # --- Generation tests ---
    prompts = ["the cat", "the light", "a small bird"]
    run_generate_test(
        m_a, tokenizer, prompts, [0.0, 1.0],
        max_new_tokens=20,
        label="A-group (step200, dropout=0.1)",
        log_file=log_file,
    )
    run_generate_test(
        m_b, tokenizer, prompts, [0.0, 1.0],
        max_new_tokens=20,
        label=f"B2c-group (step{best_step}, dropout=0.1, {max_steps} steps)",
        log_file=log_file,
    )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n" + "=" * 60)
    print("B2c EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"  Config:               smoke_b2c_500steps.yaml")
    print(f"  Dropout (B2c):        {model_config.dropout}")
    print(f"  Max steps:            {max_steps}")
    print(f"  Checkpoints dir:      {checkpoint_dir}")
    print(f"  Log file:             {log_file}")
    print(f"  Training time:        {elapsed:.2f}s")
    print(f"  Best B2c step:        {best_step}  (val_loss={best_val_loss:.6f})")
    print(f"  A-group test_loss:    {test_a['loss']:.6f}")
    print(f"  B2c-group test_loss:  {test_b['loss']:.6f}")
    diff = test_b["loss"] - test_a["loss"]
    print(f"  Delta (B2c - A):      {diff:+.6f}")
    print("=" * 60)
    log_write(log_file, "\n" + "=" * 60)
    log_write(log_file, "B2c EXPERIMENT COMPLETE")
    log_write(log_file, f"  Best B2c step: {best_step}, val_loss={best_val_loss:.6f}")
    log_write(log_file, f"  A-group test_loss: {test_a['loss']:.6f}")
    log_write(log_file, f"  B2c-group test_loss: {test_b['loss']:.6f}")
    log_write(log_file, f"  Delta (B2c - A): {diff:+.6f}")
    log_write(log_file, "=" * 60)


if __name__ == "__main__":
    main()
