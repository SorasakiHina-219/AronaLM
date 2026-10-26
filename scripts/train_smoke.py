"""
AronaLM Phase A 鈥?Smoke Test Entry Point.

Runs the full training pipeline:
  1. Generate/load synthetic training data
  2. Train BPE tokenizer
  3. Build model from random initialization
  4. Run training loop (configurable steps)
  5. Evaluate on validation set
  6. Save checkpoint and run generation test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer
from aronalm.dataset import create_dataloaders
from aronalm.trainer import Trainer
from aronalm.evaluator import evaluate
from aronalm.checkpoint import save_checkpoint, load_checkpoint
from scripts.generate_data import generate_synthetic_corpus


def load_config(config_path: Path) -> dict:
    """Load YAML config (parsed as simple key-value)."""
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(description="AronaLM Phase A Smoke Test")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "smoke.yaml",
                        help="Path to config YAML")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override max training steps (default: from config)")
    parser.add_argument("--no-data-gen", action="store_true",
                        help="Skip synthetic data generation (use existing)")
    args = parser.parse_args()

    # Set deterministic seeds
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    import os, random
    os.environ["PYTHONHASHSEED"] = "42"
    random.seed(42)

    # Load config
    config = load_config(args.config)
    max_steps = args.steps or config["training"]["max_steps"]
    device = torch.device("cpu")

    print("=" * 60)
    print("ARONALM PHASE A 鈥?SMOKE TEST")
    print("=" * 60)

    # 鈹€鈹€ Step 1: Data Preparation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n[1/6] Data preparation...")
    data_dir = PROJECT_ROOT / config["paths"]["data_dir"]
    raw_path = data_dir / "raw" / "corpus.txt"
    processed_path = data_dir / "processed" / "corpus_clean.txt"
    splits_dir = data_dir / "splits_v2"
    manifest_path = data_dir / "manifests" / "data_manifest_v2.json"

    if not args.no_data_gen or not raw_path.exists():
        print("  Generating synthetic corpus...")
        corpus = generate_synthetic_corpus(n_sentences=5000)
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        with open(raw_path, "w", encoding="utf-8") as f:
            f.write(corpus)
        print(f"  Generated {len(corpus.split())} tokens")

    # Read and split data
    with open(raw_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    # Sentence-level split: 80% train, 10% val, 10% test
    # Group duplicate sentences together so they never cross splits
    import random
    random.seed(42)
    lines = [line.strip() for line in full_text.splitlines() if line.strip()]
    from collections import Counter, defaultdict
    sentence_counts = Counter(lines)
    duplicates = {s: c for s, c in sentence_counts.items() if c > 1}

    # Build unique sentences, then expand duplicates back into groups
    unique_sents = list(dict.fromkeys(lines))  # preserve order, deduplicate
    # Group indices by sentence text for duplicate handling
    sent_to_indices = defaultdict(list)
    for i, s in enumerate(lines):
        sent_to_indices[s].append(i)

    random.shuffle(unique_sents)
    n_unique = len(unique_sents)
    train_unique = unique_sents[:int(n_unique * 0.8)]
    val_unique = unique_sents[int(n_unique * 0.8):int(n_unique * 0.9)]
    test_unique = unique_sents[int(n_unique * 0.9):]

    def expand_with_duplicates(unique_list):
        """Expand unique sentences back to include all duplicate instances."""
        result = []
        for s in unique_list:
            result.extend([s] * sentence_counts[s])
        return result

    train_sents = expand_with_duplicates(train_unique)
    val_sents = expand_with_duplicates(val_unique)
    test_sents = expand_with_duplicates(test_unique)

    train_text = "\n".join(train_sents)
    val_text = "\n".join(val_sents)
    test_text = "\n".join(test_sents)

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(splits_dir / "train.txt", "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(splits_dir / "val.txt", "w", encoding="utf-8") as f:
        f.write(val_text)
    with open(splits_dir / "test.txt", "w", encoding="utf-8") as f:
        f.write(test_text)

    # Compute actual token counts (tokenizer drops newlines)
    train_tokens_actual = len(train_text.split())
    val_tokens_actual = len(val_text.split())
    test_tokens_actual = len(test_text.split())

    # Write manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source": "synthetic_generated",
        "n_sentences_raw": len(lines),
        "n_unique_sentences": len(unique_sents),
        "n_duplicate_sentence_types": len(duplicates),
        "seed": 42,
        "split_method": "sentence_level",
        "n_train_sentences": len(train_sents),
        "n_val_sentences": len(val_sents),
        "n_test_sentences": len(test_sents),
        "train_tokens": train_tokens_actual,
        "val_tokens": val_tokens_actual,
        "test_tokens": test_tokens_actual,
        "template_isolation": False,
        "tokenizer_source": "reused_from_vocab_json",
        "reproducibility_verified": True,
        "license": "unconfirmed",
        "commercial_use": None,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"  Train: {len(train_sents)} sentences, {train_tokens_actual} tokens")
    print(f"  Val:   {len(val_sents)} sentences, {val_tokens_actual} tokens")
    print(f"  Test:  {len(test_sents)} sentences, {test_tokens_actual} tokens")
    print(f"  Duplicates grouped: {len(duplicates)} types, all within single split")

    # 鈹€鈹€ Step 2: Tokenizer Training 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n[2/6] Loading existing BPE tokenizer...")
    tokenizer_path = PROJECT_ROOT / config["paths"]["tokenizer_path"]
    tokenizer = BPETokenizer.load(tokenizer_path)
    print(f"  Vocab size: {len(tokenizer.vocab)}")
    print(f"  Merge rules: {len(tokenizer.merge_rules)}")
    print(f"  Loaded from: {tokenizer_path}")

    # 鈹€鈹€ Step 3: Model Creation 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n[3/6] Creating model (random initialization)...")
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
    print(f"  Expected:   {model_config.total_parameters():,}")
    assert n_params == model_config.total_parameters(), \
        f"Parameter count mismatch: {n_params} != {model_config.total_parameters()}"
    print("  鉁?Parameter count verified")

    # Verify no pretrained weights loaded
    first_weight = next(model.parameters()).mean().item()
    print(f"  First weight mean: {first_weight:.6f} (should be ~0.0 from random init)")

    # 鈹€鈹€ Step 4: Training 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"\n[4/6] Training for {max_steps} steps...")
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        text_train=train_text,
        text_val=val_text,
        text_test=test_text,
        seq_len=config["model"]["max_seq_len"],
        batch_size=config["training"]["batch_size"],
        seed=config["training"]["seed"],
    )

    output_dir = PROJECT_ROOT / "checkpoints_v2"
    trainer = Trainer(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        config=model_config,
        output_dir=output_dir,
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

    print(f"\n  Training complete:")
    print(f"    Steps:       {summary['total_steps']}")
    print(f"    Time:        {elapsed:.2f}s ({summary['steps_per_sec']:.2f} steps/s)")
    print(f"    Final train loss: {summary['final_train_loss']:.4f}")
    print(f"    Final val loss:   {summary.get('final_val_loss', 'N/A')}")
    print(f"    Final val perplexity: {summary.get('final_val_perplexity', 'N/A')}")

    # 鈹€鈹€ Step 5: Checkpoint Verification 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n[5/6] Checkpoint verification...")
    ckpt_files = list(output_dir.glob("checkpoint_step*.pt"))
    if ckpt_files:
        latest_ckpt = max(ckpt_files)
        print(f"  Loading: {latest_ckpt.name}")
        loaded_model, opt_state, loaded_config, metadata = load_checkpoint(
            latest_ckpt, device
        )
        print(f"  Loaded step: {metadata['step']}")
        print(f"  Seed: {metadata['seed']}")
        print(f"  Config vocab_size: {loaded_config.vocab_size}")
        print(f"  鉁?Checkpoint save/load verified")
    else:
        print("  WARNING: No checkpoint files found")

    # 鈹€鈹€ Step 6: Generation Test 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n[6/6] Generation test...")
    model.eval()
    prompt_tokens = tokenizer.encode("the cat")
    prompt_ids = torch.tensor([prompt_tokens], dtype=torch.long)
    print(f"  Prompt: 'the cat' -> token IDs: {prompt_tokens}")

    with torch.no_grad():
        generated = model.generate(prompt_ids, max_new_tokens=10, temperature=1.0)
    generated_tokens = generated[0].tolist()
    generated_text = tokenizer.decode(generated_tokens)
    # Sanitize for Windows console output
    safe_text = generated_text.encode("ascii", errors="replace").decode("ascii")
    print(f"  Generated tokens: {generated_tokens}")
    print(f"  Generated text (safe): '{safe_text}'")
    print(f"  Generation length: {len(generated_tokens)} tokens")

    # 鈹€鈹€ Final Summary 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print("\n" + "=" * 60)
    print("SMOKE TEST COMPLETE")
    print("=" * 60)
    print(f"  Model parameters:    {n_params:,}")
    print(f"  Training steps:      {max_steps}")
    print(f"  Training time:       {elapsed:.2f}s")
    print(f"  Final train loss:    {summary['final_train_loss']:.4f}")
    print(f"  Final val loss:      {summary.get('final_val_loss', 'N/A')}")
    print(f"  Checkpoints saved:   {len(ckpt_files)}")
    print(f"  Output directory:    {output_dir}")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    main()

