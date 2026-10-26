"""
Training loop for AronaLM.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
from tqdm import tqdm

from aronalm.model import AronaLM, TransformerConfig
from aronalm.evaluator import evaluate


class Trainer:
    """Simple trainer for AronaLM with progress tracking."""

    def __init__(
        self,
        model: AronaLM,
        train_loader,
        val_loader,
        config: TransformerConfig,
        output_dir: Path,
        device: torch.device,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        log_interval: int = 10,
        eval_interval: int = 50,
        max_steps: int = 200,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.output_dir = output_dir
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.max_steps = max_steps

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps
        )

        self.training_log: list[dict] = []
        self.start_time = None

    def train(self) -> dict:
        """Run training loop. Returns summary dict with final stats."""
        self.model.to(self.device)
        self.start_time = time.time()

        pbar = tqdm(
            range(1, self.max_steps + 1),
            desc="Training",
            unit="steps",
            initial=0,
            total=self.max_steps,
        )

        for step in pbar:
            self.model.train()
            total_batch_loss = 0.0
            n_batches = 0

            for input_ids in self.train_loader:
                input_ids = input_ids.to(self.device)

                self.optimizer.zero_grad()
                loss = self.model.compute_loss(input_ids)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                self.scheduler.step()

                total_batch_loss += loss.item()
                n_batches += 1

            avg_train_loss = total_batch_loss / max(n_batches, 1)

            # Logging
            if step % self.log_interval == 0 or step == 1:
                elapsed = time.time() - self.start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                log_entry = {
                    "step": step,
                    "train_loss": avg_train_loss,
                    "elapsed_sec": elapsed,
                    "steps_per_sec": steps_per_sec,
                }
                self.training_log.append(log_entry)
                pbar.set_postfix(loss=f"{avg_train_loss:.4f}")

            # Evaluation
            if step % self.eval_interval == 0 or step == self.max_steps:
                val_metrics = evaluate(self.model, self.val_loader, self.device)
                log_entry["val_loss"] = val_metrics["loss"]
                log_entry["val_perplexity"] = val_metrics["perplexity"]
                self.training_log.append(log_entry)
                pbar.set_postfix(
                    loss=f"{avg_train_loss:.4f}",
                    val_loss=f"{val_metrics['loss']:.4f}",
                )

                # Save checkpoint
                ckpt_path = self.output_dir / f"checkpoint_step{step}.pt"
                torch.save(
                    {
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "step": step,
                        "config": {
                            "vocab_size": self.config.vocab_size,
                            "max_seq_len": self.config.max_seq_len,
                            "d_model": self.config.d_model,
                            "n_heads": self.config.n_heads,
                            "n_layers": self.config.n_layers,
                            "ffn_dim": self.config.ffn_dim,
                            "dropout": self.config.dropout,
                            "use_bias_in_proj": self.config.use_bias_in_proj,
                            "use_bias_in_ffn": self.config.use_bias_in_ffn,
                            "use_bias_in_output": self.config.use_bias_in_output,
                            "use_bias_in_layernorm": self.config.use_bias_in_layernorm,
                        },
                        "seed": 42,
                    },
                    ckpt_path,
                )

        elapsed = time.time() - self.start_time
        summary = {
            "final_train_loss": self.training_log[-1]["train_loss"] if self.training_log else None,
            "final_val_loss": self.training_log[-1].get("val_loss"),
            "final_val_perplexity": self.training_log[-1].get("val_perplexity"),
            "total_steps": self.max_steps,
            "elapsed_sec": elapsed,
            "steps_per_sec": self.max_steps / elapsed if elapsed > 0 else 0,
            "params": self.model.count_parameters(),
        }
        return summary
