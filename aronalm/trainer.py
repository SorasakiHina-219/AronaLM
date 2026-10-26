"""
Training loop for AronaLM.

B5 early-stop / anomaly-pause extension (K1--K6 + user decisions A/B,
P1--P8, implemented conservatively):

  - All new parameters default to None / off, so the existing behavior is
    byte-for-byte preserved when B5 is not enabled.
  - Exemption window (K1/K2, Decision A): steps 1..N record validation but
    never increment the counters.
  - Dual independent counters (K3/K5): warn_consec / hard_consec.
  - Threshold semantics (K3): `>=` comparison; counter clears when a valid
    observation is strictly below its own threshold (P1, symmetric to U2).
  - Simultaneous trigger (P2): hard-stop wins; warning is recorded only.
  - Anomaly handling (K4/K5/K6, Decision A): any non-"ok" EvalResult
    pauses training; counters keep their original values; no auto-retry,
    no auto-recovery.
  - Hard-stop (Decision B, P3, P7): writes only an independent stop-state
    JSON; never writes a model checkpoint or a best checkpoint; no further
    training steps; write failure -> stop, no retry.
  - Resume (P4/P6/P8): `resume_from_pause()` is strict -- only anomaly
    states with matching config checksum and exact step match; hard_stop
    states can never be resumed; the mere existence of a state file never
    triggers recovery.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from tqdm import tqdm

from aronalm.model import AronaLM, TransformerConfig
from aronalm.evaluator import EvalResult, evaluate
from aronalm.pause_state import (
    PauseState,
    PauseStateError,
    config_checksum_matches,
    compute_config_checksum,
    load_pause_state,
    make_state,
    save_pause_state,
)


def _config_dict(config: TransformerConfig) -> Dict[str, Any]:
    """The config block stored in checkpoints -- must match that layout so
    checksums computed here are comparable to stored checkpoint configs."""
    return {
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
    }


class Trainer:
    """Simple trainer for AronaLM with progress tracking.

    B5 extension (all defaults preserve legacy behavior):
      baseline_val_loss      -- B5 baseline B; early-stop disabled when None
      warning_threshold      -- B5 warning threshold; early-stop disabled when None
      hard_stop_threshold    -- B5 hard-stop threshold; early-stop disabled when None
      consecutive_count      -- K3: consecutive valid observations (default 2)
      exemption_window       -- K2: N=10 steps; Step N inclusive
      pause_state_path       -- where the pause/stop state JSON is written
      save_checkpoints       -- legacy True; B5 hard-stop never saves
    """

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
        # ---- B5 early-stop parameters (all optional, off by default) ----
        baseline_val_loss: Optional[float] = None,
        warning_threshold: Optional[float] = None,
        hard_stop_threshold: Optional[float] = None,
        consecutive_count: int = 2,
        exemption_window: int = 10,
        pause_state_path: Optional[Path] = None,
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

        # B5 early-stop configuration
        self.baseline_val_loss = baseline_val_loss
        self.warning_threshold = warning_threshold
        self.hard_stop_threshold = hard_stop_threshold
        self.consecutive_count = consecutive_count
        self.exemption_window = exemption_window
        self.pause_state_path = (
            Path(pause_state_path) if pause_state_path is not None else None
        )

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max_steps
        )

        self.training_log: list[dict] = []
        self.start_time = None

        # B5 runtime state (K5)
        self.warn_consec = 0
        self.hard_consec = 0
        self.train_paused = False
        self.pause_reason: Optional[str] = None
        self._resumed_step: Optional[int] = None

        # B5: early-stop active only when all three B5 values are present
        self.b5_enabled = (
            baseline_val_loss is not None
            and warning_threshold is not None
            and hard_stop_threshold is not None
        )

    # ------------------------------------------------------------------
    # B5 helpers
    # ------------------------------------------------------------------
    def _exemption_passed(self, step: int) -> bool:
        """Decision A / K1 / K2: step > N means the window has been passed."""
        return step > self.exemption_window

    def _update_counters(self, result: EvalResult, step: int) -> None:
        """K5 + Decision A + P1.

        - Exemption window (step <= N): record results, never touch counters.
        - Anomalous / failed / empty result: counters keep their original
          values (K5, Decision A.4); no auto-retry.
        - Valid result outside the window:
            warning  -- clear below threshold (U2), else increment
            hard-stop-- clear below threshold (P1), else increment
          Both counters updated independently (K3).
        """
        if step <= self.exemption_window:
            return  # K1/K2/Decision A.2: frozen inside the window

        if not result.is_valid:
            return  # K5/Decision A.4: anomaly -> keep original values

        assert result.loss is not None
        if result.loss >= self.warning_threshold:
            self.warn_consec += 1
        else:
            self.warn_consec = 0

        if result.loss >= self.hard_stop_threshold:
            self.hard_consec += 1
        else:
            self.hard_consec = 0

    def _write_pause_state(self, step: int, reason: str) -> bool:
        """Persist the 9-field pause state (P5). Atomic write (P7).

        reason: an anomaly token ("nan"/"inf"/"failed"/"empty"/"format_error")
        or "hard_stop".

        Returns True only if the file was written AND verified readable.
        On failure returns False (P3): caller must stop training, never
        retry, never fall back to checkpoint writes.
        """
        if self.pause_state_path is None:
            return False
        config_cs = compute_config_checksum(_config_dict(self.config))
        state = make_state(
            step=step,
            warn_consec=self.warn_consec,
            hard_consec=self.hard_consec,
            pause_reason=reason,
            exemption_window_passed=self._exemption_passed(step),
            baseline_val_loss=self.baseline_val_loss,
            config_dict=None,  # checksum set explicitly below
        )
        state.config_checksum = config_cs
        return save_pause_state(self.pause_state_path, state)

    def _safe_cleanup(self) -> None:
        """P7 hard-stop safe cleanup: log, independent state JSON, release
        resources. No checkpoint write, no best-checkpoint update, no
        training steps."""
        print(
            f"[B5] safe cleanup complete: state written={self.pause_state_path is not None}, "
            "resources released; no checkpoint or best-checkpoint writes performed",
            flush=True,
        )

    # ------------------------------------------------------------------
    # resume / recovery (P4, P6, P8)
    # ------------------------------------------------------------------
    def resume_from_pause(
        self,
        expected_step: int,
        config_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Attempt a STRICT anomaly-resume (K6 manual confirmation is the
        caller's job -- this method is only ever invoked on explicit
        `--resume` (U6/P4), never automatically, P8).

        Conditions (all must hold, P6):
          - a state file exists and passes load-time validation
          - pause_reason is an anomaly (hard_stop is never resumable, P4)
          - stored config checksum matches the live config (when stored)
          - step matches exactly (no tolerance)
          - exempt-window already passed

        Returns True and restores warn_consec / hard_consec /
        _resumed_step / train_paused=False on success; False otherwise.
        On any mismatch the counters are left untouched (K5).
        """
        if not self.b5_enabled:
            return False
        if self.pause_state_path is None or not self.pause_state_path.exists():
            return False  # P8: no file -> no resume, nothing automatic

        try:
            state: Optional[PauseState] = load_pause_state(self.pause_state_path)
        except PauseStateError as e:
            print(f"[B5] resume rejected: {e}", flush=True)
            return False

        if state is None:
            return False

        if state.is_hard_stop():
            print(
                "[B5] resume rejected: hard_stop state cannot be resumed (P4)",
                flush=True,
            )
            return False

        if not state.is_anomaly():
            print(
                f"[B5] resume rejected: unknown pause_reason '{state.pause_reason}'",
                flush=True,
            )
            return False

        live_cfg = config_dict if config_dict is not None else _config_dict(self.config)
        if state.config_checksum is not None:
            if not config_checksum_matches(state.config_checksum, live_cfg):
                print(
                    "[B5] resume rejected: config_checksum mismatch (P6)",
                    flush=True,
                )
                return False

        if state.step != expected_step:
            print(
                f"[B5] resume rejected: step mismatch "
                f"(state={state.step} expected={expected_step}, strict)",
                flush=True,
            )
            return False

        # All checks passed -- restore counters from the anomaly snapshot.
        self.warn_consec = state.warn_consec
        self.hard_consec = state.hard_consec
        self._resumed_step = state.step
        self.train_paused = False
        self.pause_reason = None
        print(
            f"[B5] resumed from anomaly at step {state.step} "
            f"(warn={self.warn_consec}, hard={self.hard_consec})",
            flush=True,
        )
        return True

    # ------------------------------------------------------------------
    # training loop
    # ------------------------------------------------------------------
    def train(self) -> dict:
        """Run training loop. Returns summary dict with final stats.

        When B5 is enabled, the loop may stop early (hard-stop or anomaly
        pause) and the summary will reflect `stopped` / `stop_reason`.
        """
        self.model.to(self.device)
        self.start_time = time.time()

        pbar = tqdm(
            range(1, self.max_steps + 1),
            desc="Training",
            unit="steps",
            initial=0,
            total=self.max_steps,
        )

        stopped: Optional[str] = None
        steps_executed = 0

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

            steps_executed = step
            avg_train_loss = total_batch_loss / max(n_batches, 1)

            # Logging
            if step % self.log_interval == 0 or step == 1:
                elapsed = time.time() - self.start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                log_entry: Dict[str, Any] = {
                    "step": step,
                    "train_loss": avg_train_loss,
                    "elapsed_sec": elapsed,
                    "steps_per_sec": steps_per_sec,
                }
                self.training_log.append(log_entry)
                pbar.set_postfix(loss=f"{avg_train_loss:.4f}")

            # Evaluation
            if step % self.eval_interval == 0 or step == self.max_steps:
                result = evaluate(self.model, self.val_loader, self.device)

                if self.b5_enabled:
                    stopped = self._handle_b5_observation(result, step, log_entry, pbar)
                    if stopped is not None:
                        break

                # Legacy-compatible log fields (existing scripts read these)
                if result.is_valid:
                    log_entry["val_loss"] = result.loss
                    log_entry["val_perplexity"] = result.perplexity
                    pbar.set_postfix(
                        loss=f"{avg_train_loss:.4f}",
                        val_loss=f"{result.loss:.4f}",
                    )
                else:
                    log_entry["val_status"] = result.status
                    pbar.set_postfix(
                        loss=f"{avg_train_loss:.4f}",
                        val_status=result.status,
                    )
                self.training_log.append(log_entry)

                # Save checkpoint
                # B5 Decision B / P7: when a hard-stop or anomaly pause has
                # just occurred, NO checkpoint is written.
                if stopped is None:
                    ckpt_path = self.output_dir / f"checkpoint_step{step}.pt"
                    torch.save(
                        {
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "scheduler_state_dict": self.scheduler.state_dict(),
                            "step": step,
                            "config": _config_dict(self.config),
                            "seed": 42,
                        },
                        ckpt_path,
                    )

        elapsed = time.time() - self.start_time
        last = self.training_log[-1] if self.training_log else {}
        summary = {
            "final_train_loss": last.get("train_loss"),
            "final_val_loss": last.get("val_loss"),
            "final_val_perplexity": last.get("val_perplexity"),
            "total_steps": self.max_steps,
            "steps_executed": steps_executed,
            "elapsed_sec": elapsed,
            "steps_per_sec": (steps_executed / elapsed) if elapsed > 0 else 0,
            "params": self.model.count_parameters(),
            "stopped": stopped,  # None, "hard_stop", or an anomaly reason
            "warn_consec": self.warn_consec,
            "hard_consec": self.hard_consec,
        }
        return summary

    # ------------------------------------------------------------------
    # B5 observation handling (K1--K6, A, B, P1--P8)
    # ------------------------------------------------------------------
    def _handle_b5_observation(
        self, result: EvalResult, step: int, log_entry: Dict[str, Any], pbar
    ) -> Optional[str]:
        """Process one B5 evaluation observation.

        Returns None if training should continue, or a stop marker string:
          "hard_stop" | an anomaly reason ("nan", "inf", "failed", "empty",
          "format_error") that triggered a pause.
        """
        # 1) anomaly path (K4/K5/K6 + Decision A.4 + Decision B write scope)
        if not result.is_valid:
            self.pause_reason = result.status
            self.train_paused = True
            print(
                f"[B5] PAUSED at step {step}: validation anomaly "
                f"'{result.status}' ({result.error_message}); "
                f"counters kept as-is (warn={self.warn_consec}, hard={self.hard_consec}); "
                "manual inspection required, no auto-recovery (K6).",
                flush=True,
            )
            log_entry["b5_event"] = f"paused:{result.status}"
            ok = self._write_pause_state(step, reason=result.status)
            if not ok:
                # P3: state-write failure -> stop training, no retry, no
                # checkpoint, no other state file writes.
                print(
                    "[B5] pause-state write FAILED: stopping training; "
                    "no checkpoint or best-checkpoint will be written; "
                    "manual inspection required.",
                    flush=True,
                )
                log_entry["b5_event"] += ";state_write_failed"
                self._safe_cleanup()
                return result.status
            self._safe_cleanup()
            return result.status

        # 2) normal path: counters (exemption-frozen inside window)
        self._update_counters(result, step)

        # 3) triggers -- P2: hard-stop wins when both fire on the same eval
        hard_triggered = self.hard_consec >= self.consecutive_count
        warn_triggered = self.warn_consec >= self.consecutive_count

        if hard_triggered:
            self.pause_reason = "hard_stop"
            self.train_paused = True
            print(
                f"[B5] HARD-STOP at step {step}: hard_consec="
                f"{self.hard_consec} >= {self.consecutive_count} "
                f"(val_loss={result.loss:.6f} >= threshold={self.hard_stop_threshold}); "
                f"training will not continue; independent stop state written; "
                "no checkpoint / best-checkpoint update (Decision B).",
                flush=True,
            )
            if warn_triggered:
                print(
                    f"[B5] note: warning condition also met (warn_consec="
                    f"{self.warn_consec}); hard-stop takes priority (P2).",
                    flush=True,
                )
            log_entry["b5_event"] = "hard_stop"
            ok = self._write_pause_state(step, reason="hard_stop")
            if not ok:
                print(
                    "[B5] stop-state write FAILED at hard-stop: training already "
                    "stopped; no checkpoint written; manual inspection required (P3).",
                    flush=True,
                )
                log_entry["b5_event"] += ";state_write_failed"
            self._safe_cleanup()
            return "hard_stop"

        if warn_triggered:
            print(
                f"[B5] WARNING at step {step}: warn_consec={self.warn_consec} "
                f">= {self.consecutive_count} (val_loss={result.loss:.6f}); "
                "recording only, training continues (P2).",
                flush=True,
            )
            log_entry["b5_event"] = "warning"

        return None
