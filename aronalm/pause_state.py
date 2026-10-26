"""
Pause/resume state management for B5 early-stop.

Implements the user-confirmed 9-field pause state format (P5):
  required: version, step, warn_consec, hard_consec, pause_reason,
            exemption_window_passed
  optional: baseline_val_loss, config_checksum, timestamp

Safety semantics (P3/P6/P7/P8):
  - Atomic write: write to temp file then os.replace, so a partial file
    can never be observed by readers.
  - Strict validation on read: missing/corrupt/unknown-version files are
    rejected, never repaired, never migrated.
  - Pre-recovery identity check (P6): config checksum + strict step match
    + vocab_size / max_seq_len consistency. Any mismatch -> reject.
  - Reading a state file is NOT a recovery authorization (P8). Only an
    explicit, validated resume call may restore counters; a state file
    whose pause_reason is "hard_stop" can never be resumed.
  - If a write fails, the caller must stop training (P3) -- no retry,
    no checkpoint fallback, no other state file writes.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

STATE_SCHEMA_VERSION = "1.0"

REQUIRED_FIELDS = (
    "version",
    "step",
    "warn_consec",
    "hard_consec",
    "pause_reason",
    "exemption_window_passed",
)
OPTIONAL_FIELDS = ("baseline_val_loss", "config_checksum", "timestamp")
ALL_FIELDS = REQUIRED_FIELDS + OPTIONAL_FIELDS

# Valid anomaly reasons (K4). "hard_stop" is a stop marker, not an anomaly.
ANOMALY_REASONS = ("nan", "inf", "failed", "empty", "format_error")
STOP_REASONS = ("hard_stop",)


class PauseStateError(Exception):
    """Raised when a pause state file is missing, corrupt, or incompatible."""


@dataclass
class PauseState:
    """Serializable pause state for B5 early-stop / anomaly handling."""

    step: int
    warn_consec: int
    hard_consec: int
    pause_reason: str
    exemption_window_passed: bool = False
    # optional
    baseline_val_loss: float | None = None
    config_checksum: str | None = None
    timestamp: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "version": STATE_SCHEMA_VERSION,
            "step": self.step,
            "warn_consec": self.warn_consec,
            "hard_consec": self.hard_consec,
            "pause_reason": self.pause_reason,
            "exemption_window_passed": self.exemption_window_passed,
        }
        if self.baseline_val_loss is not None:
            d["baseline_val_loss"] = self.baseline_val_loss
        if self.config_checksum is not None:
            d["config_checksum"] = self.config_checksum
        if self.timestamp is not None:
            d["timestamp"] = self.timestamp
        return d

    def is_hard_stop(self) -> bool:
        return self.pause_reason in STOP_REASONS

    def is_anomaly(self) -> bool:
        return self.pause_reason in ANOMALY_REASONS


def save_pause_state(path: Path, state: PauseState) -> bool:
    """Atomically persist a PauseState to `path`.

    Returns True on success. On any failure (IO error, permissions, path
    issues) returns False WITHOUT retrying and WITHOUT touching any other
    file (P3): the caller is responsible for stopping training.
    Never raises to the caller.
    """
    payload = json.dumps(state.to_dict(), indent=2, sort_keys=True)
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            # Atomic replace -- readers never observe a partial file.
            os.replace(tmp_name, str(path))
        finally:
            if os.path.exists(tmp_name):
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
        # Verification read (P7): only claim success if the file is fully
        # readable and parseable.
        with open(path, "r", encoding="utf-8") as f:
            json.load(f)
        return True
    except Exception:
        return False


def load_pause_state(path: Path) -> PauseState | None:
    """Load and validate a pause state file.

    Returns the PauseState, or None if the file is missing. Raises
    PauseStateError for corrupt / missing-fields / unknown-version files.
    This is a read-only operation and grants NO recovery authorization (P8).
    """
    path = Path(path)
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        raise PauseStateError(f"corrupt pause state file {path}: {e}") from e

    if not isinstance(data, dict):
        raise PauseStateError(f"pause state {path} is not a JSON object")

    for field in REQUIRED_FIELDS:
        if field not in data:
            raise PauseStateError(f"pause state {path} missing required field '{field}'")

    if data["version"] != STATE_SCHEMA_VERSION:
        raise PauseStateError(
            f"pause state {path} has incompatible version "
            f"'{data['version']}' (expected '{STATE_SCHEMA_VERSION}')"
        )

    reason = data["pause_reason"]
    if reason not in ANOMALY_REASONS + STOP_REASONS:
        raise PauseStateError(f"pause state {path} has unknown pause_reason '{reason}'")

    step = data["step"]
    if not isinstance(step, int) or isinstance(step, bool):
        raise PauseStateError(f"pause state {path}: 'step' must be int")

    warn = data["warn_consec"]
    hard = data["hard_consec"]
    for name, val in (("warn_consec", warn), ("hard_consec", hard)):
        if not isinstance(val, int) or isinstance(val, bool) or val < 0:
            raise PauseStateError(f"pause state {path}: '{name}' must be non-negative int")

    if not isinstance(data["exemption_window_passed"], bool):
        raise PauseStateError(
            f"pause state {path}: 'exemption_window_passed' must be bool"
        )

    for field in ("baseline_val_loss",):
        if field in data and data[field] is not None:
            if not isinstance(data[field], (int, float)):
                raise PauseStateError(f"pause state {path}: '{field}' must be numeric")

    if "config_checksum" in data and data["config_checksum"] is not None:
        if not isinstance(data["config_checksum"], str):
            raise PauseStateError(f"pause state {path}: 'config_checksum' must be str")

    if "timestamp" in data and data["timestamp"] is not None:
        if not isinstance(data["timestamp"], str):
            raise PauseStateError(f"pause state {path}: 'timestamp' must be str")

    return PauseState(
        step=step,
        warn_consec=warn,
        hard_consec=hard,
        pause_reason=reason,
        exemption_window_passed=data["exemption_window_passed"],
        baseline_val_loss=data.get("baseline_val_loss"),
        config_checksum=data.get("config_checksum"),
        timestamp=data.get("timestamp"),
    )


def compute_config_checksum(config_dict: dict[str, Any]) -> str:
    """SHA-256 over the canonical JSON of the model config (P6)."""
    canonical = json.dumps(config_dict, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def config_checksum_matches(
    stored: str | None,
    config_dict: dict[str, Any],
) -> bool:
    """P6: verify the current config against the stored checksum.

    A None stored checksum (optional field not set) is treated as
    'not verifiable' -> False for strict-match recovery, unless the caller
    explicitly allows missing checksums.
    """
    if not stored:
        return False
    return stored == compute_config_checksum(config_dict)


def validate_resume_identity(
    state: PauseState,
    expected_step: int,
    config_dict: dict[str, Any],
) -> tuple[bool, str]:
    """P6: pre-recovery identity checks. Returns (ok, reason).

    Strict rules:
      - state.pause_reason must be an anomaly (hard_stop can never resume).
      - config checksum must match the stored value when present.
      - step must match exactly (no tolerance, per user confirmation).
      - vocab_size / max_seq_len must equal the stored config when the
        state carried a config checksum -- verified through the config_dict
        supplied by the caller, which must be the live config.
    """
    if state.is_hard_stop():
        return False, "hard_stop state cannot be resumed (P4)"
    if not state.is_anomaly():
        return False, f"pause_reason '{state.pause_reason}' is not a resolvable anomaly"

    if state.config_checksum is not None:
        if not config_checksum_matches(state.config_checksum, config_dict):
            return False, "config_checksum mismatch (P6)"

    if state.step != expected_step:
        return False, (
            f"step mismatch: state={state.step} expected={expected_step} (strict)"
        )

    return True, "ok"


def make_state(
    step: int,
    warn_consec: int,
    hard_consec: int,
    pause_reason: str,
    exemption_window_passed: bool,
    baseline_val_loss: float | None = None,
    config_dict: dict[str, Any] | None = None,
) -> PauseState:
    """Convenience constructor with a UTC ISO-8601 timestamp."""
    checksum = None
    if config_dict is not None:
        checksum = compute_config_checksum(config_dict)
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return PauseState(
        step=step,
        warn_consec=warn_consec,
        hard_consec=hard_consec,
        pause_reason=pause_reason,
        exemption_window_passed=exemption_window_passed,
        baseline_val_loss=baseline_val_loss,
        config_checksum=checksum,
        timestamp=ts,
    )
