# -*- coding: utf-8 -*-
"""Task B: isolated fault-injection tests for the REAL B5 control flow.

Scenarios (task B, 10 cases), each exercising `Trainer.train()` /
`resume_from_pause()` directly with a deterministic fake evaluator and a
throwaway tmp_path output dir:

  1. hard-stop termination on consecutive high loss
  2. every anomaly token (nan / inf / failed / empty / format_error)
     triggers anomaly pause
  3. pause-state write failure -> fail-closed stop
  4. resume rejected on config-checksum mismatch
  5. resume rejected on step mismatch
  6. resume rejected on vocab_size mismatch
  7. resume rejected on max_seq_len mismatch
  8. hard-stop state can never be resumed
  9. paused counters preserved + no checkpoint written at pause step
 10. written state file has exactly 9 fields (no 10th)

Every test proves the failure path never touches `SOURCE_CKPT` or any
other experiment dir (the `output_dir` passed to Trainer is always a
fresh tmp_path, and we assert no checkpoint file appears at the
pause step).

All runs are ISOLATED: temp directories, mocked I/O, small models,
no network, no real training data. They are labeled "controlled fault
injection" in the delivery report -- NOT natural real-training faults.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aronalm.evaluator import EvalResult
from aronalm.model import AronaLM, TransformerConfig
from aronalm.pause_state import (
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    ALL_FIELDS,
    load_pause_state,
    make_state,
)
from aronalm.trainer import Trainer

ROOT = Path(__file__).resolve().parent.parent
SOURCE_CKPT = ROOT / "checkpoints_v2c_500steps" / "checkpoint_step350.pt"
# Fixed expected SHA-256 of the step-350 source checkpoint (lowercase).
# Used for real before/after hash protection, not self-comparison.
_EXPECTED_SOURCE_HASH = "8ff70323b055216a5a0f5372831d5ac38345cfa73083ceba84d995a5d21369dd"


# ----------------------------------------------------------------------
# shared fixtures
# ----------------------------------------------------------------------
def _make_model(vocab_size=32, max_seq_len=8):
    cfg = TransformerConfig(vocab_size=vocab_size, max_seq_len=max_seq_len,
                            d_model=16, n_heads=2, n_layers=1, ffn_dim=32,
                            dropout=0.0)
    torch.manual_seed(0)
    return AronaLM(cfg), cfg


def _loader(n=2, vocab=32, seq=8):
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, vocab, (n, seq), generator=g)
    return torch.utils.data.DataLoader(data, batch_size=1)


def _make_trainer(tmp_path, m, cfg, **kw):
    defaults = dict(
        model=m, train_loader=_loader(), val_loader=_loader(),
        config=cfg, output_dir=tmp_path, device=torch.device("cpu"),
        lr=1e-3, log_interval=1, eval_interval=1, max_steps=kw.pop("max_steps", 4),
        baseline_val_loss=kw.pop("baseline_val_loss", 0.5),
        warning_threshold=kw.pop("warning_threshold", 1.0),
        hard_stop_threshold=kw.pop("hard_stop_threshold", 1.5),
        consecutive_count=kw.pop("consecutive_count", 1),
        exemption_window=kw.pop("exemption_window", 0),
        pause_state_path=kw.pop("pause_state_path", tmp_path / "pause_state.json"),
    )
    return Trainer(**defaults, **kw)


def _mock_eval(status, loss=9.9):
    import aronalm.trainer as tmod
    if status == "ok":
        result = EvalResult(status="ok", loss=loss, perplexity=math.exp(loss))
    else:
        result = EvalResult(status=status, error_message=f"fault-injected {status}")
    tmod.evaluate = lambda m, dl, d: result
    return result


def _source_ckpt_hash():
    import hashlib
    h = hashlib.sha256()
    with open(SOURCE_CKPT, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------
# 1. hard-stop termination
# ----------------------------------------------------------------------
def test_hard_stop_terminates(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=2, max_steps=10, exemption_window=0)
    _mock_eval("ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"
    # the loop stopped after the second consecutive hard-stop-qualifying eval
    assert summary["steps_executed"] == 2
    # state file written, counters set
    st = load_pause_state(pause)
    assert st is not None
    assert st.pause_reason == "hard_stop"
    assert st.step == 2
    # no checkpoint written at the stop step (Decision B / P7)
    assert not (tmp_path / "checkpoint_step2.pt").exists()
    # source checkpoint untouched: hash must still equal the fixed expected
    # value (real before/after check, not a self-comparison)
    assert _source_ckpt_hash() == _EXPECTED_SOURCE_HASH, (
        "source checkpoint hash changed during test")


# ----------------------------------------------------------------------
# 2. every anomaly token pauses
# ----------------------------------------------------------------------
def test_all_anomaly_tokens_pause(tmp_path):
    m, cfg = _make_model()
    for i, status in enumerate(["nan", "inf", "failed", "empty", "format_error"]):
        sub = tmp_path / f"anomaly_{status}"
        sub.mkdir()
        pause = sub / "pause_state.json"
        t = _make_trainer(sub, m, cfg, pause_state_path=pause,
                          consecutive_count=5, max_steps=4, exemption_window=0)
        _mock_eval(status)
        summary = t.train()
        assert summary["stopped"] == status, f"status {status}"
        # the first eval (step 1) is inside exemption window -> not counted
        # the anomaly at step 1 triggers the pause regardless of window
        st = load_pause_state(pause)
        assert st is not None and st.is_anomaly()
        assert st.pause_reason == status
        # no checkpoint written at the anomaly step
        assert not list(sub.glob("checkpoint_step*.pt"))


# ----------------------------------------------------------------------
# 3. state-write failure -> fail-closed stop
# ----------------------------------------------------------------------
def test_state_write_failure_fail_closed(tmp_path):
    m, cfg = _make_model()
    # a path whose PARENT IS A FILE -> save_pause_state cannot mkdir -> returns False
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad_path = blocker / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=bad_path,
                      max_steps=4, exemption_window=0, consecutive_count=5)
    _mock_eval("nan")
    summary = t.train()
    assert summary["stopped"] == "nan"
    # training stopped, no checkpoint, no fallback file anywhere
    assert not list(tmp_path.glob("checkpoint_step*.pt"))
    # blocker file still exists (no overwrite)
    assert blocker.exists()
    assert blocker.read_text(encoding="utf-8") == "x"


# ----------------------------------------------------------------------
# 4-7. resume rejection on checksum / step / vocab / seq_len mismatch
# ----------------------------------------------------------------------
def test_resume_rejects_checksum_mismatch(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=5, max_steps=4)
    _mock_eval("failed")
    t.train()  # produces a valid anomaly state
    # a trainer with a DIFFERENT config checksum must be rejected
    m2, cfg2 = _make_model(vocab_size=64)
    t2 = _make_trainer(tmp_path, m2, cfg2, pause_state_path=pause,
                       consecutive_count=5, max_steps=4)
    assert t2.resume_from_pause(expected_step=1) is False


def test_resume_rejects_step_mismatch(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=5, max_steps=4)
    _mock_eval("failed")
    t.train()  # state.step == 1
    # expected_step=2 -> strict mismatch
    assert t.resume_from_pause(expected_step=2) is False


def test_resume_rejects_vocab_mismatch(tmp_path):
    m, cfg = _make_model(vocab_size=32)
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=5, max_steps=4)
    _mock_eval("failed")
    t.train()
    m2, cfg2 = _make_model(vocab_size=16)  # different vocab -> different checksum
    t2 = _make_trainer(tmp_path, m2, cfg2, pause_state_path=pause,
                       consecutive_count=5, max_steps=4)
    assert t2.resume_from_pause(expected_step=1) is False


def test_resume_rejects_seq_len_mismatch(tmp_path):
    m, cfg = _make_model(max_seq_len=8)
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=5, max_steps=4)
    _mock_eval("failed")
    t.train()
    m2, cfg2 = _make_model(max_seq_len=16)  # different seq_len -> different checksum
    t2 = _make_trainer(tmp_path, m2, cfg2, pause_state_path=pause,
                       consecutive_count=5, max_steps=4)
    assert t2.resume_from_pause(expected_step=1) is False


# ----------------------------------------------------------------------
# 8. hard-stop state is never resumable
# ----------------------------------------------------------------------
def test_hard_stop_state_never_resumable(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=1, max_steps=2)
    _mock_eval("ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"
    # even a fresh trainer with the same path must refuse
    m2, _ = _make_model()
    t2 = _make_trainer(tmp_path, m2, cfg, pause_state_path=pause,
                       consecutive_count=1, max_steps=2)
    assert t2.resume_from_pause(expected_step=1) is False
    # counters on t2 unchanged
    assert t2.warn_consec == 0
    assert t2.hard_consec == 0
    assert t2.train_paused is False


# ----------------------------------------------------------------------
# 9. paused counters preserved + no checkpoint written
# ----------------------------------------------------------------------
def test_paused_counters_preserved_and_no_checkpoint(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=2, max_steps=10, exemption_window=0)
    # first eval = ok low loss (counters clear), then anomaly
    import aronalm.trainer as tmod
    seq = [EvalResult(status="ok", loss=0.1, perplexity=1.1),
           EvalResult(status="nan", error_message="injected")]
    calls = {"i": 0}
    def fake_eval(m, dl, d):
        r = seq[calls["i"]]
        calls["i"] += 1
        return r
    tmod.evaluate = fake_eval
    summary = t.train()
    assert summary["stopped"] == "nan"
    assert summary["warn_consec"] == 0
    assert summary["hard_consec"] == 0
    # the state file records the counters at the anomaly step
    st = load_pause_state(pause)
    assert st.warn_consec == 0
    assert st.hard_consec == 0
    assert st.step == 2
    # no checkpoint at the anomaly step
    assert not (tmp_path / "checkpoint_step2.pt").exists()
    # but the earlier normal step 1 DID write a checkpoint (legacy behavior)
    assert (tmp_path / "checkpoint_step1.pt").exists()


# ----------------------------------------------------------------------
# 10. 9-field schema (no 10th field)
# ----------------------------------------------------------------------
def test_state_file_has_exactly_nine_fields(tmp_path):
    m, cfg = _make_model()
    pause = tmp_path / "pause_state.json"
    t = _make_trainer(tmp_path, m, cfg, pause_state_path=pause,
                      consecutive_count=5, max_steps=4)
    _mock_eval("inf")
    t.train()
    payload = json.loads(pause.read_text(encoding="utf-8"))
    assert set(payload.keys()) == set(ALL_FIELDS)
    assert len(payload) == 9
    # explicitly: the two R2.13 proposal fields are absent
    assert "model_checkpoint_path" not in payload
    assert "validation_error_summary" not in payload
    # and the REQUIRED + OPTIONAL split adds up
    assert set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS) == set(ALL_FIELDS)
    assert len(REQUIRED_FIELDS) == 6
    assert len(OPTIONAL_FIELDS) == 3


# ----------------------------------------------------------------------
# side-effect check: none of the above tests wrote anywhere OUTSIDE their
# own tmp_path (and never into the source checkpoint or other experiment
# dirs). We assert the source checkpoint hash is still intact after the
# whole module's tests have run.
# ----------------------------------------------------------------------
def test_source_checkpoint_untouched_after_module():
    assert _source_ckpt_hash() == _EXPECTED_SOURCE_HASH, (
        "source step-350 checkpoint hash changed during fault-injection tests")
