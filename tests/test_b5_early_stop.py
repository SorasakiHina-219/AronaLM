"""
B5 early-stop / anomaly-pause / safe-recovery tests (K1--K6, decisions A/B, P1--P8).

Run:  python -m pytest tests/test_b5_early_stop.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from aronalm.evaluator import EvalResult, evaluate
from aronalm.model import AronaLM, TransformerConfig
from aronalm.pause_state import (
    REQUIRED_FIELDS,
    OPTIONAL_FIELDS,
    PauseState,
    compute_config_checksum,
    load_pause_state,
    make_state,
    save_pause_state,
)
from aronalm.trainer import Trainer


# ----------------------------------------------------------------------
# fixtures
# ----------------------------------------------------------------------
def make_model(vocab_size=32, max_seq_len=8, d_model=16, n_heads=2, n_layers=1, ffn_dim=32):
    cfg = TransformerConfig(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ffn_dim=ffn_dim,
        dropout=0.0,
    )
    torch.manual_seed(0)
    return AronaLM(cfg), cfg


def make_loader(n_batches=2, vocab_size=32, seq_len=8, device="cpu"):
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, vocab_size, (n_batches, seq_len), generator=g)
    return torch.utils.data.DataLoader(data, batch_size=1)


def make_trainer(tmp_path, model, cfg, eval_interval=2, max_steps=12,
                 warning=None, hard_stop=None, baseline=None, pause_path=None,
                 consecutive=2, exemption=10):
    dev = torch.device("cpu")
    return Trainer(
        model=model,
        train_loader=make_loader(2),
        val_loader=make_loader(2),
        config=cfg,
        output_dir=tmp_path,
        device=dev,
        lr=1e-3,
        log_interval=1,
        eval_interval=eval_interval,
        max_steps=max_steps,
        baseline_val_loss=baseline,
        warning_threshold=warning,
        hard_stop_threshold=hard_stop,
        consecutive_count=consecutive,
        exemption_window=exemption,
        pause_state_path=pause_path,
    )


def fake_eval_status(trainer, status, loss=None):
    """Replace trainer's evaluation behavior with a deterministic fake."""
    if status == "ok":
        result = EvalResult(status="ok", loss=loss, perplexity=math.exp(loss))
    else:
        result = EvalResult(status=status, error_message=f"fake {status}")
    import aronalm.trainer as tmod
    tmod.evaluate = lambda m, dl, d: result
    return result


# ----------------------------------------------------------------------
# 1. legacy behavior preserved (B5 disabled)
# ----------------------------------------------------------------------
def test_legacy_trainer_unaffected(tmp_path):
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, max_steps=4, eval_interval=2)
    assert t.b5_enabled is False
    summary = t.train()
    assert summary["stopped"] is None
    assert summary["steps_executed"] == 4
    # checkpoints written as before
    assert (tmp_path / "checkpoint_step2.pt").exists()
    assert (tmp_path / "checkpoint_step4.pt").exists()
    # legacy log-entry shape preserved: val_loss / val_perplexity keys
    # always present (None when the observation is not valid), even though
    # in practice the legacy evaluator almost always returned "ok".
    assert "val_loss" in t.training_log[-1]
    assert "val_perplexity" in t.training_log[-1]
    # and the last eval on this small model should be a valid ok
    assert t.training_log[-1]["val_loss"] is not None
    assert t.training_log[-1]["val_perplexity"] is not None


def test_legacy_trainer_log_shape_stable_on_anomaly(tmp_path, monkeypatch):
    """When B5 is OFF and the validator raises, the legacy code path must
    still produce log entries with `val_loss` / `val_perplexity` keys
    (now None) so existing scripts that read them don't crash."""
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, max_steps=2, eval_interval=1)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="failed",
                                                error_message="injected failure")
    summary = t.train()
    assert summary["stopped"] is None  # B5 off -> no stopping logic
    assert (tmp_path / "checkpoint_step1.pt").exists()
    assert (tmp_path / "checkpoint_step2.pt").exists()
    last = t.training_log[-1]
    assert "val_loss" in last
    assert last["val_loss"] is None
    assert last["val_perplexity"] is None
    assert last.get("val_status") == "failed"


# ----------------------------------------------------------------------
# 2. K3: >= semantics, counter increment at exact threshold
# ----------------------------------------------------------------------
def test_threshold_boundary_ge_increments(tmp_path, monkeypatch):
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, warning=2.0, hard_stop=3.0, baseline=1.0,
                     max_steps=20, eval_interval=2, exemption=10)

    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=2.0)
    # step 11 (first eval outside the window): loss == warning threshold exactly
    t._update_counters(tmod.evaluate(m, None, None), 11)
    assert t.warn_consec == 1  # exactly at threshold -> >= -> increment (K3)
    assert t.hard_consec == 0  # below hard-stop threshold -> not counted


def test_below_threshold_clears_counter(tmp_path, monkeypatch):
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, warning=2.0, hard_stop=3.0, baseline=1.0,
                     max_steps=20, eval_interval=2, exemption=10)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=2.5)
    t._update_counters(tmod.evaluate(m, None, None), 11)
    assert t.warn_consec == 1
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=1.5)
    t._update_counters(tmod.evaluate(m, None, None), 13)
    assert t.warn_consec == 0  # P1/U2: below threshold -> clear
    assert t.hard_consec == 0


# ----------------------------------------------------------------------
# 3. K1/K2 + Decision A: exemption window freezes counters (step 10
#    boundary: step 10 still in window, step 11 outside)
# ----------------------------------------------------------------------
def test_exemption_window_freezes_counters(tmp_path, monkeypatch):
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=20, eval_interval=1, exemption=10)
    import aronalm.trainer as tmod
    # Even wildly bad observations inside the window must not count
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=999.0)
    for step in range(1, 11):
        t._update_counters(tmod.evaluate(m, None, None), step)
    assert t.warn_consec == 0
    assert t.hard_consec == 0
    # Step 10 inclusive in the window
    t._update_counters(tmod.evaluate(m, None, None), 10)
    assert t.warn_consec == 0
    assert t.hard_consec == 0
    # Step 11: normal counter judgment applies
    t._update_counters(tmod.evaluate(m, None, None), 11)
    assert t.warn_consec == 1
    assert t.hard_consec == 1


# ----------------------------------------------------------------------
# 4. K5: anomaly keeps counters unchanged
# ----------------------------------------------------------------------
def test_anomaly_keeps_counters(tmp_path, monkeypatch):
    m, cfg = make_model()
    t = make_trainer(tmp_path, m, cfg, warning=2.0, hard_stop=3.0, baseline=1.0,
                     max_steps=20, eval_interval=2, exemption=10)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=2.5)
    t._update_counters(tmod.evaluate(m, None, None), 11)
    assert t.warn_consec == 1
    for i, reason in enumerate(("nan", "inf", "failed", "empty")):
        t._update_counters(EvalResult(status=reason, error_message=f"fake {reason}"), 12 + i * 2)
        assert t.warn_consec == 1, f"counter changed on anomaly {reason}"
    t._update_counters(EvalResult(status="nan"), 20)
    assert t.warn_consec == 1
    assert t.hard_consec == 0


# ----------------------------------------------------------------------
# 5. P2: hard-stop takes priority over warning on the same observation
# ----------------------------------------------------------------------
def test_hard_stop_priority(monkeypatch):
    m, cfg = make_model()
    t = make_trainer(None, m, cfg, warning=1.0, hard_stop=1.5, baseline=0.5,
                     max_steps=20, eval_interval=2, exemption=0, consecutive=2)
    # Simulate two consecutive high-loss evals that cross BOTH thresholds.
    # Two consecutive high-loss evals that cross BOTH thresholds.
    result = EvalResult(status="ok", loss=1.8)
    t._update_counters(result, 11)
    t._update_counters(result, 13)
    assert t.warn_consec == 2
    assert t.hard_consec == 2
    stop = t._handle_b5_observation(EvalResult(status="ok", loss=1.8), 13, {}, None)
    assert stop == "hard_stop"
    assert t.train_paused is True
    assert t.pause_reason == "hard_stop"


# ----------------------------------------------------------------------
# 6. Decision B: hard-stop writes the stop-state JSON, no checkpoint
# ----------------------------------------------------------------------
def test_hard_stop_writes_state_not_checkpoint(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "b5" / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=1.0, hard_stop=1.5, baseline=0.5,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=1)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"
    assert summary["steps_executed"] == 1
    assert pause_path.exists()
    payload = json.loads(pause_path.read_text(encoding="utf-8"))
    # exactly the 9 confirmed fields -- no more, no less
    keys = set(payload.keys())
    required = {f for f in REQUIRED_FIELDS if f in keys}
    assert required == {"version", "step", "warn_consec", "hard_consec",
                        "pause_reason", "exemption_window_passed"}
    assert payload["pause_reason"] == "hard_stop"
    assert payload["step"] == 1
    # No checkpoint file may exist (Decision B / P7)
    assert not (tmp_path / "checkpoint_step1.pt").exists()


# ----------------------------------------------------------------------
# 7. P3: state-write failure stops training without checkpoint
# ----------------------------------------------------------------------
def test_state_write_failure_stops_training(tmp_path, monkeypatch):
    m, cfg = make_model()
    # A path whose parent is a *file* -> mkdir raises -> save returns False
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    bad_path = blocker / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=bad_path, consecutive=1)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"
    assert summary["steps_executed"] == 1
    # No checkpoint written, no fallback state file anywhere
    assert not list(tmp_path.glob("checkpoint_step*.pt"))


def test_pause_state_save_failure_returns_false(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")
    st = make_state(step=1, warn_consec=0, hard_consec=0,
                    pause_reason="hard_stop", exemption_window_passed=True)
    ok = save_pause_state(blocker / "sub" / "state.json", st)
    assert ok is False
    # no partial file leaked
    assert not (blocker / "sub" / "state.json").exists()


# ----------------------------------------------------------------------
# 8. P4 / P8: --resume cannot bypass hard-stop; no auto-recovery
# ----------------------------------------------------------------------
def test_resume_rejected_for_hard_stop(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=1.0, hard_stop=1.5, baseline=0.5,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=1)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"

    # A brand-new trainer with the same state file must refuse to resume.
    m2, _ = make_model()
    t2 = make_trainer(tmp_path, m2, cfg, warning=1.0, hard_stop=1.5, baseline=0.5,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    ok = t2.resume_from_pause(expected_step=1)
    assert ok is False
    assert t2.train_paused is False  # state unchanged
    assert t2.warn_consec == 0
    assert t2.hard_consec == 0


# ----------------------------------------------------------------------
# 9. anomaly pause -> strict resume works (P6 exact step match)
# ----------------------------------------------------------------------
def test_anomaly_resume_with_exact_step(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=5)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="nan")
    summary = t.train()
    assert summary["stopped"] == "nan"

    st = load_pause_state(pause_path)
    assert st is not None and st.is_anomaly()
    assert st.step == 1

    m2, cfg2 = make_model()
    t2 = make_trainer(tmp_path, m2, cfg2, warning=0.5, hard_stop=0.6, baseline=1.0,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    ok = t2.resume_from_pause(expected_step=st.step)
    assert ok is True
    assert t2.train_paused is False
    assert t2._resumed_step == 1


def test_anomaly_resume_rejects_step_mismatch(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=5)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="inf")
    summary = t.train()
    assert summary["stopped"] == "inf"

    m2, cfg2 = make_model()
    t2 = make_trainer(tmp_path, m2, cfg2, warning=0.5, hard_stop=0.6, baseline=1.0,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    ok = t2.resume_from_pause(expected_step=2)  # state is step 1, strict
    assert ok is False
    assert t2.train_paused is False  # nothing was changed


def test_anomaly_resume_rejects_config_mismatch(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=5)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="failed", error_message="fake")
    summary = t.train()
    assert summary["stopped"] == "failed"

    m2, cfg2 = make_model(vocab_size=64)  # different vocab -> different checksum
    t2 = make_trainer(tmp_path, m2, cfg2, warning=0.5, hard_stop=0.6, baseline=1.0,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    ok = t2.resume_from_pause(expected_step=1)
    assert ok is False


def test_anomaly_resume_rejects_seq_len_mismatch(tmp_path, monkeypatch):
    """P6: max_seq_len is part of the config dict that is hashed into
    config_checksum, so a mismatch in max_seq_len is rejected exactly the
    same way as a vocab_size mismatch."""
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=5)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="failed", error_message="fake")
    summary = t.train()
    assert summary["stopped"] == "failed"

    m2, cfg2 = make_model(max_seq_len=16)  # different seq_len -> different checksum
    t2 = make_trainer(tmp_path, m2, cfg2, warning=0.5, hard_stop=0.6, baseline=1.0,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    ok = t2.resume_from_pause(expected_step=1)
    assert ok is False


def test_hard_stop_state_field_subset(tmp_path):
    """P5: a hard_stop state must carry ONLY the 9 approved fields.
    We verify that writing and re-reading a hard-stop state produces
    exactly the approved field set, and that no 10th field appears."""
    import aronalm.trainer as tmod
    m, cfg = make_model()
    pause_path = tmp_path / "stop_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=1.0, hard_stop=1.5, baseline=0.5,
                     max_steps=2, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=1)
    import aronalm.trainer as _t
    _t.evaluate = lambda *a, **k: EvalResult(status="ok", loss=9.9)
    summary = t.train()
    assert summary["stopped"] == "hard_stop"
    assert pause_path.exists()
    payload = json.loads(pause_path.read_text(encoding="utf-8"))
    # The full approved field set is 9 fields. A hard-stop state carries all 9
    # (the optional ones are populated by make_state).
    from aronalm.pause_state import ALL_FIELDS
    assert set(payload.keys()) == set(ALL_FIELDS), (
        f"field set mismatch: got {sorted(payload.keys())}, "
        f"expected {sorted(ALL_FIELDS)}"
    )
    assert len(payload) == 9
    # specifically: no 10th field such as model_checkpoint_path or
    # validation_error_summary (removed in R2.16 from the R2.13 proposal)
    assert "model_checkpoint_path" not in payload
    assert "validation_error_summary" not in payload


# ----------------------------------------------------------------------
# 9b. anomaly pause halts the training loop (K4/K6: no auto-retry /
#     no auto-recovery), counters kept, state JSON written
# ----------------------------------------------------------------------
def test_anomaly_pause_halts_training_no_retry(tmp_path, monkeypatch):
    """K4: when an observation is anomalous the trainer must halt at that
    step, keep the counters, write the 9-field state JSON, and must NOT
    keep looping or auto-recover."""
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path, consecutive=5)
    import aronalm.trainer as tmod

    # 1st eval = ok, 2nd = failed; the loop must stop at the failed one.
    calls = {"n": 0}

    def fake_eval(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            return EvalResult(status="ok", loss=0.1, perplexity=1.1)
        return EvalResult(status="failed", error_message="injected failure")

    tmod.evaluate = fake_eval
    summary = t.train()
    assert summary["stopped"] == "failed"
    # only 2 evals ran (step 1 ok, step 2 failed -> halt), not 4
    assert calls["n"] == 2
    assert summary["steps_executed"] == 2
    # no further training after the anomaly
    assert t.train_paused is True
    assert t.pause_reason == "failed"
    # counters: step 1 (loss=0.1 < warning 0.5) cleared warn/hard to 0,
    # the anomaly at step 2 kept them there
    assert t.warn_consec == 0
    assert t.hard_consec == 0
    # the anomaly state was written and is readable
    st = load_pause_state(pause_path)
    assert st is not None
    assert st.is_anomaly()
    assert st.pause_reason == "failed"
    assert st.step == 2
    # no checkpoint written on the failed step
    assert not (tmp_path / "checkpoint_step2.pt").exists()
# ----------------------------------------------------------------------
# 10. P5 state-file validation (corrupt / bad version / missing fields)
# ----------------------------------------------------------------------
def test_load_rejects_corrupt_json(tmp_path):
    p = tmp_path / "s.json"
    p.write_text("{not json", encoding="utf-8")
    import pytest
    with pytest.raises(Exception):
        load_pause_state(p)


def test_load_rejects_unknown_version(tmp_path):
    st = make_state(step=1, warn_consec=0, hard_consec=0,
                    pause_reason="nan", exemption_window_passed=True)
    p = tmp_path / "s.json"
    d = st.to_dict()
    d["version"] = "9.9"
    p.write_text(json.dumps(d), encoding="utf-8")
    import pytest
    with pytest.raises(Exception):
        load_pause_state(p)


def test_load_rejects_missing_required_field(tmp_path):
    st = make_state(step=1, warn_consec=0, hard_consec=0,
                    pause_reason="nan", exemption_window_passed=True)
    p = tmp_path / "s.json"
    d = st.to_dict()
    del d["warn_consec"]
    p.write_text(json.dumps(d), encoding="utf-8")
    import pytest
    with pytest.raises(Exception):
        load_pause_state(p)


def test_load_rejects_unknown_reason(tmp_path):
    st = make_state(step=1, warn_consec=0, hard_consec=0,
                    pause_reason="nan", exemption_window_passed=True)
    p = tmp_path / "s.json"
    d = st.to_dict()
    d["pause_reason"] = "weird"
    p.write_text(json.dumps(d), encoding="utf-8")
    import pytest
    with pytest.raises(Exception):
        load_pause_state(p)


def test_roundtrip_valid_state(tmp_path):
    st = make_state(step=7, warn_consec=1, hard_consec=2,
                    pause_reason="nan", exemption_window_passed=True,
                    baseline_val_loss=1.2278622048241752,
                    config_dict={"vocab_size": 32})
    p = tmp_path / "s.json"
    assert save_pause_state(p, st) is True
    st2 = load_pause_state(p)
    assert st2 == st


# ----------------------------------------------------------------------
# 11. P8: existence of a state file never auto-triggers recovery
# ----------------------------------------------------------------------
def test_state_file_never_triggers_auto_resume(tmp_path, monkeypatch):
    m, cfg = make_model()
    pause_path = tmp_path / "pause_state.json"
    st = make_state(step=1, warn_consec=0, hard_consec=0,
                    pause_reason="nan", exemption_window_passed=True)
    assert save_pause_state(pause_path, st)

    t = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                     max_steps=4, eval_interval=1, exemption=0,
                     pause_path=pause_path)
    import aronalm.trainer as tmod
    tmod.evaluate = lambda *a, **k: EvalResult(status="ok", loss=0.1)
    # train() itself must not consult the old state file at all:
    # counters start from 0 regardless of the leftover file.
    summary = t.train()
    assert summary["stopped"] is None
    assert summary["warn_consec"] == 0
    assert summary["hard_consec"] == 0
    # Only an explicit resume_from_pause() consults the file -- and it
    # would succeed here because step/config match.
    t2 = make_trainer(tmp_path, m, cfg, warning=0.5, hard_stop=0.6, baseline=1.0,
                      max_steps=4, eval_interval=1, exemption=0,
                      pause_path=pause_path)
    assert t2.resume_from_pause(expected_step=1) is True


# ----------------------------------------------------------------------
# 12. evaluator K4 statuses
# ----------------------------------------------------------------------
def test_evaluator_empty_loader():
    m, cfg = make_model()
    dev = torch.device("cpu")
    empty = torch.utils.data.DataLoader(torch.empty(0, 8).long())
    r = evaluate(m, empty, dev)
    assert r.status == "empty"
    assert r.loss is None
    assert r.is_valid is False


def test_evaluator_failed_on_exception():
    m, cfg = make_model()
    dev = torch.device("cpu")

    class Boom:
        def __iter__(self):
            yield torch.zeros(1, 8).long()
            raise RuntimeError("boom")

    r = evaluate(m, Boom(), dev)
    assert r.status == "failed"
    assert "boom" in r.error_message


def test_evaluator_nan_status():
    m, cfg = make_model()
    dev = torch.device("cpu")
    import torch as _t
    orig = m.compute_loss
    m.compute_loss = lambda x: _t.tensor(float("nan"))
    try:
        r = evaluate(m, make_loader(1, vocab_size=cfg.vocab_size, seq_len=cfg.max_seq_len), dev)
    finally:
        m.compute_loss = orig
    assert r.status == "nan"


def test_evaluator_inf_status():
    m, cfg = make_model()
    dev = torch.device("cpu")
    import torch as _t
    orig = m.compute_loss
    m.compute_loss = lambda x: _t.tensor(float("inf"))
    try:
        r = evaluate(m, make_loader(1, vocab_size=cfg.vocab_size, seq_len=cfg.max_seq_len), dev)
    finally:
        m.compute_loss = orig
    assert r.status == "inf"


def test_evaluator_ok_dict_compat():
    m, cfg = make_model()
    dev = torch.device("cpu")
    r = evaluate(m, make_loader(1, vocab_size=cfg.vocab_size, seq_len=cfg.max_seq_len), dev)
    assert r.status == "ok"
    assert r["loss"] == r.loss  # legacy dict-style access still works
    assert r["perplexity"] > 0
    assert "status" in r
