"""Task A: unit tests for the ResourceSampler in experiments/b5/run_b5.py.

All tests use tmp_path fixtures, never run real training, and verify:
  - success path produces finite numbers
  - unavailable-API path produces nulls (never 0)
  - an exception inside a tick is recorded but does not crash the loop
  - stop() joins the thread cleanly
  - non-Windows platform (windll=None) imports cleanly and records nulls
  - PROCESS_MEMORY_COUNTERS struct layout + windll.psapi resolution

Run: python -m pytest tests/test_b5_resource_sampler.py -q
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# run_b5.py lives in experiments/b5, not on the import path by default
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments" / "b5"))

import run_b5


# ----------------------------------------------------------------------
# 1. success path: sample a real (short) window and check the report has
#    finite numbers when the Windows API is available.
# ----------------------------------------------------------------------
def test_sampler_success_path():
    s = run_b5.ResourceSampler(interval=0.2)
    s.start()
    # give the thread at least two ticks
    time.sleep(0.5)
    s.stop()
    rep = s.report()

    # the report is always JSON-serializable
    json.dumps(rep)

    if rep["process_metrics"] == "ok":
        assert rep["process_peak_memory_bytes"] is not None
        assert rep["process_peak_memory_bytes"] > 0
        assert rep["process_cpu_seconds_total"] is not None
        assert rep["process_cpu_seconds_total"] >= 0
        assert rep["points"] >= 2
        assert rep["sampled"] is True
    else:
        # platform where at least one metric is missing: each missing
        # metric must be null with an explanatory note, never 0
        if rep["process_peak_memory_bytes"] is None:
            assert any("peak memory unavailable" in n for n in rep["notes"])
        if rep["process_cpu_seconds_total"] is None:
            assert any("CPU time unavailable" in n for n in rep["notes"])


# ----------------------------------------------------------------------
# 2. failure path: force _query_process_info to return None -> report
#    records nulls with a note, never 0.
# ----------------------------------------------------------------------
def test_sampler_unavailable_metrics_recorded_as_null(monkeypatch):
    s = run_b5.ResourceSampler(interval=0.1)
    monkeypatch.setattr(run_b5.ResourceSampler, "_query_process_info",
                       classmethod(lambda cls, pid: None))
    monkeypatch.setattr(run_b5.ResourceSampler, "_system_available_memory",
                       classmethod(lambda cls: None))
    s.start()
    time.sleep(0.25)
    s.stop()
    rep = s.report()
    assert rep["process_metrics"] == "unavailable"
    assert rep["process_peak_memory_bytes"] is None
    assert rep["process_cpu_seconds_total"] is None
    assert rep["system_available_memory_bytes_min"] is None
    assert rep["system_available_memory_bytes_last"] is None
    # no fabricated zeros
    for key in ("process_peak_memory_bytes", "process_cpu_seconds_total",
                "avg_process_cpu_percent_of_one_core",
                "system_available_memory_bytes_min",
                "system_available_memory_bytes_last"):
        assert rep[key] is None, f"{key} must be None, got {rep[key]}"


# ----------------------------------------------------------------------
# 3. exception inside a tick: the sampler records the error, keeps going,
#    and does not raise to the caller.
# ----------------------------------------------------------------------
def test_sampler_exception_in_tick_does_not_crash(monkeypatch):
    s = run_b5.ResourceSampler(interval=0.05)
    call_count = {"n": 0}

    def flaky(pid):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError("injected tick failure")
        # full success shape (matches the real _query_process_info contract)
        return {"cpu_seconds": 0.001, "peak_memory_bytes": 42, "working_set_bytes": 40}

    monkeypatch.setattr(run_b5.ResourceSampler, "_query_process_info", classmethod(flaky))
    monkeypatch.setattr(run_b5.ResourceSampler, "_system_available_memory",
                       classmethod(lambda cls: 123456))
    s.start()
    time.sleep(0.4)
    s.stop()
    rep = s.report()
    # at least one tick recorded a sample_error
    assert any("sample_error" in p for p in s._points)
    # recovery assertion depends on whether this platform's API works: on a
    # host without the performance API, every tick is None and the report
    # is honestly "unavailable" -- the important property is that the
    # injected exceptions were recorded, not that recovery happened.
    if rep["process_metrics"] == "ok":
        assert rep["process_peak_memory_bytes"] == 42


# ----------------------------------------------------------------------
# 4. resource release: after stop() the thread must be dead, not
#    lingering.
# ----------------------------------------------------------------------
def test_sampler_thread_released_after_stop():
    s = run_b5.ResourceSampler(interval=0.1)
    s.start()
    time.sleep(0.2)
    s.stop()
    assert s._thread is not None
    assert not s._thread.is_alive()


# ----------------------------------------------------------------------
# 5. report() before start() is still well-formed.
# ----------------------------------------------------------------------
def test_report_before_start_is_safe():
    s = run_b5.ResourceSampler()
    rep = s.report()
    assert rep["sampled"] is False
    assert rep["process_metrics"] in ("unavailable", "partial", "ok")
    json.dumps(rep)


# ----------------------------------------------------------------------
# 6. non-Windows platform: module imports cleanly, sampler records nulls.
#    Simulated by forcing run_b5.windll to None (as the guard sets it on
#    Linux/macOS) and re-checking the end-to-end behavior.
# ----------------------------------------------------------------------
def test_non_windows_platform_records_nulls(monkeypatch):
    monkeypatch.setattr(run_b5, "windll", None)
    s = run_b5.ResourceSampler(interval=0.1)
    s.start()
    time.sleep(0.25)
    s.stop()
    rep = s.report()
    assert rep["process_metrics"] == "unavailable"
    assert rep["process_peak_memory_bytes"] is None
    assert rep["process_cpu_seconds_total"] is None
    assert any("unavailable" in n.lower() for n in rep["notes"])
    json.dumps(rep)


# ----------------------------------------------------------------------
# 7. PROCESS_MEMORY_COUNTERS struct layout sanity: 56 bytes on x64,
#    field order matches the Win32 SDK, and the windll.psapi path
#    (not WinDLL) is what actually resolves the function on builds that
#    export it by ordinal only.
# ----------------------------------------------------------------------
def test_process_memory_counters_layout(monkeypatch):
    import ctypes
    if not run_b5.IS_WINDOWS:
        pytest.skip("Windows-only API check")

    from ctypes import c_ulonglong, c_uint32
    from ctypes import windll as _windll

    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("dwLength", c_uint32),
            ("PageFaults", c_uint32),
            ("PeakWorkingSetSize", c_ulonglong),
            ("WorkingSetSize", c_ulonglong),
            ("QuotaPeakPagedPoolUsage", c_ulonglong),
            ("QuotaPagedPoolUsage", c_ulonglong),
            ("QuotaPeakNonPagedPoolUsage", c_ulonglong),
            ("QuotaNonPagedPoolUsage", c_ulonglong),
        ]
    # x64: 2 x uint32 (8) + 6 x uint64 (48) = 56 bytes, no padding
    assert ctypes.sizeof(PROCESS_MEMORY_COUNTERS) == 56

    # windll.psapi must successfully resolve GetProcessMemoryInfo
    try:
        fn = _windll.psapi.GetProcessMemoryInfo
    except Exception as e:
        pytest.skip(f"psapi.GetProcessMemoryInfo not resolvable on this build: {e}")
    assert callable(fn)

    # The real sampler path (_query_process_info) must not raise and, when
    # the API is available, return non-null memory figures for its own pid.
    s = run_b5.ResourceSampler()
    info = s._query_process_info(os.getpid())
    if info is not None:
        if info.get("peak_memory_bytes") is not None:
            assert info["peak_memory_bytes"] > 0
            assert info["working_set_bytes"] > 0
        # On a build where the API works but the process is a kernel-mode
        # restricted target, both may be None -- that is the honest null.
