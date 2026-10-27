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
import ctypes
import hashlib
import json
import os
import platform
import sys
import threading
import time
from ctypes import c_ulonglong, windll
import ctypes
import platform

IS_WINDOWS = platform.system() == "Windows"
try:
    from ctypes.wintypes import HANDLE
except ImportError:
    HANDLE = None
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


# ----------------------------------------------------------------------
# Task A: lightweight resource peak sampling
#
# Scope & semantics (documented here so a reader knows exactly what the
# numbers in run_metadata.json mean):
#   - SAMPLED OBJECT: this process only (the training process), not other
#     DSH processes.
#   - PROCESS MEMORY PEAK: Windows peak committed memory
#     (PROCESS_PERF_INFO.PeakWorkingSetSize, bytes) -- the maximum the
#     process held since start. This is a monotonic high-water mark, so
#     the "peak" over the sampling window equals the value read at the
#     end of the window.
#   - PROCESS CPU TIME: cumulative kernel + user CPU seconds spent by the
#     process since start (PROCESS_PERF_INFO.UpdatedWorkingSetSize is NOT
#     used; only the two CPU-time fields).
#   - AVG PROCESS CPU PERCENT: total CPU seconds / wall-clock seconds of
#     the sampling window, as a fraction of a single core (can exceed 1.0
#     on multi-core).
#   - SYSTEM AVAILABLE MEMORY: global available physical memory
#     (GlobalMemoryStatusEx, bytes) -- an instantaneous sample, NOT a
#     minimum over the window.
#   - SAMPLING: the sampler thread wakes every 10 s and records a point;
#     the window starts just before t.train() and ends just after it.
#     Points are stored in memory; the single summary block (peak,
#     cumulative CPU, averages, sampled available-memory series) is
#     written to run_metadata.json once, after training finishes.
#   - FAILURES: any exception inside a sample tick is recorded as
#     {"ts": ..., "error": ...} and the loop continues -- sampling must
#     never abort training. If the Windows API is unavailable (non-Windows
#     host), every metric is null with an explicit unavailability reason,
#     never a fake 0.
# ----------------------------------------------------------------------
SAMPLER_INTERVAL_SEC = 10.0


class ResourceSampler:
    """Best-effort peak sampler for one training run (Windows, stdlib-only).

    Start just before `Trainer.train()`, stop just after. The `report()`
    dict is JSON-serializable: every metric is either a number or None
    (with a `notes` entry explaining why), never a fabricated value.
    """

    def __init__(self, interval: float = SAMPLER_INTERVAL_SEC) -> None:
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._points: list[dict] = []
        self._window_start: float | None = None
        self._window_end: float | None = None
        self._proc = os.getpid()
        self._notes: list[str] = []

    # -- Windows API plumbing (best effort) -----------------------------
    @staticmethod
    def _query_process_info(pid: int) -> dict | None:
        """Per-process cumulative CPU time + peak working-set memory.

        All stdlib ctypes; no third-party deps.

          - CPU: kernel32.GetProcessTimes (kernel + user, 100-ns ticks)
          - MEM: psapi.GetProcessMemoryInfo (PROCESS_MEMORY_COUNTERS,
                 peak + current working set). If this call fails on the
                 target OS/build we record "peak_memory_bytes": None and
                 keep the CPU figure -- the sampler never invents numbers.

        Returns a dict with keys:
          cpu_seconds          float | None
          peak_memory_bytes    int     | None
          working_set_bytes    int     | None
        or None when the process handle cannot be opened at all.
        """
        try:
            import ctypes.wintypes as wt

            class FILETIME(ctypes.Structure):
                _fields_ = [("dwLowDateTime", ctypes.c_uint32),
                            ("dwHighDateTime", ctypes.c_uint32)]

            k32 = ctypes.windll.kernel32
            k32.OpenProcess.argtypes = [ctypes.c_uint32, ctypes.c_int, ctypes.c_uint32]
            k32.OpenProcess.restype = wt.HANDLE
            k32.GetProcessTimes.argtypes = [wt.HANDLE, ctypes.POINTER(FILETIME),
                                            ctypes.POINTER(FILETIME),
                                            ctypes.POINTER(FILETIME),
                                            ctypes.POINTER(FILETIME)]
            k32.GetProcessTimes.restype = ctypes.c_int

            h = k32.OpenProcess(0x0400, 0, pid)  # PROCESS_QUERY_INFORMATION
            if not h:
                return None
            try:
                cre, exi, kern, user = (FILETIME() for _ in range(4))
                ok = k32.GetProcessTimes(h, ctypes.byref(cre), ctypes.byref(exi),
                                         ctypes.byref(kern), ctypes.byref(user))
                cpu_seconds: float | None = None
                if ok:
                    def _ft(ft: FILETIME) -> int:
                        return ft.dwLowDateTime | (ft.dwHighDateTime << 32)
                    cpu_seconds = (_ft(kern) + _ft(user)) / 1e7

                peak_bytes: int | None = None
                working_bytes: int | None = None
                try:
                    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_uint32),
                            ("PageFaults", ctypes.c_uint32),
                            ("PeakWorkingSetSize", c_ulonglong),
                            ("WorkingSetSize", c_ulonglong),
                            ("QuotaPeakPagedPoolUsage", c_ulonglong),
                            ("QuotaPagedPoolUsage", c_ulonglong),
                            ("QuotaPeakNonPagedPoolUsage", c_ulonglong),
                            ("QuotaNonPagedPoolUsage", c_ulonglong),
                        ]
                    psapi = ctypes.WinDLL("psapi")
                    psapi.GetProcessMemoryInfo.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_uint32]
                    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
                    pmc = PROCESS_MEMORY_COUNTERS()
                    pmc.dwLength = ctypes.sizeof(pmc)
                    if psapi.GetProcessMemoryInfo(h, ctypes.byref(pmc), ctypes.sizeof(pmc)):
                        peak_bytes = int(pmc.PeakWorkingSetSize)
                        working_bytes = int(pmc.WorkingSetSize)
                except Exception:
                    pass  # memory API unavailable; CPU figure still valid

                return {"cpu_seconds": cpu_seconds,
                        "peak_memory_bytes": peak_bytes,
                        "working_set_bytes": working_bytes}
            finally:
                k32.CloseHandle(h)
        except Exception:
            return None

    @staticmethod
    def _system_available_memory() -> int | None:
        """GlobalMemoryStatusEx -> available physical bytes, or None."""
        try:
            class MEMORY_STAT(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_uint32),
                    ("dwMemoryLoad", ctypes.c_uint32),
                    ("ullTotalPhys", c_ulonglong),
                    ("ullAvailPhys", c_ulonglong),
                    ("ullTotalPageFile", c_ulonglong),
                    ("ullAvailPageFile", c_ulonglong),
                    ("ullTotalVirtual", c_ulonglong),
                    ("ullAvailVirtual", c_ulonglong),
                    ("ullAvailExtendedVirtual", c_ulonglong),
                ]
            st = MEMORY_STAT()
            st.dwLength = ctypes.sizeof(st)
            if not windll.Kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return None
            return int(st.ullAvailPhys)
        except Exception:
            return None

    # -- sampling loop ----------------------------------------------------
    def start(self) -> None:
        self._window_start = time.time()
        self._thread = threading.Thread(target=self._run, name="b5-resource-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 5)
        self._window_end = time.time()

    def _run(self) -> None:
        while not self._stop.is_set():
            self._sample_once()
            self._stop.wait(self.interval)
        # final sample at window end (cheap; one extra API call)
        self._sample_once()

    def _sample_once(self) -> None:
        rec: dict = {"ts": time.time() - (self._window_start or 0.0)}
        try:
            info = self._query_process_info(self._proc)
            if info is not None:
                if info.get("cpu_seconds") is not None:
                    rec["cpu_seconds"] = info["cpu_seconds"]
                if info.get("peak_memory_bytes") is not None:
                    rec["peak_memory_bytes"] = info["peak_memory_bytes"]
                if info.get("working_set_bytes") is not None:
                    rec["working_set_bytes"] = info["working_set_bytes"]
                missing = [k for k in ("cpu_seconds", "peak_memory_bytes")
                           if k not in rec]
                if missing:
                    rec["partial_metrics_missing"] = missing
            else:
                rec["process_metrics_error"] = "process handle unavailable (OpenProcess failed)"
            avail = self._system_available_memory()
            if avail is not None:
                rec["available_memory_bytes"] = avail
            else:
                rec["available_memory_error"] = "GlobalMemoryStatusEx returned no data"
            self._points.append(rec)
        except Exception as e:  # sampling must never crash the run
            rec["sample_error"] = f"{type(e).__name__}: {e}"
            self._notes.append(rec["sample_error"])
            self._points.append(rec)

    # -- report -----------------------------------------------------------
    def report(self) -> dict:
        """Assemble the JSON-serializable summary (all-or-None, with notes)."""
        cpu_points = [p for p in self._points if "cpu_seconds" in p]
        mem_points = [p for p in self._points if "peak_memory_bytes" in p]
        window = None
        if self._window_start is not None and self._window_end is not None:
            window = self._window_end - self._window_start

        out: dict = {
            "sampled": bool(self._points),
            "interval_sec": self.interval,
            "window_start_utc": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._window_start))
                                 if self._window_start is not None else None),
            "window_end_utc": (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self._window_end))
                               if self._window_end is not None else None),
            "window_sec": round(window, 3) if window is not None else None,
            "points": len(self._points),
            "process_peak_memory_bytes": None,
            "process_working_set_bytes_last": None,
            "process_cpu_seconds_total": None,
            "avg_process_cpu_percent_of_one_core": None,
            "system_available_memory_bytes_min": None,
            "system_available_memory_bytes_last": None,
            "process_metrics": ("ok" if cpu_points and mem_points
                                 else "partial" if (cpu_points or mem_points)
                                 else "unavailable"),
            "notes": list(self._notes),
        }

        if mem_points:
            out["process_peak_memory_bytes"] = max(p["peak_memory_bytes"] for p in mem_points)
            if any("working_set_bytes" in p for p in mem_points):
                out["process_working_set_bytes_last"] = next(
                    p["working_set_bytes"] for p in reversed(mem_points)
                    if "working_set_bytes" in p)
        else:
            out["notes"].append(
                "process peak memory unavailable (psapi.GetProcessMemoryInfo failed on "
                "this build); recorded null, not estimated")
        if cpu_points:
            out["process_cpu_seconds_total"] = round(cpu_points[-1]["cpu_seconds"], 3)
            if window and window > 0:
                out["avg_process_cpu_percent_of_one_core"] = round(
                    (cpu_points[-1]["cpu_seconds"] / window) * 100, 2)
        else:
            out["notes"].append(
                "process CPU time unavailable (kernel32.GetProcessTimes failed on "
                "this build); recorded null, not estimated")
        avail_series = [p["available_memory_bytes"] for p in self._points
                        if "available_memory_bytes" in p]
        if avail_series:
            out["system_available_memory_bytes_min"] = min(avail_series)
            out["system_available_memory_bytes_last"] = avail_series[-1]
        else:
            out["notes"].append(
                "system available-memory unavailable (GlobalMemoryStatusEx failed); "
                "recorded null, not estimated")
        return out


# keep the module importable on non-Windows hosts: `windll` above raises
# ModuleNotFoundError there, so ResourceSampler's API-reading methods must
# return None cleanly in that case (they do, via the try/except guards).


def main() -> None:
    ap = argparse.ArgumentParser(description="B5 early-stop experiment")
    ap.add_argument("--mode", choices=["smoke", "main", "sample-only"], required=True)
    ap.add_argument("--b5-off", action="store_true",
                    help="run WITHOUT B5 early-stop (legacy comparison)")
    ap.add_argument("--sample-seconds", type=float, default=20.0,
                    help="for --mode sample-only: how long to sample (default 20 s)")
    args = ap.parse_args()

    if args.mode == "sample-only":
        # Controlled sampler verification: no checkpoint, no training data,
        # no Trainer. Just prove the Windows sampler works end-to-end and
        # that a null-failure path records null (not 0).
        sampler = ResourceSampler(interval=min(SAMPLER_INTERVAL_SEC, max(2.0, args.sample_seconds / 3)))
        sampler.start()
        # busy-ish work to register in CPU counters
        deadline = time.time() + args.sample_seconds
        while time.time() < deadline:
            time.sleep(0.5)
        sampler.stop()
        report = sampler.report()
        out_dir = make_output_dir("b5_sampler_check")
        (out_dir / "sampler_report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        log(f"sampler report written to {out_dir / 'sampler_report.json'}")
        log(json.dumps(report, indent=2, ensure_ascii=False))
        return

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
    sampler = ResourceSampler(interval=SAMPLER_INTERVAL_SEC)
    sampler.start()
    try:
        summary = t.train()
    finally:
        sampler.stop()
        resources = sampler.report()
    elapsed = time.time() - t0
    meta["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta["elapsed_sec"] = round(elapsed, 2)
    meta["summary"] = summary
    meta["resources"] = resources

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
    log(f"  resources: peak_mem={resources['process_peak_memory_bytes']} bytes, "
        f"cpu_total={resources['process_cpu_seconds_total']} s, "
        f"avg_cpu%={resources['avg_process_cpu_percent_of_one_core']} "
        f"({resources['points']} samples over {resources['window_sec']} s)")
    for note in resources.get("notes", []):
        log(f"  resource note: {note}")
    log(f"  output dir: {out_dir}")
    log("=" * 60)


if __name__ == "__main__":
    main()
