"""
AronaLM local inference server — HTTP/1.1, Python standard library only.

Design constraints (see ARONA-FOUNDATION-DEVELOPMENT-TASK-01 §5):

- No rewriting of model code: inference reuses ``AronaLM.generate``.
- No third-party runtime dependencies beyond torch (already required by AronaLM).
- Loopback only (127.0.0.1 / ::1). The server refuses to bind public interfaces.
- Graceful shutdown on SIGINT / SIGTERM.
- Lazy model loading: the checkpoint is loaded on the first request that needs it.
- Serialized inference (one generation at a time) with per-request timeout and
  cooperative cancellation between generated tokens.

API (base URL ``http://127.0.0.1:<port>/arona/v1``)::

    GET  /health              service liveness + whether a model is loaded
    GET  /status              model/resource status, active requests
    GET  /model_info          model metadata (loads the checkpoint if needed)
    POST /generate            run one generation request
    POST /cancel/{request_id} request cooperative cancellation of a generation

Run with::

    python -m aronalm.server --port 8765
    python -m aronalm                 # same entry point via __main__.py

The server never reads or writes AronaOS state: no identity, no memory, no
shared database. It is a pure inference endpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
import traceback
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable
from urllib.parse import unquote, urlparse

import torch

from aronalm.checkpoint import load_checkpoint
from aronalm.model import AronaLM, TransformerConfig
from aronalm.tokenizer import BPETokenizer

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

API_PREFIX = "/arona/v1"
DEFAULT_MODEL_ID = "aronalm-v1"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_TIMEOUT_MS = 30_000
MAX_TIMEOUT_MS = 120_000
DEFAULT_MAX_NEW_TOKENS = 20
MAX_NEW_TOKENS_LIMIT = 256
MAX_BODY_BYTES = 1 << 20  # 1 MiB
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}

LOG = logging.getLogger("aronalm.server")


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #


class ApiError(Exception):
    """Structured error mapped to an HTTP status + JSON error body."""

    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.request_id = request_id


class RequestCancelled(Exception):
    """Raised internally when a request is cancelled by the client."""


class RequestTimedOut(Exception):
    """Raised internally when a request exceeds its timeout budget."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass
class ServerConfig:
    """Runtime configuration for one AronaLM server instance."""

    project_root: Path = field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    model_id: str = DEFAULT_MODEL_ID
    checkpoint: Path | None = None
    tokenizer: Path | None = None
    device: str = "cpu"
    host: str = DEFAULT_HOST
    port: int = DEFAULT_PORT
    preload: bool = False
    port_fallback: bool = True

    def resolved_checkpoint(self) -> Path | None:
        return resolve_checkpoint(self.checkpoint, self.project_root)

    def resolved_tokenizer(self) -> Path:
        if self.tokenizer is not None:
            return Path(self.tokenizer)
        return self.project_root / "tokenizer" / "vocab.json"


def resolve_checkpoint(checkpoint: Path | str | None, project_root: Path) -> Path | None:
    """
    Resolve a checkpoint file path.

    - ``None``  -> newest ``checkpoints/checkpoint_step*.pt`` (or None if absent)
    - directory -> newest checkpoint inside it
    - file      -> the file itself (may not exist; the caller reports the error)
    """
    if checkpoint is None:
        directory = project_root / "checkpoints"
        return latest_checkpoint_in(directory)

    path = Path(checkpoint)
    if path.is_dir():
        return latest_checkpoint_in(path)
    return path


def latest_checkpoint_in(directory: Path) -> Path | None:
    """Return the highest-step checkpoint in *directory*, or None."""
    if not directory.is_dir():
        return None
    candidates = []
    for item in directory.glob("checkpoint_step*.pt"):
        try:
            step = int(item.stem.replace("checkpoint_step", ""))
        except ValueError:
            continue
        candidates.append((step, item))
    if not candidates:
        return None
    return max(candidates, key=lambda pair: pair[0])[1]


# --------------------------------------------------------------------------- #
# Inference service
# --------------------------------------------------------------------------- #


class InferenceService:
    """
    Owns the model, the tokenizer and the per-request control state.

    Threading model:
    - ``_load_lock`` guards lazy loading.
    - ``_infer_lock`` serialises inference (one generation at a time).
    - ``_state_lock`` guards the active-request / cancellation registries.
    """

    def __init__(
        self,
        config: ServerConfig,
        *,
        step_callback: Callable[[str, int], None] | None = None,
    ) -> None:
        self.config = config
        self.step_callback = step_callback
        self.started_at = time.monotonic()

        self._load_lock = threading.Lock()
        self._infer_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._model: AronaLM | None = None
        self._tokenizer: BPETokenizer | None = None
        self._model_config: TransformerConfig | None = None
        self._metadata: dict[str, Any] = {}
        self._load_error: str | None = None
        self._load_error_code: str | None = None
        self._loaded = False

        self._active: set[str] = set()
        self._cancel_events: dict[str, tuple[threading.Event, float]] = {}

    # -- lifecycle ---------------------------------------------------------- #

    def load(self) -> None:
        """Load checkpoint + tokenizer. Raises ``ApiError`` on failure."""
        with self._load_lock:
            if self._loaded:
                return
            if self._load_error is not None:
                raise ApiError(
                    503,
                    self._load_error_code or "MODEL_LOAD_ERROR",
                    self._load_error,
                )

            checkpoint_path = self.config.resolved_checkpoint()
            tokenizer_path = self.config.resolved_tokenizer()

            if checkpoint_path is None:
                self._fail_load(
                    "MODEL_NOT_LOADED",
                    f"no checkpoint found under {self.config.project_root / 'checkpoints'}",
                )
            if not checkpoint_path.exists():
                self._fail_load("MODEL_NOT_LOADED", f"checkpoint not found: {checkpoint_path}")
            if not tokenizer_path.exists():
                self._fail_load("MODEL_NOT_LOADED", f"tokenizer not found: {tokenizer_path}")

            try:
                tokenizer = BPETokenizer.load(tokenizer_path)
                device = torch.device(self.config.device)
                model, _optimizer_state, model_config, metadata = load_checkpoint(
                    checkpoint_path, device
                )
                model.eval()
            except Exception as exc:  # noqa: BLE001 - surfaced as structured error
                self._fail_load("MODEL_LOAD_ERROR", f"{type(exc).__name__}: {exc}")

            if model.token_embedding.num_embeddings != tokenizer.vocab_size:
                self._fail_load(
                    "MODEL_LOAD_ERROR",
                    "tokenizer/model vocabulary mismatch: "
                    f"tokenizer.vocab_size={tokenizer.vocab_size}, "
                    f"model.embedding={model.token_embedding.num_embeddings}",
                )

            self._tokenizer = tokenizer
            self._model = model
            self._model_config = model_config
            self._metadata = {
                **metadata,
                "checkpoint": str(checkpoint_path),
                "tokenizer": str(tokenizer_path),
                "device": str(device),
            }
            self._loaded = True
            self._load_error = None
            self._load_error_code = None
            LOG.info("model loaded: %s", checkpoint_path)

    def _fail_load(self, code: str, message: str) -> None:
        """Record a stable load failure and raise it as an ``ApiError``."""
        self._load_error = message
        self._load_error_code = code
        raise ApiError(503, code, message)

    def close(self) -> None:
        """Cancel in-flight work and release the model."""
        self.cancel_all()
        with self._load_lock:
            self._model = None
            self._tokenizer = None
            self._model_config = None
            self._metadata = {}
            self._loaded = False
            self._load_error = None
            self._load_error_code = None

    # -- introspection ------------------------------------------------------ #

    @property
    def model_loaded(self) -> bool:
        return self._loaded

    @property
    def load_error(self) -> str | None:
        return self._load_error

    @property
    def uptime_seconds(self) -> float:
        return time.monotonic() - self.started_at

    @property
    def active_requests(self) -> int:
        with self._state_lock:
            return len(self._active)

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "model_loaded": self._loaded,
            "model_id": self.config.model_id,
            "uptime_seconds": round(self.uptime_seconds, 3),
        }

    def status(self) -> dict[str, Any]:
        with self._state_lock:
            active = len(self._active)
            pending_cancels = len(self._cancel_events)
        return {
            "status": "ok",
            "model_loaded": self._loaded,
            "load_error": self._load_error,
            "model_id": self.config.model_id,
            "active_requests": active,
            "pending_cancellations": pending_cancels,
            "uptime_seconds": round(self.uptime_seconds, 3),
            "device": self.config.device,
            "checkpoint": str(self.config.resolved_checkpoint() or ""),
            "resources": _resource_snapshot(),
        }

    def model_info(self) -> dict[str, Any]:
        self.load()
        assert self._model is not None and self._tokenizer is not None and self._model_config is not None
        cfg = self._model_config
        return {
            "model_id": self.config.model_id,
            "params": sum(p.numel() for p in self._model.parameters()),
            "vocab_size": cfg.vocab_size,
            "actual_vocab_entries": self._tokenizer._actual_vocab_size,
            "max_seq_len": cfg.max_seq_len,
            "d_model": cfg.d_model,
            "n_heads": cfg.n_heads,
            "n_layers": cfg.n_layers,
            "ffn_dim": cfg.ffn_dim,
            "step": self._metadata.get("step"),
            "seed": self._metadata.get("seed"),
            "checkpoint": self._metadata.get("checkpoint"),
            "device": self._metadata.get("device"),
        }

    # -- cancellation ------------------------------------------------------- #

    def _cancel_event_for(self, request_id: str) -> threading.Event:
        with self._state_lock:
            entry = self._cancel_events.get(request_id)
            if entry is None:
                event = threading.Event()
                self._cancel_events[request_id] = (event, time.monotonic())
                return event
            return entry[0]

    def request_cancel(self, request_id: str) -> bool:
        """Mark *request_id* for cancellation; works before it starts too."""
        self._prune_cancel_events()
        event = self._cancel_event_for(request_id)
        event.set()
        return True

    def cancel_all(self) -> None:
        with self._state_lock:
            events = [event for event, _ts in self._cancel_events.values()]
        for event in events:
            event.set()

    def _prune_cancel_events(self, older_than_seconds: float = 300.0) -> None:
        now = time.monotonic()
        with self._state_lock:
            stale = [
                request_id
                for request_id, (_event, created) in self._cancel_events.items()
                if request_id not in self._active and now - created > older_than_seconds
            ]
            for request_id in stale:
                self._cancel_events.pop(request_id, None)

    def _clear_cancel_event(self, request_id: str) -> None:
        with self._state_lock:
            self._cancel_events.pop(request_id, None)

    # -- inference ---------------------------------------------------------- #

    def generate(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Validate and execute one generation request.

        Returns the response body (including ``status``). Raises ``ApiError``
        for validation / lifecycle failures, and returns ``status``
        ``"timeout"`` / ``"cancelled"`` for those two outcomes.
        """
        started = time.monotonic()
        request = _validate_generate_request(payload)

        request_id = request["request_id"]
        timeout_ms = request["timeout_ms"]
        deadline = started + timeout_ms / 1000.0

        self.load()  # raises ApiError when no model is available

        with self._state_lock:
            if request_id in self._active:
                raise ApiError(
                    409,
                    "DUPLICATE_REQUEST_ID",
                    f"request_id '{request_id}' is already in flight",
                )
            self._active.add(request_id)

        cancel_event = self._cancel_event_for(request_id)
        try:
            if cancel_event.is_set():
                return self._control_response(request_id, "cancelled", started)

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return self._control_response(request_id, "timeout", started)
            if not self._infer_lock.acquire(timeout=remaining):
                return self._control_response(request_id, "timeout", started)

            try:
                return self._run_generation(request, cancel_event, deadline, started)
            finally:
                self._infer_lock.release()
        finally:
            with self._state_lock:
                self._active.discard(request_id)
            self._clear_cancel_event(request_id)

    def _run_generation(
        self,
        request: dict[str, Any],
        cancel_event: threading.Event,
        deadline: float,
        started: float,
    ) -> dict[str, Any]:
        assert self._model is not None and self._tokenizer is not None
        assert self._model_config is not None

        model = self._model
        tokenizer = self._tokenizer
        request_id = request["request_id"]

        prompt_ids = tokenizer.encode(request["input"])
        if not prompt_ids:
            raise ApiError(400, "EMPTY_INPUT", "input produced no tokens")

        max_new_tokens = request["max_new_tokens"]
        budget = self._model_config.max_seq_len
        if len(prompt_ids) + max_new_tokens > budget:
            raise ApiError(
                400,
                "INPUT_TOO_LONG",
                f"prompt tokens ({len(prompt_ids)}) + max_new_tokens ({max_new_tokens}) "
                f"exceeds max_seq_len ({budget})",
            )

        generated = torch.tensor([prompt_ids], dtype=torch.long)
        with torch.no_grad():
            for step in range(max_new_tokens):
                if cancel_event.is_set():
                    return self._control_response(request_id, "cancelled", started)
                if time.monotonic() > deadline:
                    return self._control_response(request_id, "timeout", started)
                if self.step_callback is not None:
                    self.step_callback(request_id, step)
                generated = model.generate(
                    generated,
                    max_new_tokens=1,
                    temperature=request["temperature"],
                    top_k=request["top_k"],
                )

        output_ids = generated[0, len(prompt_ids):].tolist()
        text = tokenizer.decode(output_ids)
        latency_ms = int((time.monotonic() - started) * 1000)

        return {
            "request_id": request_id,
            "model_id": self.config.model_id,
            "status": "ok",
            "output": {"tokens": output_ids, "text": text},
            "latency_ms": latency_ms,
            "model_info": self.model_info(),
        }

    def _control_response(self, request_id: str, status: str, started: float) -> dict[str, Any]:
        code = "CANCELLED" if status == "cancelled" else "TIMEOUT"
        message = (
            "request was cancelled before completion"
            if status == "cancelled"
            else "request exceeded its timeout budget"
        )
        return {
            "request_id": request_id,
            "model_id": self.config.model_id,
            "status": status,
            "error": {"code": code, "message": message},
            "latency_ms": int((time.monotonic() - started) * 1000),
        }


def _resource_snapshot() -> dict[str, Any]:
    """Best-effort process resource snapshot; degrades to ``None`` values."""
    snapshot: dict[str, Any] = {"rss_bytes": None, "source": None}
    try:  # psutil is optional and not part of the dependency set
        import psutil  # type: ignore

        snapshot["rss_bytes"] = psutil.Process(os.getpid()).memory_info().rss
        snapshot["source"] = "psutil"
    except Exception:  # noqa: BLE001
        snapshot["source"] = "unavailable"
    return snapshot


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #


def _require_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ApiError(400, "INVALID_GENERATION_CONFIG", f"{name} must be a number")
    if value != value or value in (float("inf"), float("-inf")):
        raise ApiError(400, "INVALID_GENERATION_CONFIG", f"{name} must be finite")
    return float(value)


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ApiError(400, "INVALID_GENERATION_CONFIG", f"{name} must be an integer")
    return value


def _validate_generate_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ApiError(400, "INVALID_REQUEST", "request body must be a JSON object")

    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id.strip():
        raise ApiError(400, "INVALID_REQUEST", "request_id is required and must be a non-empty string")
    request_id = request_id.strip()

    model_id = payload.get("model_id")
    if model_id is not None and not isinstance(model_id, str):
        raise ApiError(400, "INVALID_REQUEST", "model_id must be a string when provided")

    text = payload.get("input")
    if not isinstance(text, str) or text == "":
        raise ApiError(400, "INVALID_REQUEST", "input is required and must be a non-empty string")

    generation = payload.get("generation_config") or {}
    if not isinstance(generation, dict):
        raise ApiError(400, "INVALID_REQUEST", "generation_config must be a JSON object")

    max_new_tokens = generation.get("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
    max_new_tokens = _require_int(max_new_tokens, "max_new_tokens")
    if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_LIMIT:
        raise ApiError(
            400,
            "INVALID_GENERATION_CONFIG",
            f"max_new_tokens must be in [1, {MAX_NEW_TOKENS_LIMIT}]",
        )

    temperature = _require_number(generation.get("temperature", 0.8), "temperature")
    if temperature < 0:
        raise ApiError(400, "INVALID_GENERATION_CONFIG", "temperature must be >= 0")

    top_k = generation.get("top_k")
    if top_k is not None:
        top_k = _require_int(top_k, "top_k")
        if top_k < 1:
            raise ApiError(400, "INVALID_GENERATION_CONFIG", "top_k must be >= 1 when provided")

    timeout_ms = generation.get("timeout_ms", DEFAULT_TIMEOUT_MS)
    timeout_ms = _require_int(timeout_ms, "timeout_ms")
    if timeout_ms <= 0:
        raise ApiError(400, "INVALID_GENERATION_CONFIG", "timeout_ms must be > 0")
    if timeout_ms > MAX_TIMEOUT_MS:
        raise ApiError(
            400,
            "INVALID_GENERATION_CONFIG",
            f"timeout_ms must be <= {MAX_TIMEOUT_MS}",
        )

    return {
        "request_id": request_id,
        "model_id": model_id,
        "input": text,
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_k": top_k,
        "timeout_ms": timeout_ms,
    }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class AronaRequestHandler(BaseHTTPRequestHandler):
    """Routes ``/arona/v1/*`` requests to the inference service."""

    server_version = "AronaLM/1.0"
    protocol_version = "HTTP/1.1"

    # -- plumbing ----------------------------------------------------------- #

    @property
    def service(self) -> InferenceService:
        return self.server.service  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOG.info("%s - %s", self.address_string(), fmt % args)

    def _send_json(self, status: int, body: dict[str, Any]) -> None:
        encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error_body(
        self,
        status: int,
        code: str,
        message: str,
        request_id: str | None = None,
    ) -> None:
        self._send_json(
            status,
            {
                "request_id": request_id,
                "model_id": self.service.config.model_id,
                "status": "error",
                "error": {"code": code, "message": message},
                "latency_ms": 0,
            },
        )

    def _read_json_body(self) -> Any:
        raw_length = self.headers.get("Content-Length")
        if raw_length is None:
            raise ApiError(400, "INVALID_REQUEST", "Content-Length header is required")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ApiError(400, "INVALID_REQUEST", "invalid Content-Length header") from exc
        if length < 0 or length > MAX_BODY_BYTES:
            raise ApiError(413, "PAYLOAD_TOO_LARGE", f"request body must be <= {MAX_BODY_BYTES} bytes")
        raw = self.rfile.read(length) if length else b""
        if not raw:
            raise ApiError(400, "INVALID_JSON", "request body is empty")
        try:
            return json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ApiError(400, "INVALID_JSON", f"request body is not valid JSON: {exc}") from exc

    def _route(self) -> tuple[str, str]:
        parsed = urlparse(self.path)
        path = parsed.path
        if not path.startswith(API_PREFIX):
            raise ApiError(404, "NOT_FOUND", f"unknown path: {path}")
        return path[len(API_PREFIX):] or "/", parsed.query

    # -- HTTP verbs --------------------------------------------------------- #

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def _dispatch(self, method: str) -> None:
        try:
            route, _query = self._route()
            if route == "/health":
                self._require_method(method, "GET")
                self._send_json(200, self.service.health())
            elif route == "/status":
                self._require_method(method, "GET")
                self._send_json(200, self.service.status())
            elif route == "/model_info":
                self._require_method(method, "GET")
                self._send_json(200, self.service.model_info())
            elif route == "/generate":
                self._require_method(method, "POST")
                payload = self._read_json_body()
                try:
                    result = self.service.generate(payload)
                except ApiError as exc:
                    if exc.request_id is None and isinstance(payload, dict):
                        candidate = payload.get("request_id")
                        if isinstance(candidate, str):
                            exc.request_id = candidate
                    raise
                self._send_json(200, result)
            elif route.startswith("/cancel/"):
                self._require_method(method, "POST")
                request_id = unquote(route[len("/cancel/"):]).strip()
                if not request_id:
                    raise ApiError(400, "INVALID_REQUEST", "cancel requires a request_id path segment")
                cancelled = self.service.request_cancel(request_id)
                self._send_json(200, {"request_id": request_id, "cancelled": cancelled})
            else:
                raise ApiError(404, "NOT_FOUND", f"unknown path: {self.path}")
        except ApiError as exc:
            self._send_error_body(exc.status, exc.code, exc.message, exc.request_id)
        except (BrokenPipeError, ConnectionResetError):
            LOG.warning("client disconnected before response")
        except Exception as exc:  # noqa: BLE001 - never leak a traceback to clients
            LOG.error("unhandled error: %s\n%s", exc, traceback.format_exc())
            self._send_error_body(500, "INTERNAL_ERROR", "internal server error")

    @staticmethod
    def _require_method(method: str, expected: str) -> None:
        if method != expected:
            raise ApiError(405, "METHOD_NOT_ALLOWED", f"use {expected} for this path")


class AronaHTTPServer(ThreadingHTTPServer):
    """Threading HTTP server carrying the inference service and bounded queue."""

    daemon_threads = False
    allow_reuse_address = True
    request_queue_size = 64

    def __init__(self, address: tuple[str, int], service: InferenceService) -> None:
        self.service = service
        super().__init__(address, AronaRequestHandler)


def build_server(
    service: InferenceService,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    *,
    port_fallback: bool = False,
    port_attempts: int = 20,
) -> AronaHTTPServer:
    """
    Bind a server on *host*:*port*.

    With ``port_fallback`` and a non-zero port, tries ``port .. port+attempts``
    when the requested port is busy (design report §9.1).
    """
    if host not in LOOPBACK_HOSTS:
        raise ValueError(f"only loopback hosts are allowed, got {host!r}")

    ports = [port]
    if port_fallback and port != 0:
        ports = list(range(port, port + port_attempts))

    last_error: OSError | None = None
    for candidate in ports:
        try:
            return AronaHTTPServer((host, candidate), service)
        except OSError as exc:
            last_error = exc
            LOG.warning("port %s unavailable: %s", candidate, exc)
    assert last_error is not None
    raise last_error


# --------------------------------------------------------------------------- #
# CLI entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="aronalm.server",
        description="AronaLM local inference HTTP server (loopback only, stdlib only).",
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="bind address (loopback only)")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="bind port (0 = ephemeral)")
    parser.add_argument("--checkpoint", default=None, help="checkpoint file or directory")
    parser.add_argument("--tokenizer", default=None, help="tokenizer vocab.json path")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID, help="model identifier reported to clients")
    parser.add_argument("--device", default="cpu", help="torch device (default: cpu)")
    parser.add_argument("--preload", action="store_true", help="load the checkpoint at startup")
    parser.add_argument("--no-port-fallback", action="store_true", help="fail instead of trying port+1")
    parser.add_argument("--quiet", action="store_true", help="suppress request logging")
    return parser.parse_args(argv)


def make_config(args: argparse.Namespace) -> ServerConfig:
    config = ServerConfig(
        model_id=args.model_id,
        checkpoint=Path(args.checkpoint) if args.checkpoint else None,
        tokenizer=Path(args.tokenizer) if args.tokenizer else None,
        device=args.device,
        host=args.host,
        port=args.port,
        preload=args.preload,
        port_fallback=not args.no_port_fallback,
    )
    if config.host not in LOOPBACK_HOSTS:
        raise SystemExit(
            f"error: only loopback hosts are allowed ({', '.join(sorted(LOOPBACK_HOSTS))}), got {config.host!r}"
        )
    return config


def serve(
    service: InferenceService,
    host: str,
    port: int,
    *,
    port_fallback: bool,
    install_signal_handlers: bool = True,
) -> int:
    """Run the server until SIGINT/SIGTERM, then shut down gracefully."""
    server = build_server(service, host, port, port_fallback=port_fallback)
    stop_event = threading.Event()

    def _request_stop(signum: int, _frame: Any) -> None:
        LOG.info("received signal %s, shutting down", signum)
        stop_event.set()

    previous_handlers: dict[int, Any] = {}
    if install_signal_handlers:
        for sig in (
            signal.SIGINT,
            getattr(signal, "SIGTERM", None),
            getattr(signal, "SIGBREAK", None),  # Windows: Ctrl+Break / CTRL_BREAK_EVENT
        ):
            if sig is None:
                continue
            try:
                previous_handlers[sig] = signal.signal(sig, _request_stop)
            except (ValueError, OSError):  # non-main thread or unsupported
                pass

    address, bound_port = server.server_address[:2]
    LOG.info("AronaLM server listening on http://%s:%s%s", address, bound_port, API_PREFIX)

    if service.config.preload:
        try:
            service.load()
        except ApiError as exc:
            LOG.warning("preload failed: %s", exc.message)

    if not service.model_loaded and service.load_error is None:
        LOG.info("model not loaded yet; it will be loaded on first request")

    server_thread = threading.Thread(target=server.serve_forever, name="aronalm-http", daemon=True)
    server_thread.start()

    try:
        while not stop_event.wait(0.5):
            pass
    except KeyboardInterrupt:  # pragma: no cover - interactive path
        stop_event.set()

    LOG.info("stopping server")
    service.cancel_all()
    server.shutdown()
    server.server_close()
    server_thread.join(timeout=10)
    service.close()

    for sig, handler in previous_handlers.items():
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError):  # pragma: no cover
            pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config = make_config(args)
    service = InferenceService(config)
    try:
        return serve(service, config.host, config.port, port_fallback=config.port_fallback)
    except OSError as exc:
        print(f"error: cannot bind {config.host}:{config.port}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
