"""
Concurrency tests for the AronaLM inference server.

The server serialises inference (one generation at a time) while remaining
responsive for health/status/cancel requests, and rejects duplicate in-flight
request ids.
"""

from __future__ import annotations

import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.server import InferenceService, ServerConfig, build_server
from aronalm.tokenizer import BPETokenizer

API = "/arona/v1"


class Harness:
    def __init__(self, step_callback=None, **config_kwargs):
        self.service = InferenceService(ServerConfig(**config_kwargs), step_callback=step_callback)
        self.server = build_server(self.service, "127.0.0.1", 0)
        self.port = self.server.server_address[1]
        self.base = f"http://127.0.0.1:{self.port}{API}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> "Harness":
        self.thread.start()
        return self

    def __exit__(self, *exc_info) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        self.service.close()

    def request(self, method: str, path: str, body=None, timeout: float = 60.0):
        data = None
        headers = {}
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raw = exc.read().decode("utf-8")
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, {"raw": raw}


@pytest.fixture
def server():
    with Harness() as harness:
        yield harness


def _generate(client, request_id, tokens=4, text="the "):
    return client.request(
        "POST",
        "/generate",
        {
            "request_id": request_id,
            "input": text,
            "generation_config": {"max_new_tokens": tokens, "timeout_ms": 60000},
        },
    )


def test_eight_concurrent_requests_all_succeed(server):
    results: dict[str, tuple] = {}
    lock = threading.Lock()

    def worker(index: int):
        status, body = _generate(server, f"conc-{index}", tokens=3)
        with lock:
            results[f"conc-{index}"] = (status, body)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)
        assert not thread.is_alive(), "concurrent request did not finish"

    assert len(results) == 8
    for request_id, (status, body) in results.items():
        assert status == 200, f"{request_id} failed: {body}"
        assert body["status"] == "ok"
        assert body["request_id"] == request_id
        assert len(body["output"]["tokens"]) == 3


def test_concurrent_responses_are_not_cross_contaminated(server):
    tokenizer = BPETokenizer.load(PROJECT_ROOT / "tokenizer" / "vocab.json")
    results: dict[str, dict] = {}
    lock = threading.Lock()

    def worker(index: int):
        status, body = _generate(server, f"iso-{index}", tokens=2)
        with lock:
            results[f"iso-{index}"] = {"status": status, **body}

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=120)

    for request_id, body in results.items():
        assert body["request_id"] == request_id
        # each response carries its own token payload, all valid for the model
        assert all(0 <= tok < tokenizer._actual_vocab_size for tok in body["output"]["tokens"])


def test_duplicate_in_flight_request_id_is_rejected():
    entered = threading.Event()
    release = threading.Event()

    def blocking_step(_request_id, step):
        if step == 0:
            entered.set()
            release.wait(timeout=15)

    with Harness(step_callback=blocking_step) as server:
        server.service.load()
        first: dict = {}

        def run_first():
            first["response"] = _generate(server, "dup-1", tokens=3)

        worker = threading.Thread(target=run_first)
        worker.start()
        assert entered.wait(timeout=15)

        status, body = _generate(server, "dup-1", tokens=3)
        assert status == 409
        assert body["error"]["code"] == "DUPLICATE_REQUEST_ID"

        release.set()
        worker.join(timeout=30)
        assert first["response"][0] == 200
        assert first["response"][1]["status"] == "ok"


def test_health_is_responsive_while_generation_is_running():
    entered = threading.Event()
    release = threading.Event()

    def blocking_step(_request_id, step):
        if step == 0:
            entered.set()
            release.wait(timeout=15)

    with Harness(step_callback=blocking_step) as server:
        server.service.load()
        result: dict = {}

        def run_generate():
            result["response"] = _generate(server, "busy-1", tokens=3)

        worker = threading.Thread(target=run_generate)
        worker.start()
        assert entered.wait(timeout=15)

        started = time.monotonic()
        status, health = server.request("GET", "/health", timeout=5)
        elapsed = time.monotonic() - started
        assert status == 200
        assert health["model_loaded"] is True
        assert elapsed < 3.0, "health check must not wait for the running generation"

        status, body = server.request("GET", "/status")
        assert status == 200
        assert body["active_requests"] >= 1

        release.set()
        worker.join(timeout=30)
        assert result["response"][1]["status"] == "ok"


def test_status_returns_to_idle_after_requests(server):
    status, body = _generate(server, "idle-1", tokens=2)
    assert status == 200
    status, state = server.request("GET", "/status")
    assert status == 200
    assert state["active_requests"] == 0


def test_parallel_cancellations_do_not_break_server(server):
    def cancel(index: int):
        server.request("POST", f"/cancel/ghost-{index}")

    threads = [threading.Thread(target=cancel, args=(i,)) for i in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=30)

    status, body = server.request("GET", "/health")
    assert status == 200
    assert body["status"] == "ok"


def test_repeated_sequential_requests_are_stable(server):
    for index in range(5):
        status, body = _generate(server, f"seq-{index}", tokens=2)
        assert status == 200
        assert body["status"] == "ok"
        assert body["request_id"] == f"seq-{index}"
