"""
Error handling, timeout and cancellation tests for the AronaLM inference server.
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
        return self._send(method, path, data, headers, timeout)

    def request_raw(self, method: str, path: str, raw: bytes, timeout: float = 60.0):
        return self._send(method, path, raw, {"Content-Type": "application/json"}, timeout)

    def _send(self, method, path, data, headers, timeout):
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


def _post(server, body):
    return server.request("POST", "/generate", body)


# --------------------------------------------------------------------------- #
# Malformed requests
# --------------------------------------------------------------------------- #


def test_invalid_json_body_is_rejected(server):
    status, body = server.request_raw("POST", "/generate", b"{not json")
    assert status == 400
    assert body["error"]["code"] == "INVALID_JSON"


def test_empty_body_is_rejected(server):
    status, body = server.request_raw("POST", "/generate", b"")
    assert status == 400
    assert body["error"]["code"] == "INVALID_JSON"


def test_non_object_json_body_is_rejected(server):
    status, body = server.request_raw("POST", "/generate", b"[1,2,3]")
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_missing_request_id_is_rejected(server):
    status, body = _post(server, {"input": "the "})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_blank_request_id_is_rejected(server):
    status, body = _post(server, {"request_id": "   ", "input": "the "})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_missing_input_is_rejected(server):
    status, body = _post(server, {"request_id": "x-1"})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_empty_input_string_is_rejected(server):
    status, body = _post(server, {"request_id": "x-2", "input": ""})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_non_string_input_is_rejected(server):
    status, body = _post(server, {"request_id": "x-3", "input": 123})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


def test_non_object_generation_config_is_rejected(server):
    status, body = _post(server, {"request_id": "x-4", "input": "the ", "generation_config": "fast"})
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


# --------------------------------------------------------------------------- #
# generation_config validation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "generation",
    [
        {"max_new_tokens": 0},
        {"max_new_tokens": 257},
        {"max_new_tokens": -3},
        {"max_new_tokens": "10"},
        {"max_new_tokens": True},
        {"temperature": -0.5},
        {"temperature": "hot"},
        {"temperature": None},
        {"top_k": 0},
        {"timeout_ms": 0},
        {"timeout_ms": -1},
        {"timeout_ms": 999_999},
    ],
)
def test_invalid_generation_config_is_rejected(server, generation):
    status, body = _post(
        server, {"request_id": "cfg-1", "input": "the ", "generation_config": generation}
    )
    assert status == 400
    assert body["error"]["code"] == "INVALID_GENERATION_CONFIG"


def test_prompt_exceeding_max_seq_len_is_rejected(server):
    status, body = _post(
        server,
        {"request_id": "long-1", "input": "abc" * 40, "generation_config": {"max_new_tokens": 80}},
    )
    assert status == 400
    assert body["error"]["code"] == "INPUT_TOO_LONG"


# --------------------------------------------------------------------------- #
# Routing errors
# --------------------------------------------------------------------------- #


def test_unknown_generate_sibling_path_is_404(server):
    status, body = server.request("POST", "/infer", {"request_id": "x"})
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND"


def test_cancel_without_request_id_is_400(server):
    status, body = server.request("POST", "/cancel/")
    assert status == 400
    assert body["error"]["code"] == "INVALID_REQUEST"


# --------------------------------------------------------------------------- #
# Model availability
# --------------------------------------------------------------------------- #


def test_generate_without_checkpoint_returns_model_not_loaded(tmp_path):
    with Harness(checkpoint=tmp_path / "missing.pt") as server:
        status, body = _post(server, {"request_id": "np-1", "input": "the "})
        assert status == 503
        assert body["error"]["code"] == "MODEL_NOT_LOADED"


def test_health_still_works_without_checkpoint(tmp_path):
    with Harness(checkpoint=tmp_path / "missing.pt") as server:
        status, body = server.request("GET", "/health")
        assert status == 200
        assert body["model_loaded"] is False


def test_model_info_without_checkpoint_returns_503(tmp_path):
    with Harness(checkpoint=tmp_path / "missing.pt") as server:
        status, body = server.request("GET", "/model_info")
        assert status == 503
        assert body["error"]["code"] == "MODEL_NOT_LOADED"


def test_repeated_load_failure_keeps_a_stable_error_code(tmp_path):
    with Harness(checkpoint=tmp_path / "missing.pt") as server:
        first_status, first_body = server.request("GET", "/model_info")
        second_status, second_body = server.request("POST", "/generate", {"request_id": "s-1", "input": "the "})
        assert first_status == 503 and second_status == 503
        assert first_body["error"]["code"] == "MODEL_NOT_LOADED"
        assert second_body["error"]["code"] == "MODEL_NOT_LOADED"


def test_unreadable_checkpoint_returns_model_load_error(tmp_path):
    broken = tmp_path / "checkpoint_step1.pt"
    broken.write_bytes(b"not a torch checkpoint")
    with Harness(checkpoint=broken) as server:
        status, body = _post(server, {"request_id": "bad-1", "input": "the "})
        assert status == 503
        assert body["error"]["code"] == "MODEL_LOAD_ERROR"


# --------------------------------------------------------------------------- #
# Timeout and cancellation
# --------------------------------------------------------------------------- #


def test_timeout_returns_timeout_status_without_partial_output():
    def slow_step(_request_id, _step):
        time.sleep(0.06)

    with Harness(step_callback=slow_step) as server:
        server.service.load()  # keep checkpoint load time out of the timeout budget
        status, body = server.request(
            "POST",
            "/generate",
            {
                "request_id": "timeout-1",
                "input": "the ",
                "generation_config": {"max_new_tokens": 10, "timeout_ms": 30},
            },
        )
        assert status == 200
        assert body["status"] == "timeout"
        assert body["error"]["code"] == "TIMEOUT"
        assert "output" not in body


def test_pre_cancel_before_generate_returns_cancelled(server):
    status, body = server.request("POST", "/cancel/pre-cancel-1")
    assert status == 200
    assert body["cancelled"] is True

    status, body = _post(server, {"request_id": "pre-cancel-1", "input": "the "})
    assert status == 200
    assert body["status"] == "cancelled"
    assert body["error"]["code"] == "CANCELLED"
    assert "output" not in body


def test_mid_generation_cancel_aborts_before_completion():
    entered_step = threading.Event()
    release = threading.Event()

    def blocking_step(request_id, step):
        if step == 1:
            entered_step.set()
            release.wait(timeout=10)

    with Harness(step_callback=blocking_step) as server:
        server.service.load()
        result: dict = {}

        def run():
            result["response"] = server.request(
                "POST",
                "/generate",
                {
                    "request_id": "mid-cancel-1",
                    "input": "the ",
                    "generation_config": {"max_new_tokens": 20, "timeout_ms": 15000},
                },
            )

        worker = threading.Thread(target=run)
        worker.start()
        assert entered_step.wait(timeout=15), "generation never reached step 1"

        status, cancel_body = server.request("POST", "/cancel/mid-cancel-1")
        assert status == 200
        assert cancel_body["cancelled"] is True
        release.set()
        worker.join(timeout=20)

        assert not worker.is_alive(), "generation did not stop after cancellation"
        status, body = result["response"]
        assert status == 200
        assert body["status"] == "cancelled"
        assert "output" not in body


def test_cancel_unknown_request_is_accepted(server):
    status, body = server.request("POST", "/cancel/never-existed")
    assert status == 200
    assert body["request_id"] == "never-existed"
