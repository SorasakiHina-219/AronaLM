"""
Health / status / model_info endpoint tests for the AronaLM inference server.
"""

from __future__ import annotations

import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from aronalm.server import InferenceService, ServerConfig, build_server

API = "/arona/v1"


class Harness:
    """Start one inference server on an ephemeral loopback port."""

    def __init__(self, **config_kwargs):
        self.service = InferenceService(ServerConfig(**config_kwargs))
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

    def request(self, method: str, path: str, body=None, timeout: float = 30.0):
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


def test_health_returns_ok(server):
    status, body = server.request("GET", "/health")
    assert status == 200
    assert body["status"] == "ok"
    assert body["model_id"] == "aronalm-v1"
    assert "uptime_seconds" in body


def test_health_reports_model_not_loaded_before_first_request(server):
    status, body = server.request("GET", "/health")
    assert status == 200
    assert body["model_loaded"] is False


def test_health_does_not_trigger_lazy_load(server):
    server.request("GET", "/health")
    server.request("GET", "/health")
    assert server.service.model_loaded is False


def test_health_reports_model_loaded_after_generate(server):
    status, body = server.request(
        "POST",
        "/generate",
        {"request_id": "h-1", "input": "the ", "generation_config": {"max_new_tokens": 2}},
    )
    assert status == 200
    assert body["status"] == "ok"

    status, health = server.request("GET", "/health")
    assert status == 200
    assert health["model_loaded"] is True


def test_status_reports_idle_state(server):
    status, body = server.request("GET", "/status")
    assert status == 200
    assert body["status"] == "ok"
    assert body["model_loaded"] is False
    assert body["active_requests"] == 0
    assert "resources" in body
    assert body["uptime_seconds"] >= 0


def test_status_exposes_load_error_when_model_missing(tmp_path):
    missing = tmp_path / "no_such_checkpoint.pt"
    with Harness(checkpoint=missing) as server:
        status, body = server.request("GET", "/status")
        assert status == 200
        assert body["model_loaded"] is False
        assert "checkpoint" in body["checkpoint"].lower() or body["checkpoint"].endswith(".pt")


def test_model_info_triggers_load_and_reports_dimensions(server):
    status, body = server.request("GET", "/model_info")
    assert status == 200
    assert body["model_id"] == "aronalm-v1"
    assert body["vocab_size"] == 1000
    assert body["actual_vocab_entries"] == 558
    assert body["max_seq_len"] == 64
    assert body["d_model"] == 64
    assert body["n_heads"] == 2
    assert body["n_layers"] == 2
    assert body["params"] > 0
    assert body["actual_vocab_entries"] <= body["vocab_size"]
    assert server.service.model_loaded is True


def test_model_info_reports_parameter_count_matching_architecture(server):
    status, body = server.request("GET", "/model_info")
    assert status == 200
    # TransformerConfig.total_parameters() for the shipped architecture
    assert body["params"] == 199656


def test_unknown_path_returns_404(server):
    status, body = server.request("GET", "/does-not-exist")
    assert status == 404
    assert body["status"] == "error"
    assert body["error"]["code"] == "NOT_FOUND"


def test_root_path_outside_api_prefix_returns_404(server):
    status, body = server.request("GET", "/health")
    assert status == 200
    status, body = server.request("GET", "/arona/v2/health")
    assert status == 404
    assert body["error"]["code"] == "NOT_FOUND"


def test_wrong_method_on_health_returns_405(server):
    status, body = server.request("POST", "/health", {})
    assert status == 405
    assert body["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_wrong_method_on_generate_returns_405(server):
    status, body = server.request("GET", "/generate")
    assert status == 405
    assert body["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_health_supports_query_string(server):
    status, body = server.request("GET", "/health?verbose=1")
    assert status == 200
    assert body["status"] == "ok"


def test_multiple_health_calls_are_stable(server):
    first = server.request("GET", "/health")[1]
    second = server.request("GET", "/health")[1]
    assert first["model_id"] == second["model_id"]
    assert second["uptime_seconds"] >= first["uptime_seconds"]
