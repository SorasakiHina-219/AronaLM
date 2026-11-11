"""
/generate endpoint tests for the AronaLM inference server.
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
from aronalm.tokenizer import BPETokenizer

API = "/arona/v1"


class Harness:
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


def _generate(server, request_id="gen-1", text="the ", max_new_tokens=5, **generation):
    config = {"max_new_tokens": max_new_tokens}
    config.update(generation)
    return server.request(
        "POST",
        "/generate",
        {"request_id": request_id, "input": text, "generation_config": config},
    )


def test_basic_generate_returns_ok(server):
    status, body = _generate(server)
    assert status == 200
    assert body["status"] == "ok"
    assert body["request_id"] == "gen-1"
    assert body["model_id"] == "aronalm-v1"
    assert isinstance(body["output"]["tokens"], list)
    assert len(body["output"]["tokens"]) == 5
    assert isinstance(body["output"]["text"], str)
    assert body["latency_ms"] >= 0


def test_generated_tokens_are_inside_actual_vocabulary(server):
    tokenizer = BPETokenizer.load(PROJECT_ROOT / "tokenizer" / "vocab.json")
    status, body = _generate(server, max_new_tokens=10)
    assert status == 200
    assert all(0 <= token < tokenizer._actual_vocab_size for token in body["output"]["tokens"])


def test_greedy_generation_is_deterministic(server):
    first = _generate(server, request_id="det-1", max_new_tokens=6, temperature=0.0)[1]
    second = _generate(server, request_id="det-2", max_new_tokens=6, temperature=0.0)[1]
    assert first["output"]["tokens"] == second["output"]["tokens"]


def test_default_generation_config_is_applied(server):
    status, body = server.request(
        "POST", "/generate", {"request_id": "default-1", "input": "the "}
    )
    assert status == 200
    assert len(body["output"]["tokens"]) == 20  # DEFAULT_MAX_NEW_TOKENS


def test_top_k_sampling_still_produces_valid_tokens(server):
    status, body = _generate(server, request_id="topk-1", max_new_tokens=4, temperature=1.0, top_k=5)
    assert status == 200
    assert len(body["output"]["tokens"]) == 4


def test_explicit_matching_model_id_is_accepted(server):
    status, body = server.request(
        "POST",
        "/generate",
        {
            "request_id": "mid-1",
            "model_id": "aronalm-v1",
            "input": "the ",
            "generation_config": {"max_new_tokens": 2},
        },
    )
    assert status == 200
    assert body["status"] == "ok"


def test_response_model_info_matches_model_info_endpoint(server):
    _, generated = _generate(server, request_id="info-1", max_new_tokens=2)
    _, info = server.request("GET", "/model_info")
    assert generated["model_info"]["vocab_size"] == info["vocab_size"]
    assert generated["model_info"]["actual_vocab_entries"] == info["actual_vocab_entries"]
    assert generated["model_info"]["max_seq_len"] == info["max_seq_len"]
    assert generated["model_info"]["params"] == info["params"]


def test_decoded_text_is_consistent_with_tokens(server):
    _, body = _generate(server, request_id="decode-1", max_new_tokens=3)
    tokenizer = BPETokenizer.load(PROJECT_ROOT / "tokenizer" / "vocab.json")
    assert tokenizer.decode(body["output"]["tokens"]) == body["output"]["text"]


def test_same_request_id_can_be_reused_after_completion(server):
    first = _generate(server, request_id="reuse-1", max_new_tokens=2)
    second = _generate(server, request_id="reuse-1", max_new_tokens=2)
    assert first[0] == 200 and second[0] == 200
    assert first[1]["status"] == "ok" and second[1]["status"] == "ok"


def test_longer_prompt_reduces_generation_budget_check(server):
    # "the quick brown fox " is short enough for 5 tokens, but 60 new tokens cannot fit
    status, body = server.request(
        "POST",
        "/generate",
        {
            "request_id": "budget-1",
            "input": "the quick brown fox jumps over the lazy dog ",
            "generation_config": {"max_new_tokens": 60},
        },
    )
    assert status == 400
    assert body["error"]["code"] == "INPUT_TOO_LONG"


def test_generate_works_for_every_generation_budget_boundary(server):
    tokenizer = BPETokenizer.load(PROJECT_ROOT / "tokenizer" / "vocab.json")
    prompt_tokens = len(tokenizer.encode("the "))
    budget = 64 - prompt_tokens
    status, body = _generate(server, request_id="budget-2", max_new_tokens=budget)
    assert status == 200
    assert len(body["output"]["tokens"]) == budget


CHECKPOINT_DIRS = [
    "checkpoints",
    "checkpoints_v2",
    "checkpoints_v2b_dropout0",
    "checkpoints_v2b_dropout05",
    "checkpoints_v2c_500steps",
    "checkpoints_v2c_dropout015",
]


@pytest.mark.parametrize("checkpoint_dir", CHECKPOINT_DIRS)
def test_server_serves_every_delivered_checkpoint_directory(checkpoint_dir):
    """L7: 每组已交付 checkpoint 都能被 server 加载并返回推理结果。"""
    directory = PROJECT_ROOT / checkpoint_dir
    if not any(directory.glob("checkpoint_step*.pt")):
        pytest.skip(f"no checkpoints in {checkpoint_dir}")

    with Harness(checkpoint=directory) as server:
        status, body = _generate(server, request_id=f"ckpt-{checkpoint_dir}", max_new_tokens=2)
        assert status == 200, body
        assert body["status"] == "ok"
        assert len(body["output"]["tokens"]) == 2
        assert body["model_info"]["vocab_size"] == 1000
        assert body["model_info"]["actual_vocab_entries"] == 558
