"""The three HTTP clients against stub servers on localhost: live call_vllm, the static baseline and the judge.

The clients stay separate because what they send, record and report differs. call_vllm reports a
requests error by its exception's class name; the static baseline sends stream=False, falls back to
the model's Hugging Face id, records an error by its text and appends ensure_ascii=False JSON lines it
can resume from; the judge always sends an Authorization header, resumes by (query id, model) and
writes ASCII-only JSON lines. The static baseline waits 120 s for a reply, so its timeouts are raised
by a stand-in for requests.post instead of a slow server.
"""

import contextlib
import json
import socket
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest
import requests

from pickspin.baseline.judge import PROMPTS, JudgeConfig, run_benchmark
from pickspin.baseline.static import call_endpoint, run_model, run_static_baseline
from pickspin.config import MODELS
from pickspin.data import Query
from pickspin.live.vllm import ChatResult, Endpoint, bearer_headers, call_vllm

BASELINE_KEYS = [
    "id",
    "benchmark",
    "model",
    "query",
    "ground_truth",
    "query_type",
    "success",
    "response",
    "latency_ms",
    "prompt_tokens",
    "completion_tokens",
    "error",
    "timestamp",
]
RESULT_KEYS = BASELINE_KEYS[6:12]  # what call_endpoint returns
JUDGE_KEYS = ["id", "model", "benchmark", "is_correct", "judge", "error"]


def chat_reply(content: Any, **usage: int) -> dict[str, Any]:
    """An OpenAI-style chat completion body with one choice and the given token usage."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}], "usage": usage}


def prompt_of(request: Any) -> str:
    """The text of the one user message of a chat completion request."""
    (message,) = request.body["messages"]
    return str(message["content"])


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSON-lines file, one object per line, in file order."""
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def refused_port() -> int:
    """A localhost port that nothing listens on, so a connection to it is refused."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@contextlib.contextmanager
def unanswered_port() -> Iterator[int]:
    """A localhost port that accepts connections but never answers, so a request to it times out."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        yield int(s.getsockname()[1])


# call_vllm, the live runner's client


def test_call_vllm_returns_the_reply_and_sends_the_openai_payload(http_stub):
    server = http_stub(
        lambda request: (200, chat_reply("Bonjour, \u00e7a va ?", prompt_tokens=12, completion_tokens=7))
    )
    endpoint: Endpoint = {"base_url": server.base_url + "/", "model": "served/model", "deployment": "d"}

    result = call_vllm(endpoint, "Wie viele \u00c4pfel?", 64, bearer_headers("secret"))

    assert result == ChatResult(True, "Bonjour, \u00e7a va ?", 7, result.seconds)
    assert isinstance(result.seconds, float)
    assert result.seconds >= 0
    (request,) = server.received
    assert (request.method, request.path) == ("POST", "/v1/chat/completions")
    assert list(request.body) == ["model", "messages", "max_tokens", "temperature"]
    assert request.body == {
        "model": "served/model",
        "messages": [{"role": "user", "content": "Wie viele \u00c4pfel?"}],
        "max_tokens": 64,
        "temperature": 0.1,
    }
    assert request.headers["Authorization"] == "Bearer secret"


def test_call_vllm_without_an_api_key_sends_no_authorization_header(http_stub):
    server = http_stub(lambda request: (200, chat_reply("ok")))

    result = call_vllm({"base_url": server.base_url, "model": "m"}, "q", 8, bearer_headers(None))

    assert result[:3] == (True, "ok", 0)  # a reply without completion_tokens counts 0 tokens
    (request,) = server.received
    assert "Authorization" not in request.headers


def test_call_vllm_reports_an_http_error_as_a_failure(http_stub):
    server = http_stub(lambda request: (503, {}))

    result = call_vllm({"base_url": server.base_url, "model": "m"}, "q", 8, {})

    assert result[:3] == (False, "HTTP 503", 0)
    assert result.seconds >= 0


@pytest.mark.usefixtures("no_proxy")
def test_call_vllm_reports_a_refused_connection_by_the_exception_class_name():
    endpoint: Endpoint = {"base_url": f"http://127.0.0.1:{refused_port()}", "model": "m"}

    result = call_vllm(endpoint, "q", 8, {}, timeout=30)

    assert result[:3] == (False, "ConnectionError", 0)
    assert result.seconds >= 0


@pytest.mark.usefixtures("no_proxy")
def test_call_vllm_reports_a_timeout_by_the_exception_class_name():
    with unanswered_port() as port:
        result = call_vllm({"base_url": f"http://127.0.0.1:{port}", "model": "m"}, "q", 8, {}, timeout=0.2)

    assert result[:3] == (False, "ReadTimeout", 0)
    assert result.seconds >= 0.1  # the time spent waiting for the reply


def test_call_vllm_reports_a_reply_that_is_not_json_by_the_exception_class_name(http_stub):
    server = http_stub(lambda request: (200, None))  # HTTP 200 with an empty body

    result = call_vllm({"base_url": server.base_url, "model": "m"}, "q", 8, {})

    assert result[:3] == (False, "JSONDecodeError", 0)


def test_call_vllm_lets_a_reply_without_choices_raise(http_stub):
    # Only requests errors become failures, as in v1.1.0: a malformed HTTP 200 reply raises.
    server = http_stub(lambda request: (200, {"usage": {"completion_tokens": 1}}))

    with pytest.raises(KeyError, match="choices"):
        call_vllm({"base_url": server.base_url, "model": "m"}, "q", 8, {})


# The static baseline


@pytest.mark.parametrize(
    ("endpoint_model", "sent_model"),
    [(None, MODELS["qwen2.5_7B"].hf_id), ("served-name", "served-name")],
    ids=["hf-id-fallback", "endpoint-model"],
)
def test_call_endpoint_sends_stream_false_and_the_served_model(http_stub, endpoint_model, sent_model):
    server = http_stub(lambda request: (200, chat_reply("4", prompt_tokens=11, completion_tokens=2)))
    endpoint: Endpoint = {"base_url": server.base_url}
    if endpoint_model is not None:
        endpoint["model"] = endpoint_model

    result = call_endpoint(endpoint, "qwen2.5_7B", "What is 2 + 2?", 32, {"Authorization": "Bearer k"})

    (request,) = server.received
    assert request.path == "/v1/chat/completions"
    assert list(request.body) == ["model", "messages", "max_tokens", "temperature", "stream"]
    assert request.body == {
        "model": sent_model,
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 32,
        "temperature": 0.1,
        "stream": False,
    }
    assert request.headers["Authorization"] == "Bearer k"
    assert list(result) == RESULT_KEYS
    assert result == {
        "success": True,
        "response": "4",
        "latency_ms": result["latency_ms"],
        "prompt_tokens": 11,
        "completion_tokens": 2,
        "error": None,
    }
    assert result["latency_ms"] >= 0


def test_call_endpoint_reads_a_reply_without_content_or_usage_as_empty(http_stub):
    server = http_stub(lambda request: (200, {"choices": [{"message": {"role": "assistant"}}]}))

    result = call_endpoint({"base_url": server.base_url}, "llama3.2_1B", "q", 8, {})

    assert {k: result[k] for k in ("success", "response", "prompt_tokens", "completion_tokens", "error")} == {
        "success": True,
        "response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": None,
    }


@pytest.mark.usefixtures("no_proxy")
def test_call_endpoint_reports_a_refused_connection_by_the_exception_text():
    port = refused_port()

    result = call_endpoint({"base_url": f"http://127.0.0.1:{port}"}, "llama3.2_1B", "q", 8, {})

    assert list(result) == RESULT_KEYS
    assert {k: result[k] for k in ("success", "response", "prompt_tokens", "completion_tokens")} == {
        "success": False,
        "response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
    }
    # Where call_vllm records the class name, 'ConnectionError', the baseline records the error's text.
    assert f"port={port}" in result["error"]
    assert result["latency_ms"] >= 0


def test_call_endpoint_reports_a_reply_that_is_not_json_by_the_error_text(http_stub):
    server = http_stub(lambda request: (200, None))  # HTTP 200 with an empty body

    result = call_endpoint({"base_url": server.base_url}, "llama3.2_1B", "q", 8, {})

    assert list(result) == RESULT_KEYS
    assert (result["success"], result["response"], result["error"]) == (
        False,
        "",
        "Expecting value: line 1 column 1 (char 0)",
    )


@pytest.mark.parametrize(
    ("error", "recorded"),
    [
        (requests.exceptions.ReadTimeout("read timed out"), "Timeout"),
        (requests.exceptions.ConnectTimeout("connect timed out"), "Timeout"),  # also a ConnectionError
        (requests.exceptions.ConnectionError("connection reset"), "connection reset"),
        (ValueError("not a valid URL"), "not a valid URL"),
    ],
    ids=["read-timeout", "connect-timeout", "connection-error", "any-other-exception"],
)
def test_call_endpoint_records_a_timeout_as_timeout_and_any_other_error_by_its_text(monkeypatch, error, recorded):
    def post(*args: Any, **kwargs: Any) -> Any:
        raise error

    monkeypatch.setattr(requests, "post", post)

    result = call_endpoint({"base_url": "http://127.0.0.1:9"}, "llama3.2_1B", "q", 8, {})

    assert list(result) == RESULT_KEYS
    assert (result["success"], result["response"], result["error"]) == (False, "", recorded)
    assert (result["prompt_tokens"], result["completion_tokens"]) == (0, 0)
    assert result["latency_ms"] >= 0


def baseline_server(http_stub: Callable[..., Any]) -> Any:
    """A model server that answers 'answer: <prompt>' with fixed token counts, or HTTP 500 to a prompt with 'FAIL'."""

    def reply(request: Any) -> tuple[int, Any]:
        prompt = prompt_of(request)
        if "FAIL" in prompt:
            return 500, {"error": "boom"}
        return 200, chat_reply(f"answer: {prompt[:20]}", prompt_tokens=5, completion_tokens=3)

    return http_stub(reply)


LONG = "Wie viele \u00c4pfel? " + "x" * 600  # longer than the 500 characters the record keeps
QUERIES = [
    Query("q0", "gsm8k", "Already done", "1", "math"),
    Query("q1", "gsm8k", "What is 2 + 2?", "4", "math"),
    Query("q2", "arc", LONG, "\u00c4", "multiple_choice"),
    Query("q3", "mbpp", "Please FAIL", "", "code"),
]


def test_run_model_appends_records_in_the_old_format_and_resumes(http_stub, tmp_path):
    server = baseline_server(http_stub)
    endpoint: Endpoint = {"base_url": server.base_url, "model": "served"}
    out_dir = tmp_path / "static"
    out_dir.mkdir()
    results = out_dir / "llama3.2_1B_results.jsonl"
    earlier = {"id": "q0", "benchmark": "gsm8k", "model": "llama3.2_1B", "success": True}
    results.write_text(json.dumps(earlier) + "\n", encoding="utf-8")

    out = run_model("llama3.2_1B", endpoint, QUERIES, out_dir, workers=2, max_tokens=16, headers={})

    assert out == results
    # q0 is already in the file, so only the other three were sent, each in full.
    assert sorted(prompt_of(r) for r in server.received) == sorted(q.query for q in QUERIES[1:])
    assert {r.body["max_tokens"] for r in server.received} == {16}
    first, *added = read_jsonl(results)
    assert first == earlier
    assert [list(record) for record in added] == [BASELINE_KEYS] * 3
    by_id = {record["id"]: record for record in added}
    assert sorted(by_id) == ["q1", "q2", "q3"]
    for q in QUERIES[1:]:
        record = by_id[q.id]
        expected = {
            "benchmark": q.benchmark,
            "model": "llama3.2_1B",
            "query": q.query[:500],
            "ground_truth": q.ground_truth,
            "query_type": q.query_type,
        }
        assert {k: record[k] for k in expected} == expected
        assert record["latency_ms"] >= 0
        datetime.fromisoformat(record["timestamp"])
    assert {k: by_id["q1"][k] for k in RESULT_KEYS if k != "latency_ms"} == {
        "success": True,
        "response": "answer: What is 2 + 2?",
        "prompt_tokens": 5,
        "completion_tokens": 3,
        "error": None,
    }
    assert {k: by_id["q3"][k] for k in RESULT_KEYS if k != "latency_ms"} == {
        "success": False,
        "response": "",
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": "HTTP 500",
    }
    # The JSON keeps non-ASCII text as it is.
    text = results.read_text(encoding="utf-8")
    assert "Wie viele \u00c4pfel?" in text
    assert "\\u00c4" not in text

    # A second run finds every query in the file, failures included, and sends nothing.
    run_model("llama3.2_1B", endpoint, QUERIES, out_dir, workers=2, max_tokens=16, headers={})

    assert len(server.received) == 3
    assert results.read_text(encoding="utf-8") == text


def test_run_static_baseline_runs_every_model_in_the_given_order(http_stub, tmp_path):
    server = baseline_server(http_stub)
    models = ["qwen2.5_7B", "llama3.2_1B"]
    endpoints: dict[str, Endpoint] = {m: {"base_url": server.base_url} for m in models}
    out_dir = tmp_path / "live" / "static"

    paths = run_static_baseline(endpoints, models, QUERIES[1:3], out_dir, workers=2, max_tokens=8, headers={})

    assert paths == [out_dir / f"{m}_results.jsonl" for m in models]
    for m, path in zip(models, paths):
        assert sorted(record["id"] for record in read_jsonl(path)) == ["q1", "q2"]
        assert {record["model"] for record in read_jsonl(path)} == {m}
    # The endpoints name no served model, so each model was asked for by its Hugging Face id.
    sent = sorted(r.body["model"] for r in server.received)
    assert sent == sorted([MODELS["qwen2.5_7B"].hf_id] * 2 + [MODELS["llama3.2_1B"].hf_id] * 2)


# The LLM judge


def response(qid: str, model: str, benchmark: str, text: str, *, success: bool = True) -> dict[str, Any]:
    """One line of a static-baseline results file."""
    return {
        "id": qid,
        "benchmark": benchmark,
        "model": model,
        "query": f"Question {qid}",
        "ground_truth": f"truth {qid}",
        "query_type": "math",
        "success": success,
        "response": text,
    }


def judge_reply(request: Any) -> tuple[int, Any]:
    """The stub judge: its verdict depends on the model response quoted in the prompt."""
    prompt = prompt_of(request)
    if "response: right" in prompt.lower():
        return 200, chat_reply("  correct \u2014 tr\u00e8s bien\n")
    if "response: wrong" in prompt.lower():
        return 200, chat_reply("Incorrect")
    if "response: thinking" in prompt.lower():
        return 200, {"choices": [{"message": {"content": None, "reasoning_content": "correct"}}]}
    return 503, {}


def test_judge_labels_responses_resumes_by_pair_and_writes_ascii_json(http_stub, tmp_path):
    server = http_stub(judge_reply)
    responses_dir = tmp_path / "static"
    responses_dir.mkdir()
    lines = {
        "gemma2_2B_results.jsonl": [
            response("g1", "gemma2_2B", "gsm8k", "right"),
            response("g2", "gemma2_2B", "gsm8k", "wrong"),
            response("g3", "gemma2_2B", "gsm8k", "right", success=False),  # failed runs are not judged
            response("a1", "gemma2_2B", "arc", "right"),  # another benchmark
        ],
        "llama3.2_1B_results.jsonl": [
            response("g1", "llama3.2_1B", "gsm8k", "wrong"),  # already judged
            response("g2", "llama3.2_1B", "gsm8k", "thinking"),
            response("g4", "llama3.2_1B", "gsm8k", "garbled"),  # the judge answers HTTP 503
        ],
    }
    for name, records in lines.items():
        (responses_dir / name).write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out_dir = tmp_path / "judgments"
    out_dir.mkdir()
    out = out_dir / "gsm8k_accuracy.jsonl"
    earlier = {"id": "g1", "model": "llama3.2_1B", "benchmark": "gsm8k", "is_correct": False}
    out.write_text(json.dumps(earlier) + "\n", encoding="utf-8")
    config = JudgeConfig(api_base=server.base_url + "/v1", api_key="", model="judge-model", workers=3)

    assert run_benchmark("gsm8k", config, responses_dir, out_dir) == out

    # g1 on llama3.2_1B was judged before; g1 on gemma2_2B was not, so the pair, not the id, counts.
    first, *added = read_jsonl(out)
    assert first == earlier
    assert [list(label) for label in added] == [JUDGE_KEYS] * 4
    assert {label["benchmark"] for label in added} == {"gsm8k"}
    # (is_correct, judge, error) per (id, model); the judge keeps the upper-cased answer.
    labels = {(label["id"], label["model"]): (label["is_correct"], label["judge"], label["error"]) for label in added}
    assert labels == {
        ("g1", "gemma2_2B"): (True, "CORRECT \u2014 TR\u00c8S BIEN", None),
        ("g2", "gemma2_2B"): (False, "INCORRECT", None),
        ("g2", "llama3.2_1B"): (True, "CORRECT", None),  # from reasoning_content, as content is null
        ("g4", "llama3.2_1B"): (None, "", "HTTP 503"),
    }
    # json.dumps defaults: the file is ASCII, with non-ASCII text escaped.
    raw = out.read_bytes()
    assert raw.isascii()
    assert b"\\u2014" in raw

    # Every request: the chat completions path under the API base, the Authorization header even
    # without a key, and the prompt of the benchmark with the item's fields.
    assert len(server.received) == 4
    for request in server.received:
        assert request.path == "/v1/chat/completions"
        assert "Authorization" in request.headers
        assert list(request.body) == ["model", "messages", "max_tokens", "temperature"]
        assert (request.body["model"], request.body["max_tokens"], request.body["temperature"]) == (
            "judge-model",
            500,
            0.0,
        )
    item = lines["gemma2_2B_results.jsonl"][0]
    expected_prompt = PROMPTS["gsm8k"].format(
        question=item["query"], ground_truth=item["ground_truth"], response=item["response"]
    )
    assert expected_prompt in [prompt_of(request) for request in server.received]

    # A second run finds every pair judged, including the one the judge could not label.
    before = out.read_bytes()
    run_benchmark("gsm8k", config, responses_dir, out_dir)

    assert len(server.received) == 4
    assert out.read_bytes() == before
