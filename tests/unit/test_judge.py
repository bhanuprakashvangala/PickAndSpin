"""The LLM judge: prompts, verdict parsing, configuration from the environment, and one judge call.

The judge call runs against a fake requests.post, so no network is used. run_benchmark, which adds a
thread pool and files, is covered with a localhost server in tests/integration/test_http_clients.py.
"""

import dataclasses
import hashlib
import string
from typing import Any

import pytest
import requests

from pickspin.baseline import judge
from pickspin.baseline.judge import JUDGE_BENCHMARKS, PROMPTS, JudgeConfig, judge_response, parse_verdict
from pickspin.errors import ConfigError, PickSpinError

# sha256 of the UTF-8 lines '<benchmark>' TAB '<prompt>' joined by newlines, in dict order; the
# prompts are those of src/baseline/llm_judge.py at tag v1.1.0, which labelled the released traces.
PROMPTS_SHA256 = "256c6c488d3c43beb78495ce020d78347dc15a121921171493a62017109d3fd3"


class FakeResponse:
    """The parts of requests.Response the judge reads."""

    def __init__(self, status_code: int, payload: Any = None) -> None:
        self.status_code = status_code
        self._payload = payload

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class FakePost:
    """Stands in for requests.post: records each call and returns (or raises) a fixed reply."""

    def __init__(self, reply: FakeResponse | Exception) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def __call__(self, url: str, **kwargs: Any) -> FakeResponse:
        self.calls.append((url, kwargs))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


CONFIG = JudgeConfig(api_base="http://judge.example/v1", api_key="k3y", model="gpt-oss-120b", workers=4)
ITEM = {
    "id": "gsm8k_7",
    "benchmark": "gsm8k",
    "model": "qwen2.5_7B",
    "query": "Q" * 2100,
    "ground_truth": "G" * 1600,
    "query_type": "math",
    "success": True,
    "response": "R" * 2100,
}


def fake_post(monkeypatch: pytest.MonkeyPatch, reply: FakeResponse | Exception) -> FakePost:
    post = FakePost(reply)
    monkeypatch.setattr(judge.requests, "post", post)
    return post


def chat_reply(message: dict[str, Any]) -> FakeResponse:
    return FakeResponse(200, {"choices": [{"message": message}]})


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("INCORRECT", False),
        ("CORRECT", True),
        ("", False),
        ("THE ANSWER IS CORRECT.", True),
        ("NOT CORRECT: IT IS INCORRECT", False),  # INCORRECT contains CORRECT, so it is tested first
        ("UNSURE", False),
        ("correct", False),  # judge_response upper-cases the answer before parsing it
    ],
)
def test_parse_verdict(text: str, verdict: bool) -> None:
    assert parse_verdict(text) is verdict


def test_prompts_and_benchmarks_are_pinned() -> None:
    assert JUDGE_BENCHMARKS == ("gsm8k", "math", "arc", "mmlu_pro", "hellaswag", "truthfulqa", "humaneval", "mbpp")
    assert tuple(PROMPTS) == JUDGE_BENCHMARKS
    text = "\n".join(f"{benchmark}\t{prompt}" for benchmark, prompt in PROMPTS.items())
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == PROMPTS_SHA256
    for prompt in PROMPTS.values():
        fields = {name for _, name, _, _ in string.Formatter().parse(prompt) if name is not None}
        assert fields <= {"question", "ground_truth", "response"}
        assert prompt.endswith("Reply ONLY: CORRECT or INCORRECT")
    with pytest.raises(TypeError):
        PROMPTS["gsm8k"] = "changed"  # type: ignore[index]


def test_from_env_reads_the_judge_settings() -> None:
    env = {
        "JUDGE_API_BASE": "https://judge.example/v1/",
        "JUDGE_API_KEY": "k",
        "JUDGE_MODEL": "gpt-oss-120b",
        "JUDGE_WORKERS": "8",
    }
    assert JudgeConfig.from_env(env) == JudgeConfig("https://judge.example/v1", "k", "gpt-oss-120b", 8)


def test_from_env_defaults() -> None:
    config = JudgeConfig.from_env({"JUDGE_API_BASE": "http://judge.example/v1//"})
    assert dataclasses.astuple(config) == ("http://judge.example/v1", "", "gpt-oss", 100)


@pytest.mark.parametrize(
    "env",
    [{}, {"JUDGE_API_BASE": ""}, {"JUDGE_API_BASE": "/"}, {"JUDGE_API_KEY": "k", "JUDGE_MODEL": "m"}],
    ids=["unset", "empty", "only a slash", "key without base"],
)
def test_from_env_requires_the_api_base(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError, match=r"^Set JUDGE_API_BASE \(and JUDGE_API_KEY\) first; see \.env\.example$") as e:
        JudgeConfig.from_env(env)
    assert isinstance(e.value, PickSpinError)


def test_from_env_rejects_a_non_integer_worker_count() -> None:
    with pytest.raises(ConfigError, match="JUDGE_WORKERS"):
        JudgeConfig.from_env({"JUDGE_API_BASE": "http://judge.example/v1", "JUDGE_WORKERS": "many"})


def test_from_env_reads_the_process_environment_when_called(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("JUDGE_API_KEY", "JUDGE_MODEL", "JUDGE_WORKERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("JUDGE_API_BASE", "http://judge.example/v1/")
    assert JudgeConfig.from_env() == JudgeConfig("http://judge.example/v1")
    monkeypatch.delenv("JUDGE_API_BASE")
    with pytest.raises(ConfigError):
        JudgeConfig.from_env()


def test_the_api_key_is_not_in_repr() -> None:
    config = JudgeConfig.from_env({"JUDGE_API_BASE": "http://judge.example/v1", "JUDGE_API_KEY": "s3cr3t-value"})
    assert config.api_key == "s3cr3t-value"
    assert "s3cr3t-value" not in repr(config)
    assert "s3cr3t-value" not in str(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.workers = 1  # type: ignore[misc]


def test_judge_response_sends_the_judge_request(monkeypatch: pytest.MonkeyPatch) -> None:
    post = fake_post(monkeypatch, chat_reply({"content": "  correct \n"}))
    result = judge_response(ITEM, "gsm8k", CONFIG)
    prompt = PROMPTS["gsm8k"].format(question="Q" * 2000, ground_truth="G" * 1500, response="R" * 2000)
    assert post.calls == [
        (
            "http://judge.example/v1/chat/completions",
            {
                "headers": {"Authorization": "Bearer k3y"},
                "json": {
                    "model": "gpt-oss-120b",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 500,
                    "temperature": 0.0,
                },
                "timeout": 180,
            },
        )
    ]
    assert list(result.items()) == [
        ("id", "gsm8k_7"),
        ("model", "qwen2.5_7B"),
        ("benchmark", "gsm8k"),
        ("is_correct", True),
        ("judge", "CORRECT"),
        ("error", None),
    ]


def test_judge_response_always_sends_the_authorization_header(monkeypatch: pytest.MonkeyPatch) -> None:
    post = fake_post(monkeypatch, chat_reply({"content": "CORRECT"}))
    item = {"id": "arc_1", "model": "gemma2_9B", "ground_truth": 3}
    judge_response(item, "arc", JudgeConfig("http://judge.example/v1"))
    _, kwargs = post.calls[0]
    assert kwargs["headers"] == {"Authorization": "Bearer "}
    # Missing fields become '' and other values are converted with str().
    assert kwargs["json"]["messages"][0]["content"] == PROMPTS["arc"].format(question="", ground_truth="3", response="")


@pytest.mark.parametrize(
    ("payload", "is_correct", "judge_text"),
    [
        ({"choices": [{"message": {"content": "The answer is incorrect."}}]}, False, "THE ANSWER IS INCORRECT."),
        ({"choices": [{"message": {"content": None, "reasoning_content": "so: correct"}}]}, True, "SO: CORRECT"),
        ({"choices": [{"message": {"content": "", "reasoning_content": None}}]}, False, ""),
        (
            {"choices": [{"message": {"content": "Correct \u00fc" + "z" * 300}}]},
            True,
            ("CORRECT \u00dc" + "Z" * 300)[:100],
        ),
        ({}, False, ""),
    ],
    ids=["incorrect", "reasoning content", "empty", "long and non-ASCII", "no choices"],
)
def test_judge_response_reads_the_verdict(
    monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any], is_correct: bool, judge_text: str
) -> None:
    fake_post(monkeypatch, FakeResponse(200, payload))
    result = judge_response(ITEM, "math", CONFIG)
    assert (result["is_correct"], result["judge"], result["error"]) == (is_correct, judge_text, None)
    assert result["benchmark"] == "math"


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        (FakeResponse(503), "HTTP 503"),
        (
            FakeResponse(200, ValueError("Expecting value: line 1 column 1 (char 0)")),
            "Expecting value: line 1 column 1 (char 0)",
        ),
        (FakeResponse(200, {"choices": []}), "list index out of range"),
        (requests.ConnectionError("connection refused"), "connection refused"),
        (requests.Timeout("read timed out"), "read timed out"),
    ],
    ids=["http error", "invalid json", "empty choices", "connection error", "timeout"],
)
def test_judge_response_failures_have_no_verdict(
    monkeypatch: pytest.MonkeyPatch, reply: FakeResponse | Exception, error: str
) -> None:
    fake_post(monkeypatch, reply)
    result = judge_response(ITEM, "mbpp", CONFIG)
    assert list(result.items()) == [
        ("id", "gsm8k_7"),
        ("model", "qwen2.5_7B"),
        ("benchmark", "mbpp"),
        ("is_correct", None),
        ("judge", ""),
        ("error", error),
    ]


def test_judge_response_raises_for_an_unknown_benchmark(monkeypatch: pytest.MonkeyPatch) -> None:
    post = fake_post(monkeypatch, chat_reply({"content": "CORRECT"}))
    with pytest.raises(KeyError):
        judge_response(ITEM, "squad", CONFIG)
    assert post.calls == []
