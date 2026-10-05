"""The LLM judge: label the static baseline's responses CORRECT or INCORRECT.

The released labels (results/traces/judgments.csv.gz) were produced with gpt-oss-120b behind an
OpenAI-compatible API, with the prompts below. The judge reads <responses>/*_results.jsonl and appends
to <out>/<benchmark>_accuracy.jsonl, skipping (id, model) pairs that are already judged. Its settings
come from the environment through JudgeConfig.from_env, which the command line calls at run time.

A response the judge could not label (an HTTP error or an exception) is still written, with
is_correct null and the error, and counts as judged when a run resumes.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import requests

from pickspin.errors import ConfigError

log = logging.getLogger(__name__)

JUDGE_BENCHMARKS: Final[tuple[str, ...]] = (
    "gsm8k",
    "math",
    "arc",
    "mmlu_pro",
    "hellaswag",
    "truthfulqa",
    "humaneval",
    "mbpp",
)

PROMPTS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "gsm8k": "Question: {question}\nCorrect Answer: {ground_truth}\nModel Response: {response}\n\nDoes the model's final numerical answer equal {ground_truth}? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "math": "Question: {question}\nCorrect Answer: {ground_truth}\nModel Response: {response}\n\nDoes the model get the correct final answer? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "arc": "Question: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model select {ground_truth}? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "mmlu_pro": "Question: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model select {ground_truth}? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "hellaswag": "Context: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model choose {ground_truth}? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "truthfulqa": "Question: {question}\nAcceptable: {ground_truth}\nResponse: {response}\n\nIs response factually correct? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "humaneval": "Task: {question}\nReference: {ground_truth}\nCode: {response}\n\nIs code correct? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
        "mbpp": "Task: {question}\nReference: {ground_truth}\nCode: {response}\n\nIs code correct? Reply ONLY: CORRECT or INCORRECT",  # noqa: E501
    }
)


@dataclass(frozen=True)
class JudgeConfig:
    """Where the judge runs: the API base URL (including /v1), API key, served model and worker threads.

    api_base has no trailing slash; from_env removes one. The API key is left out of repr().
    """

    api_base: str
    api_key: str = field(default="", repr=False)
    model: str = "gpt-oss"
    workers: int = 100

    @classmethod
    def from_env(cls, env: Mapping[str, str] = os.environ) -> JudgeConfig:
        """Read JUDGE_API_BASE (required), JUDGE_API_KEY, JUDGE_MODEL and JUDGE_WORKERS.

        Raises ConfigError when JUDGE_API_BASE is unset or empty, or JUDGE_WORKERS is not an integer.
        """
        api_base = env.get("JUDGE_API_BASE", "").rstrip("/")
        if not api_base:
            raise ConfigError("Set JUDGE_API_BASE (and JUDGE_API_KEY) first; see .env.example")
        api_key = env.get("JUDGE_API_KEY", "")
        model = env.get("JUDGE_MODEL", "gpt-oss")
        raw_workers = env.get("JUDGE_WORKERS", "100")
        try:
            workers = int(raw_workers)
        except ValueError as e:
            raise ConfigError(f"JUDGE_WORKERS must be an integer, not {raw_workers!r}") from e
        return cls(api_base=api_base, api_key=api_key, model=model, workers=workers)


def parse_verdict(text: str) -> bool:
    """Return True if the judge's answer says CORRECT and not INCORRECT."""
    # "INCORRECT" contains "CORRECT", so test it first.
    return False if "INCORRECT" in text else "CORRECT" in text


def _label(
    item: Mapping[str, Any], benchmark: str, is_correct: bool | None, judge: str, error: str | None
) -> dict[str, Any]:
    """Return one line of <benchmark>_accuracy.jsonl; the key order is the file's."""
    return {
        "id": item["id"],
        "model": item["model"],
        "benchmark": benchmark,
        "is_correct": is_correct,
        "judge": judge,
        "error": error,
    }


def judge_response(item: Mapping[str, Any], benchmark: str, config: JudgeConfig) -> dict[str, Any]:
    """Ask the judge about one response and return the label record.

    item is one line of a static-baseline results file. The query, ground truth and response are cut
    to 2,000, 1,500 and 2,000 characters for the prompt. is_correct is the verdict of the judge's
    upper-cased answer (its content, else its reasoning_content), and judge keeps the first 100
    characters of that answer. A reply other than HTTP 200 gives is_correct None and error
    'HTTP <code>'; any exception gives is_correct None and the exception's text as error.
    """
    prompt = PROMPTS[benchmark].format(
        question=str(item.get("query", ""))[:2000],
        ground_truth=str(item.get("ground_truth", ""))[:1500],
        response=str(item.get("response", ""))[:2000],
    )
    try:
        r = requests.post(
            f"{config.api_base}/chat/completions",
            headers={"Authorization": f"Bearer {config.api_key}"},
            json={
                "model": config.model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.0,
            },
            timeout=180,
        )
        if r.status_code != 200:
            return _label(item, benchmark, None, "", f"HTTP {r.status_code}")
        msg = r.json().get("choices", [{}])[0].get("message", {})
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip().upper()
        return _label(item, benchmark, parse_verdict(text), text[:100], None)
    except Exception as e:
        return _label(item, benchmark, None, "", str(e))


def run_benchmark(benchmark: str, config: JudgeConfig, responses_dir: Path, out_dir: Path) -> Path:
    """Judge every successful, not yet judged response of one benchmark and return the output file.

    Responses come from every <responses_dir>/*_results.jsonl in sorted order. Labels are appended to
    <out_dir>/<benchmark>_accuracy.jsonl in completion order by the calling thread.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{benchmark}_accuracy.jsonl"
    done: set[tuple[str, str]] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {(d["id"], d["model"]) for d in map(json.loads, f)}
    items: list[dict[str, Any]] = []
    for results_file in sorted(responses_dir.glob("*_results.jsonl")):
        with results_file.open(encoding="utf-8") as f:
            for d in map(json.loads, f):
                if d.get("success") and d["benchmark"] == benchmark and (d["id"], d["model"]) not in done:
                    items.append(d)
    log.info("%s: %d responses to judge", benchmark, len(items))
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=config.workers) as pool:
        for future in as_completed([pool.submit(judge_response, item, benchmark, config) for item in items]):
            f.write(json.dumps(future.result()) + "\n")
    return out
