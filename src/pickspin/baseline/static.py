"""The static baseline: send every query to every model, with no routing.

Each model is an OpenAI-compatible vLLM endpoint. Responses are appended to
<out>/<model>_results.jsonl as they complete, one JSON object per line, and an interrupted run resumes
where it stopped because the queries already in the file are skipped. The HTTP call is separate from
the live runner's, because the fields it records, its timer and its error strings differ.
"""

import json
import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

from pickspin.config import MODELS
from pickspin.data import Query
from pickspin.live.vllm import Endpoint, chat_completions_url

log = logging.getLogger(__name__)


def call_endpoint(
    endpoint: Endpoint, model: str, query: str, max_tokens: int, headers: Mapping[str, str]
) -> dict[str, Any]:
    """Send one query to one model and return the fields recorded for it.

    The served model name is the endpoint's 'model', else the Hugging Face id of the model in
    pickspin.config (so model must be a catalog key). latency_ms is wall-clock time from just before
    the request until the reply arrives, or until the error. On failure the response is empty, the
    token counts are 0 and error is 'HTTP <code>', 'Timeout', or the text of any other exception.
    """
    payload = {
        "model": endpoint.get("model", MODELS[model].hf_id),
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.1,
        "stream": False,
    }
    start = time.time()
    try:
        r = requests.post(chat_completions_url(endpoint), json=payload, headers=dict(headers), timeout=120)
        ms = (time.time() - start) * 1000
        if r.status_code == 200:
            d = r.json()
            u = d.get("usage", {})
            return {
                "success": True,
                "response": d["choices"][0]["message"].get("content", ""),
                "latency_ms": ms,
                "prompt_tokens": u.get("prompt_tokens", 0),
                "completion_tokens": u.get("completion_tokens", 0),
                "error": None,
            }
        err = f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        ms, err = (time.time() - start) * 1000, "Timeout"
    except Exception as e:
        ms, err = (time.time() - start) * 1000, str(e)
    return {
        "success": False,
        "response": "",
        "latency_ms": ms,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "error": err,
    }


def run_model(
    model: str,
    endpoint: Endpoint,
    queries: Sequence[Query],
    out_dir: Path,
    *,
    workers: int,
    max_tokens: int,
    headers: Mapping[str, str],
) -> Path:
    """Send every query not yet in the model's results file to the model and append the results.

    Queries run on a pool of worker threads, and the calling thread writes each record as its query
    completes, so the file is in completion order. Returns the results file.
    """
    out = out_dir / f"{model}_results.jsonl"
    done: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [q for q in queries if q.id not in done]
    log.info("%s: %d queries to run (%d already done)", model, len(todo), len(done))
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futures: dict[Future[dict[str, Any]], Query] = {
            pool.submit(call_endpoint, endpoint, model, q.query, max_tokens, headers): q for q in todo
        }
        for future in as_completed(futures):
            q = futures[future]
            record = {
                "id": q.id,
                "benchmark": q.benchmark,
                "model": model,
                "query": q.query[:500],
                "ground_truth": q.ground_truth,
                "query_type": q.query_type,
                **future.result(),
                "timestamp": datetime.now().isoformat(),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return out


def run_static_baseline(
    endpoints: Mapping[str, Endpoint],
    models: Sequence[str],
    queries: Sequence[Query],
    out_dir: Path,
    *,
    workers: int = 50,
    max_tokens: int = 512,
    headers: Mapping[str, str],
) -> list[Path]:
    """Run the static baseline for each model in the given order and return the results files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    return [
        run_model(m, endpoints[m], queries, out_dir, workers=workers, max_tokens=max_tokens, headers=headers)
        for m in models
    ]
