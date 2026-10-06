"""Load-test a running Pick and Spin gateway: send it the benchmark queries and record what comes back.

replay sends each query as a chat completion with "model": "auto" from a pool of worker threads, the
way clients of the gateway would. A request answered 503 (its model is still loading) or that cannot
reach the gateway is sent again after the Retry-After the gateway gives (30 s otherwise), until
give_up_s seconds have passed since its first attempt. A run writes <out>/replay_<time>.jsonl, one
line per query in completion order, and <out>/replay_<time>_summary.json with the success rate, the
latency percentiles, the queries per model and tier, and the gateway's /stats at the end.

Only the gateway's URL is needed: the client neither classifies nor scales anything itself.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from http import HTTPStatus
from pathlib import Path
from typing import Any

import requests

from pickspin.data import Query

log = logging.getLogger(__name__)

RETRY_S = 30.0  # the wait before resending when the gateway gives no Retry-After

# post(url, body, timeout_s) -> (HTTP status, response headers, decoded body or text)
Post = Callable[[str, dict[str, Any], float], tuple[int, Mapping[str, str], Any]]


@dataclass(frozen=True, slots=True)
class ReplayRecord:
    """The outcome of one query sent to the gateway."""

    id: str
    benchmark: str
    status: int  # the last HTTP status, 0 if the gateway could not be reached
    success: bool
    model: str | None  # the model the gateway chose, from its pickspin object
    tier: str | None
    stage: str | None
    cold_start: bool | None
    wait_s: float | None  # the gateway's wait for the model, on the attempt that was answered
    total_s: float  # from the first attempt to the last answer, retries included
    attempts: int
    tokens: int
    response: str  # the first 500 characters of the answer
    error: str

    def to_json(self) -> str:
        """Return the record as one JSON line (non-ASCII text kept as is)."""
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


def _post(url: str, body: dict[str, Any], timeout_s: float) -> tuple[int, Mapping[str, str], Any]:
    r = requests.post(url, json=body, timeout=timeout_s)
    try:
        return r.status_code, r.headers, r.json()
    except ValueError:
        return r.status_code, r.headers, r.text


def send(
    base_url: str,
    query: Query,
    *,
    max_tokens: int = 256,
    timeout_s: float = 900.0,
    give_up_s: float = 3600.0,
    post: Post = _post,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> ReplayRecord:
    """Send one query to the gateway, resending it while its model loads, and record the outcome."""
    url = base_url.rstrip("/") + "/v1/chat/completions"
    body = {"model": "auto", "max_tokens": max_tokens, "messages": [{"role": "user", "content": query.query}]}
    t0 = clock()
    attempts = 0
    while True:
        attempts += 1
        headers: Mapping[str, str] = {}
        try:
            status, headers, data = post(url, body, timeout_s)
        except requests.RequestException as e:
            status, data = 0, {"error": {"message": f"gateway unreachable: {type(e).__name__}"}}
        if status not in (0, HTTPStatus.SERVICE_UNAVAILABLE) or clock() - t0 >= give_up_s:
            break
        sleep(float(headers.get("Retry-After", RETRY_S)))
    total_s = clock() - t0
    reply: dict[str, Any] = data if isinstance(data, dict) else {"error": {"message": str(data)[:500]}}
    meta: dict[str, Any] = reply.get("pickspin") or {}
    ok = status == HTTPStatus.OK
    text = ""
    if ok:
        choices = reply.get("choices") or [{}]
        text = str((choices[0].get("message") or {}).get("content") or "")
    error = "" if ok else str((reply.get("error") or {}).get("message") or f"HTTP {status}")
    return ReplayRecord(
        id=query.id,
        benchmark=query.benchmark,
        status=status,
        success=ok,
        model=meta.get("model"),
        tier=meta.get("tier"),
        stage=meta.get("stage"),
        cold_start=meta.get("cold_start"),
        wait_s=meta.get("wait_s"),
        total_s=round(total_s, 3),
        attempts=attempts,
        tokens=int((reply.get("usage") or {}).get("completion_tokens") or 0),
        response=text[:500],
        error=error,
    )


def _percentile(sorted_values: Sequence[float], p: float) -> float | None:
    """The nearest-rank p-th percentile of sorted values, or None without any."""
    if not sorted_values:
        return None
    return sorted_values[max(0, math.ceil(p / 100 * len(sorted_values)) - 1)]


def summarize(records: Sequence[ReplayRecord]) -> dict[str, Any]:
    """Return the success rate, latency percentiles and counts of a replay."""
    served = [r for r in records if r.success]
    latencies = sorted(r.total_s for r in served)
    return {
        "queries": len(records),
        "succeeded": len(served),
        "success_rate": len(served) / len(records) if records else 0.0,
        "latency_s": {p: _percentile(latencies, q) for p, q in (("p50", 50), ("p90", 90), ("p99", 99), ("max", 100))},
        "cold_starts": sum(1 for r in records if r.cold_start),
        "resent": sum(1 for r in records if r.attempts > 1),
        "per_model": dict(Counter(r.model for r in served)),
        "per_tier": dict(Counter(r.tier for r in served)),
        "errors": dict(Counter(r.error for r in records if not r.success).most_common(10)),
    }


def replay(
    base_url: str,
    queries: Sequence[Query],
    out_dir: Path,
    *,
    workers: int = 20,
    max_tokens: int = 256,
    give_up_s: float = 3600.0,
    post: Post = _post,
    progress_every: int = 50,
) -> Path:
    """Send the queries to the gateway with a pool of workers, write the outputs and return their path stem."""
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"replay_{datetime.now():%Y%m%d_%H%M%S}"
    log.info("Sending %s queries to %s with %d workers -> %s.jsonl", f"{len(queries):,}", base_url, workers, stem)
    records: list[ReplayRecord] = []
    with (
        Path(f"{stem}.jsonl").open("w", encoding="utf-8") as f,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [
            pool.submit(send, base_url, q, max_tokens=max_tokens, give_up_s=give_up_s, post=post) for q in queries
        ]
        for fut in as_completed(futures):
            record = fut.result()
            f.write(record.to_json() + "\n")
            f.flush()
            records.append(record)
            if len(records) % progress_every == 0:
                log.info("[%d/%d] %d answered", len(records), len(queries), sum(r.success for r in records))
    summary = summarize(records)
    try:
        summary["gateway_stats"] = requests.get(base_url.rstrip("/") + "/stats", timeout=30).json()
    except (requests.RequestException, ValueError) as e:
        log.warning("Could not read the gateway's /stats: %s", e)
    Path(f"{stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    log.info(
        "%d of %d answered; p50 %.1f s, p90 %.1f s; %d cold starts",
        summary["succeeded"],
        summary["queries"],
        summary["latency_s"]["p50"] or 0.0,
        summary["latency_s"]["p90"] or 0.0,
        summary["cold_starts"],
    )
    return stem
