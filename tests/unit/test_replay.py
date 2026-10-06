"""pickspin replay: the client that load-tests a running gateway."""

import json
import random
import threading

import requests

from pickspin.config import ModelSpec, Tier, tiers_of
from pickspin.data import Query
from pickspin.live.gateway import Gateway, serve
from pickspin.live.replay import ReplayRecord, replay, send, summarize
from pickspin.pick.classifier import HybridClassifier
from pickspin.pick.router import Pick
from pickspin.spin.lifecycle import Spin

QUERY = Query("q1", "gsm8k", "What is 2 + 2?", "4", "math")
SERVED = {
    "choices": [{"message": {"role": "assistant", "content": "4"}}],
    "usage": {"completion_tokens": 3},
    "pickspin": {"model": "tiny", "tier": "SIMPLE", "stage": "keyword", "cold_start": True, "wait_s": 40.0},
}


class FakeTime:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, seconds):
        self.t += seconds


def test_send_resends_while_the_model_loads_and_while_the_gateway_is_unreachable():
    clock = FakeTime()
    answers = [
        (503, {"Retry-After": "20"}, {"error": {"message": "tiny is still loading; retry in 20 s"}}),
        None,  # the gateway restarts: connection refused
        (200, {}, SERVED),
    ]
    sent = []

    def post(url, body, timeout_s):
        sent.append((url, body))
        answer = answers.pop(0)
        if answer is None:
            raise requests.ConnectionError("connection refused")
        clock.t += 1.0
        return answer

    record = send("http://gw:8080/", QUERY, max_tokens=64, post=post, sleep=clock.sleep, clock=clock)

    assert sent[0] == (
        "http://gw:8080/v1/chat/completions",
        {"model": "auto", "max_tokens": 64, "messages": [{"role": "user", "content": "What is 2 + 2?"}]},
    )
    assert (record.success, record.status, record.attempts) == (True, 200, 3)
    assert record.total_s == 1.0 + 20.0 + 30.0 + 1.0  # two answers, Retry-After, then the default 30 s
    assert (record.model, record.tier, record.cold_start, record.wait_s) == ("tiny", "SIMPLE", True, 40.0)
    assert (record.tokens, record.response, record.error) == (3, "4", "")


def test_send_gives_up_after_the_deadline():
    clock = FakeTime()

    def post(url, body, timeout_s):
        clock.t += 1.0
        return 503, {"Retry-After": "30"}, {"error": {"message": "big is still loading"}, "pickspin": {"model": "big"}}

    record = send("http://gw", QUERY, give_up_s=100, post=post, sleep=clock.sleep, clock=clock)

    assert (record.success, record.status, record.attempts, record.total_s) == (False, 503, 5, 125.0)
    assert (record.model, record.error, record.response) == ("big", "big is still loading", "")


def record(total_s, *, success=True, model="tiny", attempts=1, error=""):
    return ReplayRecord(
        id="q",
        benchmark="b",
        status=200 if success else 503,
        success=success,
        model=model,
        tier="SIMPLE",
        stage=None,
        cold_start=attempts > 1,
        wait_s=0.0,
        total_s=total_s,
        attempts=attempts,
        tokens=1,
        response="",
        error=error,
    )


def test_summarize_reports_nearest_rank_percentiles_and_counts():
    records = [record(float(t)) for t in range(10, 0, -1)]
    records += [record(99.0, success=False, error="tiny failed to load")] * 2
    records.append(record(3.0, model="mid", attempts=2))
    s = summarize(records)
    assert (s["queries"], s["succeeded"]) == (13, 11)
    assert s["success_rate"] == 11 / 13
    assert s["latency_s"] == {"p50": 5.0, "p90": 9.0, "p99": 10.0, "max": 10.0}
    assert (s["cold_starts"], s["resent"]) == (1, 1)
    assert s["per_model"] == {"tiny": 10, "mid": 1}
    assert s["errors"] == {"tiny failed to load": 2}
    assert summarize([])["latency_s"]["p50"] is None


def test_replay_against_a_running_gateway(tmp_path):
    catalog = {
        "tiny": ModelSpec("tiny", "Tiny", Tier.SIMPLE, "org/tiny", 1, 10, 1),
        "mid": ModelSpec("mid", "Mid", Tier.MEDIUM, "org/mid", 5, 20, 1),
        "big": ModelSpec("big", "Big", Tier.COMPLEX, "org/big", 20, 30, 1),
    }
    endpoints = {k: {"base_url": f"http://{k}:8000", "model": s.hf_id} for k, s in catalog.items()}
    spin = Spin(catalog, catalog=catalog, scale_to_zero=False)
    pick = Pick(HybridClassifier(None), spin, rng=random.Random(0), tiers=tiers_of(catalog), models=catalog)

    def upstream(url, body, headers, timeout):
        return 200, {"choices": [{"message": {"role": "assistant", "content": f"from {body['model']}"}}]}

    gateway = Gateway(
        pick=pick, spin=spin, catalog=catalog, endpoints=endpoints, actuator=None, static=True, post=upstream
    )
    server = serve(gateway, "127.0.0.1", 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    queries = [
        Query(f"q{i}", "mix", text, "", "")
        for i, text in enumerate(["What is 2 + 2?", "Prove that 2 is prime.", "Tell me a story."] * 2)
    ]
    try:
        stem = replay(f"http://127.0.0.1:{server.server_address[1]}", queries, tmp_path, workers=3)
    finally:
        server.shutdown()

    lines = [json.loads(line) for line in (tmp_path / f"{stem.name}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in lines) == [f"q{i}" for i in range(6)]
    assert all(r["success"] and r["response"].startswith("from org/") for r in lines)
    summary = json.loads((tmp_path / f"{stem.name}_summary.json").read_text(encoding="utf-8"))
    assert (summary["queries"], summary["succeeded"]) == (6, 6)
    assert summary["gateway_stats"]["requests_served"] == 6
    assert sum(summary["per_tier"].values()) == 6
