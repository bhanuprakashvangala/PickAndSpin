"""The live runner end to end, against stub vLLM servers on localhost and a fake cluster.

Ported from tests/test_live.py at v1.1.0. run_live runs as in a real experiment, with its worker pool,
bring-up threads, HTTP client and output files; only the cluster is fake (the stub_cluster fixture in
conftest.py). The old test shortened T_cooldown by monkeypatching the configuration and replaced the
actuator and the classifier by monkeypatching the runner module; LiveConfig(cooldown_s=0.5) and the
actuator and classifier arguments of run_live replace all three.

A run of 60 queries usually ends before the reaper's first pass, 5 seconds in, so the last test drives
a LiveRunner with a fast reaper to show the whole cycle: a model scaled to zero after T_cooldown idle
is brought up again by the next query routed to it.
"""

import json
import random
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pickspin.config import MODELS, TIERS, Tier
from pickspin.data import Query, load_queries
from pickspin.live.runner import LiveConfig, LiveRunner, run_live
from pickspin.paths import Paths
from pickspin.pick import HybridClassifier, Pick, Stage
from pickspin.spin import ModelState, Spin

LIMIT = 60  # queries per run

RECORD_KEYS = [
    "id",
    "benchmark",
    "tier",
    "stage",
    "model",
    "cold_start",
    "waited_for_load",
    "wait_s",
    "latency",
    "total_latency",
    "success",
    "tokens",
    "response",
    "error",
]
SUMMARY_KEYS = [
    "gpu_hours",
    "busy_gpu_hours",
    "gpu_utilization",
    "cold_starts",
    "cold_start_rate",
    "per_model",
    "measured_load_s",
]
PER_MODEL_KEYS = ["gpu_hours", "busy_gpu_hours", "loading_gpu_hours", "cold_starts"]


def live_config(endpoints: Path, root: Path, tmp_path: Path, **settings: Any) -> LiveConfig:
    """The ported test's run: 8 workers and 60 queries, with the outputs written to tmp_path.

    The classifier is passed to run_live, so model_dir is never read.
    """
    return LiveConfig(
        endpoints=endpoints,
        queries=Paths(root).queries,
        out_dir=tmp_path,
        model_dir=tmp_path / "unused",
        workers=8,
        limit=LIMIT,
        **settings,
    )


def read_outputs(stem: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read a run's records, in file order, and its summary."""
    with Path(f"{stem}.jsonl").open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    with Path(f"{stem}_summary.json").open(encoding="utf-8") as f:
        summary = json.load(f)
    return rows, summary


def test_live_runner_scales_models_up_and_down(stub_cluster, released_data, tmp_path):
    config = live_config(stub_cluster.endpoints_file, released_data, tmp_path, cooldown_s=0.5)
    stem = run_live(config, actuator=stub_cluster.actuator, classifier=HybridClassifier(None))

    rows, summary = read_outputs(stem)
    calls = stub_cluster.cluster.calls
    assert len(rows) == LIMIT
    assert all(r["success"] for r in rows)
    cold = [r for r in rows if r["cold_start"]]
    assert cold
    assert all(r["wait_s"] >= stub_cluster.load_s * 0.9 for r in cold)
    assert summary["cold_starts"] == len(cold)
    assert calls[: len(MODELS)] == [(m, 0) for m in MODELS]  # the run starts cold
    assert any(replicas == 1 for _, replicas in calls)


def test_live_run_writes_the_old_record_and_summary_formats(stub_cluster, released_data, tmp_path):
    config = live_config(stub_cluster.endpoints_file, released_data, tmp_path, cooldown_s=0.5)
    stem = run_live(config, actuator=stub_cluster.actuator, classifier=HybridClassifier(None))

    assert stem.parent == config.out_dir
    assert stem.name.startswith("pick_spin_")
    rows, summary = read_outputs(stem)
    assert [list(r) for r in rows] == [RECORD_KEYS] * LIMIT
    assert list(summary) == SUMMARY_KEYS
    assert list(summary["per_model"]) == list(MODELS)
    assert [list(usage) for usage in summary["per_model"].values()] == [PER_MODEL_KEYS] * len(MODELS)
    cold = Counter(r["model"] for r in rows if r["cold_start"])
    assert {m: usage["cold_starts"] for m, usage in summary["per_model"].items()} == {m: cold[m] for m in MODELS}
    # Each cold start brought its model up once and measured the load.
    scaled_up = Counter(m for m, replicas in stub_cluster.cluster.calls if replicas == 1)
    assert scaled_up == cold
    assert summary["measured_load_s"] == stub_cluster.actuator.measured
    assert Counter({m: len(times) for m, times in summary["measured_load_s"].items()}) == cold


def test_live_run_routes_the_seeded_sample_and_sends_each_query_to_its_model(stub_cluster, released_data, tmp_path):
    config = live_config(stub_cluster.endpoints_file, released_data, tmp_path, cooldown_s=0.5, seed=3)
    stem = run_live(config, actuator=stub_cluster.actuator, classifier=HybridClassifier(None))
    rows, _ = read_outputs(stem)

    # The queries are the first LIMIT of the file after a shuffle with random.Random(seed).
    queries = load_queries(config.queries)
    random.Random(config.seed).shuffle(queries)
    sample = {q.id: q for q in queries[:LIMIT]}
    assert sorted(r["id"] for r in rows) == sorted(sample)

    classifier = HybridClassifier(None)
    for r in rows:
        query = sample[r["id"]]
        assert r["benchmark"] == query.benchmark
        assert (r["tier"], r["stage"]) == classifier.classify(query.query)
        assert r["model"] in TIERS[Tier(r["tier"])]
        assert (r["tokens"], r["response"], r["error"]) == (3, "ok", "")

    # Every query reached the server of the model it was routed to, once, with the run's max_tokens
    # and no Authorization header (the run has no API key).
    posts = [(m, request) for m, server in stub_cluster.servers.items() for request in server.received]
    posts = [(m, request) for m, request in posts if request.method == "POST"]
    sent = Counter((m, request.body["messages"][0]["content"]) for m, request in posts)
    assert sent == Counter((r["model"], sample[r["id"]].query) for r in rows)
    assert {request.body["max_tokens"] for _, request in posts} == {config.max_tokens}
    assert not any("Authorization" in request.headers for _, request in posts)


def test_static_live_run_scales_nothing_and_records_no_cold_starts(stub_cluster, released_data, tmp_path):
    stub_cluster.cluster.start_all()  # a static deployment keeps every model running
    config = live_config(stub_cluster.endpoints_file, released_data, tmp_path, static=True)
    stem = run_live(config, actuator=None, classifier=HybridClassifier(None))

    rows, summary = read_outputs(stem)
    assert len(rows) == LIMIT
    assert all(r["success"] for r in rows)
    assert not any(r["cold_start"] for r in rows)
    assert not any(r["waited_for_load"] for r in rows)
    assert summary["cold_starts"] == 0
    assert all(usage["cold_starts"] == 0 for usage in summary["per_model"].values())
    assert summary["measured_load_s"] == {}
    assert stub_cluster.cluster.calls == []


def wait_until(condition: Callable[[], bool], timeout_s: float = 10.0) -> bool:
    """Check the condition every 10 ms until it holds or timeout_s seconds pass; return whether it held."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.01)
    return True


def test_reaper_scales_an_idle_model_to_zero_and_the_next_query_brings_it_up_again(stub_cluster, tmp_path):
    model = "qwen2.5_7B"
    config = LiveConfig(
        endpoints=stub_cluster.endpoints_file,
        queries=tmp_path / "unused.jsonl.gz",
        out_dir=tmp_path,
        model_dir=tmp_path / "unused",
        cooldown_s=0.2,
        reaper_interval_s=0.02,
    )
    actuator, cluster = stub_cluster.actuator, stub_cluster.cluster
    spin = Spin(cooldown_s=config.cooldown_s, now=time.monotonic(), load_estimate=actuator.load_estimate)
    # Without a stage-2 model a query that no keyword list matches is MEDIUM, and here that tier holds one
    # model, so both queries go to it.
    pick = Pick(HybridClassifier(None), spin, rng=random.Random(0), tiers={**TIERS, Tier.MEDIUM: (model,)})
    runner = LiveRunner(config, pick=pick, spin=spin, endpoints=stub_cluster.endpoints, headers={}, actuator=actuator)
    query = Query("q1", "gsm8k", "Tell me a story.", "", "")

    runner.prepare()
    runner.start_reaper()
    try:
        first = runner.process(query)
        reaped = wait_until(lambda: (model, 0) in cluster.calls[len(MODELS) :])
        state_after_reap, ready_after_reap = spin.status(model), runner.is_ready(model)
        second = runner.process(query)
    finally:
        runner.stop.set()

    assert (first.model, first.tier, first.stage) == (model, Tier.MEDIUM, Stage.DEFAULT)
    assert reaped, f"the reaper did not scale the idle {model} to zero within 10 s"
    # The reaper marked the model COLD and closed it to queries before scaling it to zero.
    assert state_after_reap is ModelState.COLD
    assert not ready_after_reap
    for record in (first, second):
        assert (record.model, record.cold_start, record.waited_for_load, record.success) == (model, True, True, True)
        assert record.wait_s >= stub_cluster.load_s * 0.9
    # Up for the first query, down by the reaper, up again for the second (the reaper may since have
    # scaled the model down again).
    assert cluster.calls[len(MODELS) :][:3] == [(model, 1), (model, 0), (model, 1)]
    assert len(actuator.measured[model]) == 2
    usage = spin.summary(time.monotonic())
    assert (usage.cold_starts, usage.per_model[model].cold_starts) == (2, 2)
