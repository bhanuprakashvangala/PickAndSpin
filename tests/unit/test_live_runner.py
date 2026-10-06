"""The live runner, driven synchronously with fakes: no network and no cluster.

A manual clock stands in for time.monotonic, a scripted chat callable for call_vllm and a recording
actuator for the Kubernetes one. The classifier is HybridClassifier(None) and Pick sees one model per
tier, so every 'What is ...' query goes to llama3.2_1B. A cold model is still brought up in the
runner's own daemon thread, but process() waits for it, so each call returns after the load and the
clock readings are deterministic.

The run_live tests replace LiveRunner.start_reaper and LiveRunner.run with recorders, so they check
the wiring and the order of the steps without starting threads or sending queries. The last section
checks KubernetesActuator against a fake kubernetes package.
"""

import dataclasses
import gzip
import json
import logging
import os
import random
import re
import sys
import threading
import time
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest
import requests

from pickspin.config import DEFAULT_SPIN, MODELS, Tier
from pickspin.data import Query
from pickspin.errors import MissingDependencyError
from pickspin.live import runner as runner_module
from pickspin.live.actuator import KubernetesActuator
from pickspin.live.runner import LiveConfig, LiveRecord, LiveRunner, run_live
from pickspin.live.vllm import ChatResult, Endpoint, call_vllm
from pickspin.pick import HybridClassifier, Pick, Stage
from pickspin.spin import LatencySignal, ModelState, Spin

T0 = 1000.0  # the manual clock's start; a monotonic clock does not start at zero
LOAD_S = 30.0  # how long a load takes on the recording actuator
MODEL = "llama3.2_1B"  # where Pick sends every 'What is ...' query
TIERS_OF_ONE = {Tier.SIMPLE: ("llama3.2_1B",), Tier.MEDIUM: ("qwen2.5_7B",), Tier.COMPLEX: ("gemma3_27B",)}
ENDPOINTS: dict[str, Endpoint] = {
    m: {"base_url": f"http://{m}.invalid:8000/", "model": spec.hf_id, "deployment": f"vllm-{m}"}
    for m, spec in MODELS.items()
}
HEADERS = {"Authorization": "Bearer test-key"}
OK = ChatResult(True, "ok", 3, 0.25)
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


class ManualClock:
    """Stands in for time.monotonic: the time moves only when a test or a fake advances it."""

    def __init__(self, now: float = T0) -> None:
        self.now = now
        self._lock = threading.Lock()

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        with self._lock:
            self.now += seconds


class ScriptedChat:
    """Stands in for call_vllm: records its arguments, lets the reply's seconds pass on the clock, and
    returns the scripted replies in turn, then OK."""

    def __init__(self, clock: ManualClock, *replies: ChatResult) -> None:
        self.clock = clock
        self.replies = list(replies)
        self.calls: list[tuple[Endpoint, str, int, Mapping[str, str]]] = []
        self._lock = threading.Lock()

    def __call__(self, endpoint: Endpoint, query: str, max_tokens: int, headers: Mapping[str, str]) -> ChatResult:
        with self._lock:
            self.calls.append((endpoint, query, max_tokens, headers))
            reply = self.replies.pop(0) if self.replies else OK
        self.clock.advance(reply.seconds)
        return reply


class RecordingActuator:
    """Implements the Actuator protocol on the manual clock and records every call, in order.

    A load takes LOAD_S: wait_ready moves the clock on by that much, then raises `fail` if it is set.
    ready_replicas reports a ready pod for a model in `draining` that many more times. Once a test
    hands it the runner's scale locks, it also records whether the model's lock was held during scale
    and wait_ready.
    """

    def __init__(
        self,
        clock: ManualClock,
        *,
        fail: Exception | None = None,
        draining: Mapping[str, int] | None = None,
        calls: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.clock = clock
        self.fail = fail
        self.draining = dict(draining or {})
        self.calls: list[tuple[Any, ...]] = [] if calls is None else calls
        self.measured: dict[str, list[float]] = {}
        self.scale_lock: Mapping[str, threading.Lock] | None = None
        self.lock_held: list[tuple[str, str, bool]] = []
        self.scaled_to_zero = threading.Event()

    def _note_lock(self, what: str, model: str) -> None:
        if self.scale_lock is not None:
            self.lock_held.append((what, model, self.scale_lock[model].locked()))

    def scale(self, model: str, replicas: int) -> None:
        self.calls.append(("scale", model, replicas))
        self._note_lock("scale", model)
        if replicas == 0:
            self.scaled_to_zero.set()

    def ready_replicas(self, model: str) -> int:
        self.calls.append(("ready_replicas", model))
        left = self.draining.get(model, 0)
        if left:
            self.draining[model] = left - 1
            return 1
        return 0

    def wait_ready(self, model: str, t0: float) -> float:
        self.calls.append(("wait_ready", model, t0))
        self._note_lock("wait_ready", model)
        self.clock.advance(LOAD_S)
        if self.fail is not None:
            raise self.fail
        took = self.clock() - t0
        self.measured.setdefault(model, []).append(took)
        return took

    def load_estimate(self, model: str, now: float | None = None) -> float:
        self.calls.append(("load_estimate", model, now))
        return LOAD_S


def make_runner(
    clock: ManualClock,
    *,
    actuator: RecordingActuator | None = None,
    chat: ScriptedChat | None = None,
    **settings: Any,
) -> LiveRunner:
    """Wire a LiveRunner the way run_live does, except that Pick sees one model per tier."""
    paths = {"endpoints": Path("e.json"), "queries": Path("q.jsonl.gz"), "out_dir": Path("o"), "model_dir": Path("m")}
    config = LiveConfig(**(paths | {"scale_down_poll_s": 0.0} | settings))
    spin = Spin(
        cooldown_s=config.cooldown_s,
        scale_to_zero=not config.static,
        now=clock(),
        load_estimate=None if config.static or actuator is None else actuator.load_estimate,
    )
    pick = Pick(HybridClassifier(None), spin, config.latency_signal, rng=random.Random(config.seed), tiers=TIERS_OF_ONE)
    return LiveRunner(
        config,
        pick=pick,
        spin=spin,
        endpoints=ENDPOINTS,
        headers=HEADERS,
        actuator=actuator,
        clock=clock,
        chat=chat if chat is not None else ScriptedChat(clock),
    )


def query(i: int, text: str = "What is 2 + 2?") -> Query:
    return Query(f"q{i}", "gsm8k", text, "4", "math")


@pytest.fixture
def clock() -> ManualClock:
    return ManualClock()


@pytest.fixture
def runner_log(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog at INFO for the runner, unaffected by the handler the command line installs on 'pickspin'."""
    pickspin_logger = logging.getLogger("pickspin")
    monkeypatch.setattr(pickspin_logger, "handlers", [])
    monkeypatch.setattr(pickspin_logger, "propagate", True)
    caplog.set_level(logging.INFO, logger="pickspin.live.runner")
    return caplog


def runner_messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "pickspin.live.runner"]


# --- settings and records ------------------------------------------------------------------------


def test_live_config_defaults_are_the_old_constants() -> None:
    config = LiveConfig(Path("e.json"), Path("q.jsonl.gz"), Path("o"), Path("m"), api_key="secret-key")
    assert config.namespace == "pick-and-spin"
    assert (config.workers, config.limit, config.max_tokens, config.static, config.seed) == (250, None, 256, False, 0)
    assert config.latency_signal is LatencySignal.SPIN
    assert config.cooldown_s == DEFAULT_SPIN.cooldown_s == 300.0
    assert (config.reaper_interval_s, config.scale_down_poll_s, config.progress_every) == (5.0, 2.0, 2000)
    assert "secret-key" not in repr(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.workers = 1  # type: ignore[misc]


def test_records_serialize_like_the_old_dicts() -> None:
    record = LiveRecord(
        id="q7",
        benchmark="truthfulqa",
        tier=Tier.COMPLEX,
        stage=Stage.DISTILBERT,
        model="gemma3_27B",
        cold_start=True,
        waited_for_load=True,
        wait_s=95.123,
        latency=1.5,
        total_latency=96.623,
        success=True,
        tokens=42,
        response="Grüße, 你好",
        error="",
    )
    old = {  # what v1.1.0 wrote for the same query: a dict of plain strings, numbers and booleans
        "id": "q7",
        "benchmark": "truthfulqa",
        "tier": "COMPLEX",
        "stage": "distilbert",
        "model": "gemma3_27B",
        "cold_start": True,
        "waited_for_load": True,
        "wait_s": 95.123,
        "latency": 1.5,
        "total_latency": 96.623,
        "success": True,
        "tokens": 42,
        "response": "Grüße, 你好",
        "error": "",
    }
    line = record.to_json()
    assert line == json.dumps(old, ensure_ascii=False)
    assert "Grüße, 你好" in line  # non-ASCII text is kept, not escaped
    assert list(json.loads(line)) == RECORD_KEYS
    assert [f.name for f in dataclasses.fields(LiveRecord)] == RECORD_KEYS
    with pytest.raises(dataclasses.FrozenInstanceError):
        record.success = False  # type: ignore[misc]


# --- one query at a time ---------------------------------------------------------------------------


def test_a_static_runner_never_calls_the_actuator(clock: ManualClock) -> None:
    actuator = RecordingActuator(clock)
    runner = make_runner(clock, actuator=actuator, static=True)
    assert all(runner.is_ready(m) for m in MODELS)
    runner.prepare()
    runner.start_reaper()
    record = runner.process(query(1))
    assert record.model == MODEL
    assert record.cold_start is False
    assert record.waited_for_load is False
    assert (record.wait_s, record.latency, record.total_latency) == (0.0, 0.25, 0.25)
    clock.advance(100 * DEFAULT_SPIN.cooldown_s)
    assert runner.reap_once() == []
    assert actuator.calls == []
    assert runner.spin.status(MODEL) is ModelState.WARM


def test_only_a_static_run_may_go_without_an_actuator(clock: ManualClock) -> None:
    assert make_runner(clock, static=True).process(query(1)).success
    with pytest.raises(ValueError, match="needs an actuator"):
        make_runner(clock)


def test_the_cold_path_brings_the_model_up_once(clock: ManualClock) -> None:
    actuator = RecordingActuator(clock)
    chat = ScriptedChat(clock)
    runner = make_runner(clock, actuator=actuator, chat=chat)
    actuator.scale_lock = runner.scale_lock
    assert not runner.is_ready(MODEL)

    record = runner.process(query(1))

    # Spin counts the query first, which asks for the load estimate; then the bring-up scales the model
    # with its lock held and waits for it with the lock released, timing the load from the arrival.
    assert actuator.calls == [("load_estimate", MODEL, T0), ("scale", MODEL, 1), ("wait_ready", MODEL, T0)]
    assert actuator.lock_held == [("scale", MODEL, True), ("wait_ready", MODEL, False)]
    assert record.cold_start is True
    assert record.waited_for_load is True
    assert (record.tier, record.stage, record.model) == (Tier.SIMPLE, Stage.KEYWORD, MODEL)
    assert (record.wait_s, record.latency, record.total_latency) == (30.0, 0.25, 30.25)
    assert chat.calls == [(ENDPOINTS[MODEL], "What is 2 + 2?", 256, HEADERS)]
    assert runner.spin.status(MODEL) is ModelState.WARM
    assert runner.is_ready(MODEL)
    usage = runner.spin.summary(clock()).per_model[MODEL]
    assert (usage.cold_starts, usage.loading_gpu_hours) == (1, LOAD_S / 3600)
    assert runner.pick.sampler.model_ab[MODEL] == [2.0, 1.0]

    # The model is WARM now, so the next query neither scales nor waits.
    again = runner.process(query(2))
    assert (again.cold_start, again.waited_for_load, again.wait_s) == (False, False, 0.0)
    assert len(actuator.calls) == 3


def test_a_query_that_finds_its_model_loading_waits_without_a_second_bring_up(clock: ManualClock) -> None:
    actuator = RecordingActuator(clock)
    runner = make_runner(clock, actuator=actuator)
    assert runner.spin.request(MODEL, clock()) is ModelState.COLD  # an earlier query started the load
    waiting = threading.Event()

    class SignallingEvent(threading.Event):
        def wait(self, timeout: float | None = None) -> bool:
            waiting.set()
            return super().wait(timeout)

    runner.loads[MODEL].done = SignallingEvent()
    records: list[LiveRecord] = []
    worker = threading.Thread(target=lambda: records.append(runner.process(query(2))), daemon=True)
    worker.start()
    assert waiting.wait(timeout=10)
    assert runner.spin.status(MODEL) is ModelState.LOADING
    runner.bring_up(MODEL)  # the earlier query's bring-up finishes
    worker.join(timeout=10)

    (record,) = records
    assert (record.cold_start, record.waited_for_load) == (False, True)
    assert record.wait_s == LOAD_S
    assert actuator.calls.count(("scale", MODEL, 1)) == 1


def test_a_failed_load_still_sends_the_query(clock: ManualClock, runner_log: pytest.LogCaptureFixture) -> None:
    actuator = RecordingActuator(clock, fail=TimeoutError(f"{MODEL} not ready after 1200s"))
    chat = ScriptedChat(clock, ChatResult(False, "ConnectionError", 0, 0.01))
    runner = make_runner(clock, actuator=actuator, chat=chat)

    record = runner.process(query(1))

    assert runner.spin.status(MODEL) is ModelState.WARM
    assert runner.loads[MODEL].done.is_set()
    assert not runner.loads[MODEL].ok
    assert not runner.is_ready(MODEL)
    assert len(chat.calls) == 1
    assert (record.cold_start, record.waited_for_load, record.wait_s) == (True, True, LOAD_S)
    assert (record.success, record.response, record.error) == (False, "", "ConnectionError")
    assert actuator.measured == {}
    assert (
        "pickspin.live.runner",
        logging.WARNING,
        f"Loading {MODEL} failed: {MODEL} not ready after 1200s",
    ) in runner_log.record_tuples


def test_process_truncates_responses_and_records_failures(clock: ManualClock) -> None:
    reply = "Привет! " + "x" * 600
    chat = ScriptedChat(clock, ChatResult(True, reply, 12, 1.5), ChatResult(False, "HTTP 503", 0, 0.5))
    runner = make_runner(clock, chat=chat, static=True)

    ok = runner.process(query(1))
    assert ok.success
    assert ok.response == reply[:500]
    assert len(ok.response) == 500
    assert (ok.tokens, ok.latency, ok.total_latency, ok.error) == (12, 1.5, 1.5, "")
    assert "Привет!" in ok.to_json()

    failed = runner.process(query(2))
    assert not failed.success
    assert (failed.response, failed.error, failed.tokens, failed.latency) == ("", "HTTP 503", 0, 0.5)
    assert runner.pick.sampler.model_ab[MODEL] == [2.0, 2.0]  # one success and one failure
    assert runner.pick.sampler.tier_ab[Tier.SIMPLE] == [2.0, 2.0]

    unmatched = runner.process(query(3, "Tell me a story"))  # no keyword list matches
    assert (unmatched.tier, unmatched.stage, unmatched.model) == (Tier.MEDIUM, Stage.DEFAULT, "qwen2.5_7B")


# --- scaling to zero -------------------------------------------------------------------------------


def test_prepare_scales_every_model_to_zero_then_waits_until_none_is_ready(
    clock: ManualClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    actuator = RecordingActuator(clock, draining={"gemma2_2B": 2})
    runner = make_runner(clock, actuator=actuator, scale_down_poll_s=2.0)
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)

    runner.prepare()

    first_three = list(MODELS)[:3]  # each poll stops at gemma2_2B while it still has a ready pod
    assert actuator.calls == [
        *(("scale", m, 0) for m in MODELS),
        *(("ready_replicas", m) for m in first_three),
        *(("ready_replicas", m) for m in first_three),
        *(("ready_replicas", m) for m in MODELS),
    ]
    assert sleeps == [2.0, 2.0]


def test_reap_once_scales_a_model_idle_for_the_cooldown_to_zero(clock: ManualClock) -> None:
    actuator = RecordingActuator(clock)
    runner = make_runner(clock, actuator=actuator)
    actuator.scale_lock = runner.scale_lock
    runner.process(query(1))  # a cold start; the model is idle from T0 + 30.25 on

    clock.advance(DEFAULT_SPIN.cooldown_s - 1)
    assert runner.reap_once() == []
    assert ("scale", MODEL, 0) not in actuator.calls
    clock.advance(1)
    assert runner.reap_once() == [MODEL]
    assert actuator.calls[-1] == ("scale", MODEL, 0)
    assert actuator.lock_held[-1] == ("scale", MODEL, True)
    assert runner.spin.status(MODEL) is ModelState.COLD
    assert not runner.is_ready(MODEL)
    assert runner.spin.summary(clock()).per_model[MODEL].gpu_hours == pytest.approx(330.25 / 3600)

    # The next query finds the model COLD again and brings it back up.
    assert runner.process(query(2)).cold_start
    assert actuator.calls.count(("scale", MODEL, 1)) == 2


def test_the_reaper_thread_reaps_until_stopped(clock: ManualClock) -> None:
    actuator = RecordingActuator(clock)
    runner = make_runner(clock, actuator=actuator, reaper_interval_s=0.001)
    runner.process(query(1))
    clock.advance(DEFAULT_SPIN.cooldown_s)
    before = set(threading.enumerate())

    runner.start_reaper()

    (reaper,) = set(threading.enumerate()) - before
    assert reaper.daemon
    assert actuator.scaled_to_zero.wait(timeout=10)
    runner.stop.set()
    reaper.join(timeout=10)
    assert not reaper.is_alive()
    assert runner.spin.status(MODEL) is ModelState.COLD


# --- whole runs ------------------------------------------------------------------------------------


def test_run_writes_the_records_and_the_summary(
    clock: ManualClock, tmp_path: Path, runner_log: pytest.LogCaptureFixture
) -> None:
    actuator = RecordingActuator(clock)
    runner = make_runner(clock, actuator=actuator, out_dir=tmp_path / "live", workers=1, progress_every=2)

    stem = runner.run([query(i) for i in range(5)])

    assert stem.parent == tmp_path / "live"
    assert re.fullmatch(r"pick_spin_\d{8}_\d{6}", stem.name)
    lines = Path(f"{stem}.jsonl").read_bytes().decode("utf-8").split(os.linesep)  # text mode line ends
    assert lines[-1] == ""
    rows = [json.loads(line) for line in lines[:-1]]
    assert [r["id"] for r in rows] == [f"q{i}" for i in range(5)]  # one worker: completion order = submission
    assert all(list(r) == RECORD_KEYS for r in rows)
    assert [r["cold_start"] for r in rows] == [True, False, False, False, False]

    expected = runner.spin.summary(clock()).to_dict() | {"measured_load_s": {MODEL: [LOAD_S]}}
    summary = Path(f"{stem}_summary.json").read_bytes()
    assert summary == json.dumps(expected, indent=1).replace("\n", os.linesep).encode("utf-8")
    assert list(json.loads(summary)) == SUMMARY_KEYS
    assert list(expected["per_model"]) == list(MODELS)
    assert runner.stop.is_set()
    assert runner_messages(runner_log) == [
        f"Routing 5 queries with 1 workers (scale to zero) -> {stem}.jsonl",
        "[2] 1 cold starts so far",
        "[4] 1 cold starts so far",
        f"{expected['gpu_hours']:.2f} GPU-hours, utilization {100 * expected['gpu_utilization']:.1f}%, "
        f"1 cold starts ({100 * expected['cold_start_rate']:.2f}% of queries)",
    ]


def test_a_static_run_on_many_workers(clock: ManualClock, tmp_path: Path, runner_log: pytest.LogCaptureFixture) -> None:
    runner = make_runner(clock, static=True, out_dir=tmp_path, workers=4, progress_every=1000)

    stem = runner.run([query(i) for i in range(2000)])

    rows = [json.loads(line) for line in Path(f"{stem}.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(r["id"] for r in rows) == sorted(f"q{i}" for i in range(2000))
    assert not any(r["cold_start"] or r["waited_for_load"] for r in rows)
    summary = json.loads(Path(f"{stem}_summary.json").read_text(encoding="utf-8"))
    assert (summary["cold_starts"], summary["measured_load_s"]) == (0, {})
    assert runner_messages(runner_log)[:3] == [
        f"Routing 2,000 queries with 4 workers (static) -> {stem}.jsonl",
        "[1,000] 0 cold starts so far",
        "[2,000] 0 cold starts so far",
    ]


# --- run_live ----------------------------------------------------------------------------------------


@pytest.fixture
def live_inputs(tmp_path: Path) -> tuple[Path, Path]:
    """An endpoint map and a queries file of ten 'What is ...' queries, both in tmp_path."""
    endpoints = tmp_path / "endpoints.json"
    endpoints.write_text(json.dumps(ENDPOINTS), encoding="utf-8")
    queries = tmp_path / "queries.jsonl.gz"
    with gzip.open(queries, "wt", encoding="utf-8") as f:
        for i in range(10):
            record = {"id": f"q{i}", "benchmark": "gsm8k", "query": f"What is {i} + {i}?", "ground_truth": str(2 * i)}
            f.write(json.dumps(record | {"query_type": "math"}) + "\n")
    return endpoints, queries


def shuffled_ids(seed: int) -> list[str]:
    ids = [f"q{i}" for i in range(10)]
    random.Random(seed).shuffle(ids)
    return ids


class Wiring:
    """What run_live did: its steps, interleaved with the actuator's calls, and the runner it built."""

    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []
        self.runner: LiveRunner | None = None


@pytest.fixture
def wiring(monkeypatch: pytest.MonkeyPatch) -> Wiring:
    """Records start_reaper, load_queries and run in place of starting the reaper and sending queries."""
    wiring = Wiring()
    real_load_queries = runner_module.load_queries

    def load_queries(path: Path) -> list[Query]:
        wiring.events.append(("load_queries", path))
        return real_load_queries(path)

    def run(self: LiveRunner, queries: list[Query]) -> Path:
        wiring.events.append(("run", [q.id for q in queries]))
        wiring.runner = self
        return self.config.out_dir / "stem"

    monkeypatch.setattr(runner_module, "load_queries", load_queries)
    monkeypatch.setattr(LiveRunner, "start_reaper", lambda self: wiring.events.append(("start_reaper",)))
    monkeypatch.setattr(LiveRunner, "run", run)
    return wiring


def test_run_live_wires_the_parts_in_the_old_order(
    clock: ManualClock, live_inputs: tuple[Path, Path], wiring: Wiring, tmp_path: Path
) -> None:
    endpoints, queries = live_inputs
    actuator = RecordingActuator(clock, calls=wiring.events)
    classifier = HybridClassifier(None)
    config = LiveConfig(
        endpoints=endpoints,
        queries=queries,
        out_dir=tmp_path / "out",
        model_dir=tmp_path / "unused",
        limit=4,
        latency_signal=LatencySignal.OBSERVED,
        seed=7,
        cooldown_s=0.5,
        api_key="secret",
    )

    stem = run_live(config, actuator=actuator, classifier=classifier, clock=clock)

    prepare = [*(("scale", m, 0) for m in MODELS), *(("ready_replicas", m) for m in MODELS)]
    # The reaper starts before the queries are loaded.
    assert wiring.events == [*prepare, ("start_reaper",), ("load_queries", queries), ("run", shuffled_ids(7)[:4])]
    runner = wiring.runner
    assert runner is not None
    assert stem == tmp_path / "out" / "stem"
    assert runner.config is config
    assert runner.actuator is actuator
    assert runner.endpoints == ENDPOINTS
    assert runner.headers == {"Authorization": "Bearer secret"}
    assert runner.clock is clock
    assert runner.chat is call_vllm
    assert (runner.spin.t0, runner.spin.scale_to_zero, runner.spin.cooldown_s) == (T0, True, 0.5)
    assert runner.spin.load_estimate == actuator.load_estimate
    assert runner.pick.classifier is classifier
    assert runner.pick.signal is LatencySignal.OBSERVED
    assert runner.pick.sampler.rng.getstate() == random.Random(7).getstate()  # its own stream, unused so far
    assert not any(runner.is_ready(m) for m in MODELS)


@pytest.mark.parametrize(("limit", "kept"), [(None, 10), (0, 10), (3, 3), (25, 10)])
def test_run_live_shuffles_with_the_seed_then_keeps_the_first_limit_queries(
    clock: ManualClock, live_inputs: tuple[Path, Path], wiring: Wiring, tmp_path: Path, limit: int | None, kept: int
) -> None:
    endpoints, queries = live_inputs
    config = LiveConfig(endpoints, queries, tmp_path, tmp_path, static=True, limit=limit)
    run_live(config, classifier=HybridClassifier(None), clock=clock)
    assert wiring.events[-1] == ("run", shuffled_ids(0)[:kept])


def test_a_static_run_counts_the_classifier_load_and_never_scales(
    clock: ManualClock,
    live_inputs: tuple[Path, Path],
    wiring: Wiring,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    endpoints, queries = live_inputs
    actuator = RecordingActuator(clock, calls=wiring.events)
    loaded_from: list[Path] = []

    def from_pretrained(model_dir: Path, **kwargs: Any) -> HybridClassifier:
        loaded_from.append(model_dir)
        clock.advance(7.0)  # loading DistilBERT takes a while
        return HybridClassifier(None)

    monkeypatch.setattr(HybridClassifier, "from_pretrained", from_pretrained)
    config = LiveConfig(endpoints, queries, tmp_path, tmp_path / "model", static=True)

    run_live(config, actuator=actuator, clock=clock)

    runner = wiring.runner
    assert runner is not None
    assert loaded_from == [tmp_path / "model"]
    assert runner.actuator is None
    assert wiring.events == [("start_reaper",), ("load_queries", queries), ("run", shuffled_ids(0))]
    assert all(runner.spin.status(m) is ModelState.WARM for m in MODELS)
    # Spin's clock started before the classifier loaded, so the load counts in the static GPU-hours.
    assert runner.spin.t0 == T0
    assert runner.spin.summary(clock()).gpu_hours == pytest.approx(len(MODELS) * 7.0 / 3600)


# --- KubernetesActuator ------------------------------------------------------------------------------


class FakeConfigError(Exception):
    """Stands in for kubernetes.config.ConfigException."""


class FakeAppsV1Api:
    """Stands in for kubernetes.client.AppsV1Api: records scale patches and reports ready replicas."""

    def __init__(self) -> None:
        self.scaled: list[tuple[str, str, dict[str, Any]]] = []
        self.ready: dict[str, int] = {}  # by Deployment; the API reports None while no pod is ready

    def patch_namespaced_deployment_scale(self, name: str, namespace: str, body: dict[str, Any]) -> None:
        self.scaled.append((name, namespace, body))

    def read_namespaced_deployment_status(self, name: str, namespace: str) -> types.SimpleNamespace:
        return types.SimpleNamespace(status=types.SimpleNamespace(ready_replicas=self.ready.get(name)))


@pytest.fixture
def fake_kubernetes(monkeypatch: pytest.MonkeyPatch) -> types.SimpleNamespace:
    """Installs a fake kubernetes package. Returns its AppsV1Api, the config loaders called and whether
    the in-cluster configuration is available (it is not, unless a test says so)."""
    k8s = types.SimpleNamespace(apps=FakeAppsV1Api(), config_calls=[], in_cluster=False)

    def load_incluster_config() -> None:
        k8s.config_calls.append("load_incluster_config")
        if not k8s.in_cluster:
            raise FakeConfigError("Service host/port is not set.")

    client = types.ModuleType("kubernetes.client")
    client.AppsV1Api = lambda: k8s.apps  # type: ignore[attr-defined]
    config = types.ModuleType("kubernetes.config")
    config.ConfigException = FakeConfigError  # type: ignore[attr-defined]
    config.load_incluster_config = load_incluster_config  # type: ignore[attr-defined]
    config.load_kube_config = lambda: k8s.config_calls.append("load_kube_config")  # type: ignore[attr-defined]
    package = types.ModuleType("kubernetes")
    package.client, package.config = client, config  # type: ignore[attr-defined]
    for name, module in {"kubernetes": package, "kubernetes.client": client, "kubernetes.config": config}.items():
        monkeypatch.setitem(sys.modules, name, module)
    return k8s


def test_the_kubernetes_actuator_needs_the_live_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "kubernetes", None)
    monkeypatch.setitem(sys.modules, "kubernetes.client", None)
    with pytest.raises(MissingDependencyError, match=r"pip install 'pick-and-spin\[live\]'"):
        KubernetesActuator(ENDPOINTS)


@pytest.mark.parametrize(
    ("in_cluster", "loaders"),
    [(True, ["load_incluster_config"]), (False, ["load_incluster_config", "load_kube_config"])],
)
def test_the_kubernetes_actuator_prefers_the_in_cluster_config(
    fake_kubernetes: types.SimpleNamespace, in_cluster: bool, loaders: list[str]
) -> None:
    fake_kubernetes.in_cluster = in_cluster
    actuator = KubernetesActuator(ENDPOINTS)
    assert fake_kubernetes.config_calls == loaders
    assert actuator.apps is fake_kubernetes.apps
    assert (actuator.namespace, actuator.poll_s, actuator.timeout_s) == ("pick-and-spin", 2.0, 1200)
    assert actuator.measured == {}


def test_the_kubernetes_actuator_scales_deployments(fake_kubernetes: types.SimpleNamespace) -> None:
    actuator = KubernetesActuator(ENDPOINTS, "research")
    actuator.scale("gemma3_27B", 1)
    actuator.scale("gemma3_27B", 0)
    assert fake_kubernetes.apps.scaled == [
        ("vllm-gemma3_27B", "research", {"spec": {"replicas": 1}}),
        ("vllm-gemma3_27B", "research", {"spec": {"replicas": 0}}),
    ]
    assert actuator.ready_replicas("gemma3_27B") == 0  # None from the API
    fake_kubernetes.apps.ready["vllm-gemma3_27B"] = 2
    assert actuator.ready_replicas("gemma3_27B") == 2


def test_the_kubernetes_actuator_asks_vllm_for_its_health(
    fake_kubernetes: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    replies: list[int | Exception] = [200, 503, requests.ConnectionError("refused")]
    urls: list[tuple[str, float]] = []

    def get(url: str, timeout: float) -> types.SimpleNamespace:
        urls.append((url, timeout))
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return types.SimpleNamespace(status_code=reply)

    monkeypatch.setattr(requests, "get", get)
    actuator = KubernetesActuator(ENDPOINTS)
    assert [actuator.healthy("qwen2.5_7B") for _ in range(3)] == [True, False, False]
    assert urls == [("http://qwen2.5_7B.invalid:8000/health", 5)] * 3


def test_wait_ready_polls_until_a_pod_is_ready_and_vllm_is_healthy(
    fake_kubernetes: types.SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    statuses = [503, 200]
    monkeypatch.setattr(requests, "get", lambda url, timeout: types.SimpleNamespace(status_code=statuses.pop(0)))
    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", sleeps.append)
    actuator = KubernetesActuator(ENDPOINTS, poll_s=0.25, timeout_s=60)
    fake_kubernetes.apps.ready["vllm-gemma2_9B"] = 1

    took = actuator.wait_ready("gemma2_9B", time.monotonic() - 12.0)

    assert (statuses, sleeps) == ([], [0.25])
    assert 12.0 <= took < 60.0
    assert actuator.measured == {"gemma2_9B": [took]}


def test_wait_ready_gives_up_after_the_timeout(fake_kubernetes: types.SimpleNamespace) -> None:
    actuator = KubernetesActuator(ENDPOINTS, poll_s=0.0, timeout_s=60)
    with pytest.raises(TimeoutError, match=r"^llama3\.2_1B not ready after 60s$"):
        actuator.wait_ready("llama3.2_1B", time.monotonic() - 61.0)  # no pod ever becomes ready
    assert actuator.measured == {}


def test_load_estimate_is_the_mean_measured_load_time(fake_kubernetes: types.SimpleNamespace) -> None:
    actuator = KubernetesActuator(ENDPOINTS)
    assert actuator.load_estimate("gemma3_27B") == 95.0  # the stated cold-start time before any load
    assert type(actuator.load_estimate("gemma3_27B")) is float
    actuator.measured["gemma3_27B"] = [10.0, 20.0, 40.0]
    assert actuator.load_estimate("gemma3_27B", 123.0) == sum([10.0, 20.0, 40.0]) / 3
    assert actuator.load_estimate("qwen2.5_14B") == 65.0


def test_run_live_scales_through_a_kubernetes_actuator_by_default(
    clock: ManualClock,
    live_inputs: tuple[Path, Path],
    wiring: Wiring,
    fake_kubernetes: types.SimpleNamespace,
    tmp_path: Path,
) -> None:
    endpoints, queries = live_inputs
    config = LiveConfig(endpoints, queries, tmp_path, tmp_path, namespace="research")

    run_live(config, classifier=HybridClassifier(None), clock=clock)

    runner = wiring.runner
    assert runner is not None
    actuator = runner.actuator
    assert isinstance(actuator, KubernetesActuator)
    assert (actuator.namespace, actuator.endpoints) == ("research", ENDPOINTS)
    assert runner.spin.load_estimate == actuator.load_estimate
    assert fake_kubernetes.apps.scaled == [(f"vllm-{m}", "research", {"spec": {"replicas": 0}}) for m in MODELS]
