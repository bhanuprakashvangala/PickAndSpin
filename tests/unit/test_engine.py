"""The trace-driven simulator: its policies, its inputs and the discrete-event engine."""

import csv
import dataclasses
import gzip
import hashlib
import itertools
import json
import logging
import math
import random
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from pickspin.config import MODELS, TIER_ORDER, Tier
from pickspin.data import Query
from pickspin.errors import ConfigError
from pickspin.pick import HybridClassifier, Stage
from pickspin.simulation.engine import (
    ARRIVAL_SEED_OFFSET,
    SAMPLER_SEED_OFFSET,
    SHUFFLE_SEED_OFFSET,
    QueryRecord,
    SimulationResult,
    SimulationSettings,
    simulate,
)
from pickspin.simulation.inputs import (
    TierAssignment,
    TraceData,
    describe_stages,
    load_trace_data,
    query_tiers,
    read_tier_cache,
    write_tier_cache,
)
from pickspin.simulation.policies import POLICIES, Policy
from pickspin.spin import LatencySignal

# The make_tiny_trace fixture of tests/conftest.py: make_tiny_trace(n=40, latency=2.0) -> (data, tiers).
MakeTinyTrace = Callable[..., tuple[TraceData, dict[str, TierAssignment]]]


# --- ported from tests/test_pickspin.py at v1.1.0 ------------------------------------------------


def test_simulator_static_and_scale_to_zero(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace()
    static = simulate(Policy.STATIC, 0, data, tiers, SimulationSettings(workers=4))
    assert len(static.records) == 40
    assert static.usage.cold_starts == 0
    assert all(r.wait_s == 0 for r in static.records)
    assert static.usage.gpu_hours == pytest.approx(len(MODELS) * static.makespan_s / 3600)
    result = simulate(Policy.PICK_AND_SPIN, 0, data, tiers, SimulationSettings(workers=4))
    first = min(result.records, key=lambda r: r.order)
    assert result.usage.cold_starts >= 1
    assert first.cold_start
    assert first.wait_s >= MODELS[first.model].cold_start_s - 1e-6
    assert all(MODELS[r.model].tier is Tier.SIMPLE for r in result.records)


def test_sparse_load_scales_models_down(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=20)
    sparse = SimulationSettings(arrival_rate=1 / 1000)
    result = simulate(Policy.PICK_AND_SPIN, 0, data, tiers, sparse)
    static = simulate(Policy.STATIC, 0, data, tiers, sparse)
    assert result.usage.cold_starts >= 10
    assert result.usage.gpu_hours < 0.2 * static.usage.gpu_hours


def test_simulator_is_deterministic(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace()
    settings = SimulationSettings(arrival_rate=0.01)
    a = simulate(Policy.PICK_AND_SPIN, 3, data, tiers, settings)
    b = simulate(Policy.PICK_AND_SPIN, 3, data, tiers, settings)
    assert (a.usage.gpu_hours, a.usage.cold_starts) == (b.usage.gpu_hours, b.usage.cold_starts)
    assert a == b  # every record and all of the accounting repeat exactly


# --- the engine against v1.1.0 -------------------------------------------------------------------

BENCHMARKS = ("gsm8k", "mmlu_pro", "humaneval")


def varied_trace(n: int = 60) -> tuple[TraceData, dict[str, TierAssignment]]:
    """A synthetic trace that exercises every path of the engine.

    It has queries of every tier and both classifier stages, latencies from 0.25 s to 6.25 s that
    differ by query and model, failed executions, and (query, model) pairs the judge did not score.
    """
    queries = [Query(f"v{i}", BENCHMARKS[i % 3], "x", "", "") for i in range(n)]
    runs: dict[tuple[str, str], tuple[bool, float]] = {}
    correct: dict[tuple[str, str], bool] = {}
    for i, q in enumerate(queries):
        for j, m in enumerate(MODELS):
            runs[(q.id, m)] = ((i + j) % 6 != 0, 0.25 + ((3 * i + 5 * j) % 13) * 0.5)
            if (i + 2 * j) % 5:
                correct[(q.id, m)] = (i * j) % 3 != 1
    tiers = {
        q.id: (TIER_ORDER[(0, 0, 1, 2)[i % 4]], Stage.KEYWORD if i % 2 == 0 else Stage.DISTILBERT)
        for i, q in enumerate(queries)
    }
    return TraceData(queries=queries, runs=runs, correct=correct), tiers


def fingerprint(result: SimulationResult) -> str:
    """sha256 of the canonical lines of scripts/make_goldens.py, leaving out the totals that use sum().

    The totals (GPU-hours, busy GPU-hours, utilization) are left out because since CPython 3.12 sum()
    of floats is compensated, so they can differ in the last bit on 3.11. Everything else here comes
    from +, -, * and / alone, which give the same bits on every platform. The routing uses Beta draws,
    whose log and exp come from the platform's math library, but a last-bit difference there could only
    change a decision between two scores that tie to the last bit.
    """
    lines = [
        repr(
            (
                r.order,
                r.id,
                r.model,
                str(r.tier),
                str(r.stage),
                r.arrive,
                r.start,
                r.end,
                r.infer_s,
                r.wait_s,
                r.total_s,
                r.cold_start,
                r.success,
                r.correct,
                r.worker,
            )
        )
        for r in result.records
    ]
    usage = result.usage
    lines.append(repr((usage.cold_starts, usage.cold_start_rate, result.makespan_s)))
    for m in MODELS:
        pm = usage.per_model[m]
        lines.append(repr((m, pm.gpu_hours, pm.busy_gpu_hours, pm.loading_gpu_hours, pm.cold_starts)))
    return hashlib.sha256("".join(line + "\n" for line in lines).encode("utf-8")).hexdigest()


# Computed by the v1.1.0 code (simulate.simulate of tag v1.1.0) on varied_trace(), with the same
# fingerprint. Closed loops only: Poisson arrival times come from the platform's log().
V1_1_0_RUNS = [
    pytest.param(
        Policy.PICK_AND_SPIN,
        0,
        SimulationSettings(workers=5, cooldown_s=3.0, max_concurrency=1),
        "2d4c53cf7332623646cc93771d3e755d2c876be09f3145e9d00bcf8c8453f323",
        id="pick-and-spin",
    ),
    pytest.param(
        Policy.PICK_AND_SPIN_OBSERVED,
        1,
        SimulationSettings(workers=8, cooldown_s=6.0),
        "a9d7abb7497e3e9c5e4f704b3f505740b885706e7a72ec20c8b07510bcaf8d9c",
        id="pick-and-spin-observed",
    ),
    pytest.param(
        Policy.UNAWARE,
        2,
        SimulationSettings(workers=3, cooldown_s=2.0, max_concurrency=2),
        "71e7a1e13b37cae191d62faf2b533cf9bfd8ae94553f61db8c1e1afed668301d",
        id="unaware",
    ),
    pytest.param(
        Policy.STATIC,
        3,
        SimulationSettings(workers=4),
        "c3312a277573a6c17777438e993f0c770f0820030f91bc88370fa28d3fea0b6d",
        id="static",
    ),
]


@pytest.mark.parametrize(("policy", "seed", "settings", "digest"), V1_1_0_RUNS)
def test_runs_match_v1_1_0_bit_for_bit(policy: Policy, seed: int, settings: SimulationSettings, digest: str) -> None:
    """Fast, hermetic counterpart of the golden fingerprints in tests/data/golden (full data, @slow)."""
    data, tiers = varied_trace()
    result = simulate(policy, seed, data, tiers, settings)
    assert fingerprint(result) == digest
    assert {r.model for r in result.records} == set(MODELS)
    assert any(not r.success for r in result.records)
    assert any(r.correct is None for r in result.records)
    if POLICIES[policy].scale_to_zero:
        # Models went cold after their short cooldown and loaded again, and queries waited for loads.
        assert result.usage.cold_starts > len(MODELS)
        assert any(r.wait_s > 0 for r in result.records)


# --- what holds in every run ---------------------------------------------------------------------

SCENARIOS = {
    "closed": SimulationSettings(workers=6),
    "closed-capped": SimulationSettings(workers=8, cooldown_s=5.0, max_concurrency=1),
    "poisson": SimulationSettings(arrival_rate=0.2, cooldown_s=10.0),
    "poisson-capped": SimulationSettings(arrival_rate=2.0, cooldown_s=1.0, max_concurrency=2),
}


def max_in_flight(records: Sequence[QueryRecord], model: str) -> int:
    """The most queries executing at once on the model; a query ending at t frees its slot for one starting at t."""
    mine = [r for r in records if r.model == model]
    changes = sorted([(r.start, 1) for r in mine] + [(r.end, -1) for r in mine])
    running = peak = 0
    for _, delta in changes:
        running += delta
        peak = max(peak, running)
    return peak


@pytest.mark.parametrize("policy", list(Policy))
@pytest.mark.parametrize("settings", SCENARIOS.values(), ids=SCENARIOS.keys())
def test_every_run_keeps_the_record_invariants(policy: Policy, settings: SimulationSettings) -> None:
    data, tiers = varied_trace()
    result = simulate(policy, 7, data, tiers, settings)
    records = result.records
    assert [r.id for r in records] == [q.id for q in data.queries]  # records[i] belongs to data.queries[i]
    assert sorted(r.order for r in records) == list(range(1, len(records) + 1))
    for q, r in zip(data.queries, records):
        assert r.wait_s == r.start - r.arrive
        assert r.total_s == r.end - r.arrive
        assert r.end == r.start + r.infer_s
        assert r.arrive <= r.start < r.end
        assert r.benchmark == q.benchmark
        assert (r.tier, r.stage) == tiers[r.id]
        assert MODELS[r.model].tier is r.tier
        assert (r.success, r.infer_s) == data.runs[(r.id, r.model)]
        assert r.correct == data.correct.get((r.id, r.model))
        assert (r.worker is None) == bool(settings.arrival_rate)
    assert result.makespan_s == max(r.end for r in records)
    usage = result.usage
    assert usage.cold_starts == sum(r.cold_start for r in records)
    cold = Counter(r.model for r in records if r.cold_start)
    assert {m: u.cold_starts for m, u in usage.per_model.items()} == {m: cold[m] for m in MODELS}
    assert usage.cold_start_rate == usage.cold_starts / len(records)
    if not POLICIES[policy].scale_to_zero:
        assert usage.cold_starts == 0
    for m in MODELS:
        if settings.max_concurrency is not None:
            assert max_in_flight(records, m) <= settings.max_concurrency
        # The queries of a model start in the order they were routed to it.
        starts = [r.start for r in sorted(records, key=lambda r: r.order) if r.model == m]
        assert starts == sorted(starts)


def test_max_concurrency_one_serves_each_model_first_come_first_served() -> None:
    data, tiers = varied_trace()
    settings = SimulationSettings(workers=12, cooldown_s=30.0, max_concurrency=1)
    result = simulate(Policy.PICK_AND_SPIN, 4, data, tiers, settings)
    waited_for_a_slot = 0
    for m in MODELS:
        routed = sorted((r for r in result.records if r.model == m), key=lambda r: r.order)
        for earlier, later in itertools.pairwise(routed):
            assert later.start >= earlier.end  # never two queries at once, and never overtaken
            waited_for_a_slot += later.arrive < earlier.end == later.start
    assert waited_for_a_slot > 0


def test_a_policy_may_be_given_by_its_old_name(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=12)
    assert simulate("unaware", 2, data, tiers) == simulate(Policy.UNAWARE, 2, data, tiers)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="bogus"):
        simulate("bogus", 0, data, tiers)  # type: ignore[arg-type]


# --- load and random streams ---------------------------------------------------------------------


def test_queries_are_dispatched_in_the_order_of_the_seeded_shuffle(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=30)
    for seed, settings in [(0, SimulationSettings(workers=3)), (5, SimulationSettings(arrival_rate=0.5))]:
        result = simulate(Policy.PICK_AND_SPIN, seed, data, tiers, settings)
        order = list(range(30))
        random.Random(seed).shuffle(order)
        dispatched = sorted(result.records, key=lambda r: r.order)
        assert [r.id for r in dispatched] == [data.queries[i].id for i in order]


def test_poisson_arrivals_accumulate_their_own_random_stream(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=25)
    result = simulate(Policy.UNAWARE, 4, data, tiers, SimulationSettings(workers=3, arrival_rate=0.5))
    rng, t, expected = random.Random(20_000 + 4), 0.0, []
    for _ in range(25):
        t += rng.expovariate(0.5)
        expected.append(t)
    assert [r.arrive for r in sorted(result.records, key=lambda r: r.order)] == expected
    assert all(r.worker is None for r in result.records)


def test_each_random_stream_has_its_own_seed(make_tiny_trace: MakeTinyTrace, monkeypatch: pytest.MonkeyPatch) -> None:
    """The shuffle, Pick's sampler and the arrivals each get a new random.Random, seeded seed + offset."""
    assert (SHUFFLE_SEED_OFFSET, SAMPLER_SEED_OFFSET, ARRIVAL_SEED_OFFSET) == (0, 10_000, 20_000)
    created: list[tuple[random.Random, int | None]] = []

    class Recording(random.Random):
        def __init__(self, x: int | None = None) -> None:
            created.append((self, x))
            super().__init__(x)

    data, tiers = make_tiny_trace(n=10)
    monkeypatch.setattr(random, "Random", Recording)  # the engine calls random.Random(...)
    poisson = simulate(Policy.PICK_AND_SPIN, 3, data, tiers, SimulationSettings(arrival_rate=0.1))
    assert [seed for _, seed in created] == [3, 10_003, 20_003]
    assert len({id(rng) for rng, _ in created}) == 3
    created.clear()
    simulate(Policy.PICK_AND_SPIN, 3, data, tiers, SimulationSettings(workers=2))
    assert [seed for _, seed in created] == [3, 10_003]
    monkeypatch.undo()
    assert simulate(Policy.PICK_AND_SPIN, 3, data, tiers, SimulationSettings(arrival_rate=0.1)) == poisson


def test_closed_loop_workers_send_their_next_query_when_the_last_returns(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=30)
    result = simulate(Policy.PICK_AND_SPIN, 1, data, tiers, SimulationSettings(workers=4))
    dispatched = sorted(result.records, key=lambda r: r.order)
    assert [(r.worker, r.arrive) for r in dispatched[:4]] == [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0)]
    by_worker: defaultdict[int, list[QueryRecord]] = defaultdict(list)
    for r in dispatched:
        assert r.worker is not None
        by_worker[r.worker].append(r)
    assert sorted(by_worker) == [0, 1, 2, 3]
    for sent in by_worker.values():
        for earlier, later in itertools.pairwise(sent):
            assert later.arrive == earlier.end
    # By default 250 workers: with fewer queries than workers, every query is sent at once.
    everyone = simulate(Policy.STATIC, 0, data, tiers).records
    assert {r.worker for r in everyone} == set(range(30))
    assert all(r.arrive == 0.0 for r in everyone)


# --- settings and policies -----------------------------------------------------------------------


def test_load_names() -> None:
    assert SimulationSettings().load_name == "closed-250"
    assert SimulationSettings(workers=8).load_name == "closed-8"
    assert SimulationSettings(arrival_rate=0.25).load_name == "poisson-0.25qps"
    assert SimulationSettings(arrival_rate=1.0).load_name == "poisson-1qps"
    assert SimulationSettings(workers=8, arrival_rate=4).load_name == "poisson-4qps"
    assert SimulationSettings(arrival_rate=0.0).load_name == "closed-250"  # any falsy rate is the closed loop


def test_settings_default_to_the_old_command_line() -> None:
    settings = SimulationSettings()
    assert dataclasses.astuple(settings) == (250, None, None, 300.0)
    names = [f.name for f in dataclasses.fields(settings)]
    assert names == ["workers", "arrival_rate", "max_concurrency", "cooldown_s"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        settings.workers = 8  # type: ignore[misc]
    assert dataclasses.replace(settings, arrival_rate=4.0).load_name == "poisson-4qps"


def test_policies_follow_the_old_order_and_settings() -> None:
    old = {  # POLICIES and the policy list of src/pickspin/simulate.py at v1.1.0
        "pick-and-spin": (True, "spin", "scale to zero; Pick scores Spin's latency, which adds the cold-start "
                          "penalty of a model that is not warm (default design)"),
        "pick-and-spin-observed": (True, "observed",
                                   "scale to zero; Pick scores mean observed latency, cold-start waits included"),
        "unaware": (True, "inference", "scale to zero; Pick scores inference latency only"),
        "static": (False, "spin", "every model warm for the whole run, never scaled down"),
    }  # fmt: skip
    assert list(POLICIES) == list(Policy) == list(old)
    for policy, spec in POLICIES.items():
        assert (spec.scale_to_zero, spec.signal, spec.description) == old[policy]
        assert type(spec.signal) is LatencySignal
        assert POLICIES[policy.value] is spec  # type: ignore[index]  # the old strings still look policies up
    with pytest.raises(TypeError):
        POLICIES[Policy.STATIC] = POLICIES[Policy.UNAWARE]  # type: ignore[index]
    with pytest.raises(dataclasses.FrozenInstanceError):
        POLICIES[Policy.STATIC].scale_to_zero = True  # type: ignore[misc]


def test_simulate_rejects_runs_that_could_not_finish(make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace(n=5)
    with pytest.raises(ValueError, match="no queries"):
        simulate(Policy.STATIC, 0, TraceData(queries=[], runs={}, correct={}), {})
    for settings in (
        SimulationSettings(max_concurrency=0),
        SimulationSettings(workers=0),
        SimulationSettings(arrival_rate=-1.0),
        SimulationSettings(arrival_rate=math.nan),
    ):
        with pytest.raises(ConfigError):
            simulate(Policy.STATIC, 0, data, tiers, settings)
    # Poisson arrivals ignore the number of workers.
    assert len(simulate(Policy.STATIC, 0, data, tiers, SimulationSettings(workers=0, arrival_rate=1.0)).records) == 5


# --- inputs --------------------------------------------------------------------------------------

STATIC_HEADER = ["model", "id", "benchmark", "success", "latency_ms", "prompt_tokens", "completion_tokens", "error"]
JUDGMENTS_HEADER = ["model", "id", "benchmark", "is_correct"]


def write_csv_gz(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def test_load_trace_data_applies_the_simulators_parsing_rules(tmp_path: Path) -> None:
    queries_path = tmp_path / "queries.jsonl.gz"
    with gzip.open(queries_path, "wt", encoding="utf-8") as f:
        for i in range(3):
            record = {"id": f"q{i}", "benchmark": "gsm8k", "query": f"What is {i} + {i}?", "ground_truth": str(2 * i),
                      "query_type": "math"}  # fmt: skip
            f.write(json.dumps(record) + "\n")
    traces = tmp_path / "traces"
    traces.mkdir()

    def static_row(model: str, qid: str, success: str, latency_ms: str) -> list[str]:
        return [model, qid, "gsm8k", success, latency_ms, "12", "34", ""]

    write_csv_gz(
        traces / "static_baseline.csv.gz",
        STATIC_HEADER,
        [
            static_row("llama3.2_1B", "q0", "1", "1598.6"),
            static_row("llama3.2_1B", "q1", "True", ""),  # no latency recorded: 0.0 s
            static_row("llama3.2_1B", "q2", "true", "250"),
            static_row("gemma3_27B", "q0", "0", "1000"),
            static_row("gemma3_27B", "q1", "False", "1000"),
            static_row("gemma3_27B", "q2", "yes", "1000"),  # not one of the true spellings
            static_row("llama3_70B", "q0", "1", "5"),  # not in MODELS: skipped
            static_row("llama3.2_1B", "q2", "0", "300"),  # a repeated pair: the later row wins
        ],
    )
    write_csv_gz(
        traces / "judgments.csv.gz",
        JUDGMENTS_HEADER,
        [
            ["llama3.2_1B", "q0", "gsm8k", "1"],
            ["llama3.2_1B", "q1", "gsm8k", "0"],
            ["gemma3_27B", "q0", "gsm8k", "true"],
            ["gemma3_27B", "q1", "gsm8k", "TRUE"],
            ["kimi_1T_MoE", "q0", "gsm8k", "1"],
            ["gemma3_27B", "q0", "gsm8k", "0"],
        ],
    )
    data = load_trace_data(queries_path, traces)
    assert data.queries == [Query(f"q{i}", "gsm8k", f"What is {i} + {i}?", str(2 * i), "math") for i in range(3)]
    assert data.runs == {
        ("q0", "llama3.2_1B"): (True, 1.5985999999999998),
        ("q1", "llama3.2_1B"): (True, 0.0),
        ("q2", "llama3.2_1B"): (False, 0.3),
        ("q0", "gemma3_27B"): (False, 1.0),
        ("q1", "gemma3_27B"): (False, 1.0),
        ("q2", "gemma3_27B"): (False, 1.0),
    }
    # Milliseconds become seconds as ms / 1000, which differs from ms * 0.001 in the last bit here.
    assert data.runs[("q0", "llama3.2_1B")][1] == 1598.6 / 1000 != 1598.6 * 0.001
    assert data.correct == {
        ("q0", "llama3.2_1B"): True,
        ("q1", "llama3.2_1B"): False,
        ("q0", "gemma3_27B"): False,
        ("q1", "gemma3_27B"): False,
    }


def test_tier_cache_round_trip(tmp_path: Path) -> None:
    queries = [Query(f"q{i}", "b", "x", "", "") for i in range(3)]
    labels = [(Tier.MEDIUM, Stage.KEYWORD), (Tier.COMPLEX, Stage.DISTILBERT), (Tier.SIMPLE, Stage.DEFAULT)]
    cache = tmp_path / "query_tiers.csv.gz"
    write_tier_cache(cache, queries, labels)
    assert gzip.decompress(cache.read_bytes()) == (
        b"id,tier,stage\r\nq0,MEDIUM,keyword\r\nq1,COMPLEX,distilbert\r\nq2,SIMPLE,default\r\n"
    )
    tiers = read_tier_cache(cache)
    assert tiers == {"q0": labels[0], "q1": labels[1], "q2": labels[2]}
    assert all(type(t) is Tier and type(s) is Stage for t, s in tiers.values())
    assert query_tiers(queries, cache) == tiers  # an existing cache needs no model
    bad = write_csv_gz(tmp_path / "bad.csv.gz", ["id", "tier", "stage"], [["q0", "LARGE", "keyword"]])
    with pytest.raises(ValueError, match="LARGE"):
        read_tier_cache(bad)


def test_query_tiers_needs_a_model_directory_to_classify(tmp_path: Path) -> None:
    queries = [Query("q0", "b", "x", "", "")]
    cache = tmp_path / "query_tiers.csv.gz"
    with pytest.raises(ConfigError, match="no tier cache"):
        query_tiers(queries, cache)
    assert not cache.exists()
    write_tier_cache(cache, queries, [(Tier.SIMPLE, Stage.KEYWORD)])
    with pytest.raises(ConfigError, match="reclassifying"):
        query_tiers(queries, cache, reclassify=True)


class StubTierPredictor:
    """Stands in for DistilBERT: every query it sees is COMPLEX. Records each predict() call."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def predict(self, queries: Sequence[str]) -> list[str]:
        self.calls.append(list(queries))
        return ["COMPLEX"] * len(queries)


def test_query_tiers_classifies_and_rewrites_the_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    stub = StubTierPredictor()
    loaded_from: list[Path] = []

    def from_pretrained(cls: type[HybridClassifier], /, model_dir: Path, **kwargs: object) -> HybridClassifier:
        loaded_from.append(model_dir)
        return cls(stub)

    monkeypatch.setattr(HybridClassifier, "from_pretrained", classmethod(from_pretrained))
    monkeypatch.setattr(logging.getLogger("pickspin"), "propagate", True)
    texts = ["What is 2 + 2?", "Finish the story", "Prove that x > 0", "Tell me more", *["Define x"] * 997]
    queries = [Query(f"q{i}", "b", text, "", "") for i, text in enumerate(texts)]
    cache = tmp_path / "query_tiers.csv.gz"
    write_tier_cache(cache, queries, [(Tier.SIMPLE, Stage.DEFAULT)] * len(queries))  # an outdated cache
    with caplog.at_level(logging.INFO, logger="pickspin.simulation.inputs"):
        tiers = query_tiers(queries, cache, reclassify=True, model_dir=tmp_path / "model")
    assert caplog.messages == ["Classifying 1,001 queries (keyword lists, then DistilBERT) ..."]
    assert loaded_from == [tmp_path / "model"]
    assert stub.calls == [["Finish the story", "Tell me more"]]  # one batch of the unmatched queries
    assert list(tiers) == [q.id for q in queries]
    assert tiers["q0"] == (Tier.SIMPLE, Stage.KEYWORD)
    assert tiers["q1"] == tiers["q3"] == (Tier.COMPLEX, Stage.DISTILBERT)
    assert tiers["q2"] == (Tier.COMPLEX, Stage.KEYWORD)
    assert read_tier_cache(cache) == tiers


def test_describe_stages_counts_the_stages_most_common_first() -> None:
    tiers: dict[str, TierAssignment] = {f"k{i}": (Tier.SIMPLE, Stage.KEYWORD) for i in range(1500)}
    tiers |= {f"d{i}": (Tier.MEDIUM, Stage.DISTILBERT) for i in range(400)}
    tiers |= {f"f{i}": (Tier.MEDIUM, Stage.DEFAULT) for i in range(100)}
    assert describe_stages(2000, tiers) == (
        "2,000 queries; classifier stages: keyword 1,500 (75.0%), distilbert 400 (20.0%), default 100 (5.0%)"
    )
    # Stages with equal counts keep the order in which they first appear.
    tied = {
        "a": (Tier.SIMPLE, Stage.DISTILBERT),
        "b": (Tier.SIMPLE, Stage.KEYWORD),
        "c": (Tier.COMPLEX, Stage.KEYWORD),
        "d": (Tier.MEDIUM, Stage.DISTILBERT),
    }
    assert describe_stages(4, tied) == "4 queries; classifier stages: distilbert 2 (50.0%), keyword 2 (50.0%)"
