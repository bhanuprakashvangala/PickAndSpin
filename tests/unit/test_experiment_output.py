"""The simulator's output files (pickspin.simulation.experiment): exact bytes, and the policy x seed grid.

write_csv and write_query_trace write what the v1.1.0 simulator wrote: UTF-8 in the csv module's default
dialect (CRLF line ends, minimal quoting), floats rounded with round() and written in their shortest
form, and enum members as their plain values. The full-size equivalence with v1.1.0 is checked against
the golden digests by tests/integration/test_simulation_golden.py.
"""

import csv
import gzip
import logging
from collections.abc import Callable
from dataclasses import fields, replace
from pathlib import Path

import pytest

from pickspin.config import MODELS, TIER_ORDER, Tier
from pickspin.pick.classifier import Stage
from pickspin.simulation import experiment
from pickspin.simulation.engine import QueryRecord, SimulationResult, SimulationSettings, simulate
from pickspin.simulation.experiment import (
    QUERY_TRACE_COLUMNS,
    run_load,
    write_csv,
    write_overview,
    write_query_trace,
)
from pickspin.simulation.inputs import TierAssignment, TraceData
from pickspin.simulation.metrics import OVERVIEW_METRICS
from pickspin.simulation.policies import Policy
from pickspin.spin.lifecycle import ModelUsage, SpinSummary

MakeTinyTrace = Callable[..., tuple[TraceData, dict[str, TierAssignment]]]

OUTPUT_FILES = ["cold_starts_by_tier.csv", "per_model.csv", "summary.csv"]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def progress(caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch) -> pytest.LogCaptureFixture:
    """caplog, recording pickspin.simulation.experiment at INFO even after the CLI has configured logging."""
    monkeypatch.setattr(logging.getLogger("pickspin"), "propagate", True)  # configure_logging turns it off
    caplog.set_level(logging.INFO, logger=experiment.log.name)
    return caplog


def messages(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == experiment.log.name]


# --- write_csv -------------------------------------------------------------------------------------


def test_write_csv_bytes(tmp_path: Path) -> None:
    path = tmp_path / "closed-4" / "nested" / "rows.csv"  # missing parents are created
    first = {"policy": Policy.STATIC, "tier": Tier.SIMPLE, "half": 0.5, "third": 1 / 3, "flag": True, "n": 3, "e": ""}
    # Values are matched to the header by key, whatever the order of a later row's keys.
    second = {"e": "a,b", "n": 0, "flag": False, "third": 2 / 3, "half": 123.456789, "tier": "MEDIUM", "policy": "x"}
    third = {"policy": "y", "tier": "COMPLEX", "half": 2.0, "third": 1e-05, "flag": True, "n": 10, "e": ""}
    write_csv(path, [first, second, third])
    assert path.read_bytes() == (
        b"policy,tier,half,third,flag,n,e\r\n"
        b"static,SIMPLE,0.5,0.3333,True,3,\r\n"
        b'x,MEDIUM,123.4568,0.6667,False,0,"a,b"\r\n'
        b"y,COMPLEX,2.0,0.0,True,10,\r\n"
    )


def test_write_csv_needs_at_least_one_row(tmp_path: Path) -> None:
    path = tmp_path / "closed-4" / "summary.csv"
    with pytest.raises(ValueError, match="no rows"):
        write_csv(path, [])
    assert not path.parent.exists()


# --- write_query_trace -----------------------------------------------------------------------------


def test_query_trace_columns_are_the_record_fields_without_the_worker() -> None:
    assert QUERY_TRACE_COLUMNS == (
        "order", "id", "benchmark", "tier", "stage", "model", "arrive", "start", "end", "infer_s", "wait_s",
        "total_s", "cold_start", "success", "correct",
    )  # fmt: skip
    assert set(QUERY_TRACE_COLUMNS) == {f.name for f in fields(QueryRecord)} - {"worker"}


def test_write_query_trace_bytes(tmp_path: Path) -> None:
    records = [
        QueryRecord(
            order=2, id="q0", benchmark="math", tier=Tier.COMPLEX, stage=Stage.DISTILBERT, model="gemma3_27B",
            arrive=0.25, infer_s=2 / 3, success=False, correct=None, cold_start=True, worker=None,
            start=95.2504, end=95.2504 + 2 / 3, wait_s=95.0004, total_s=95.2504 + 2 / 3 - 0.25,
        ),
        QueryRecord(
            order=1, id="q1", benchmark="gsm8k", tier=Tier.SIMPLE, stage=Stage.KEYWORD, model="llama3.2_1B",
            arrive=0.0, infer_s=1 / 3, success=True, correct=True, cold_start=False, worker=3,
            start=0.0, end=1 / 3, wait_s=0.0, total_s=1 / 3,
        ),
    ]  # fmt: skip
    path = tmp_path / "poisson-0.25qps" / "queries_pick-and-spin.csv.gz"  # missing parents are created
    write_query_trace(path, records)
    raw = path.read_bytes()
    assert raw[:2] == b"\x1f\x8b"  # gzip
    # Sorted by order; floats rounded to 3 places; no label -> empty 'correct'; no worker column.
    assert gzip.decompress(raw) == (
        b"order,id,benchmark,tier,stage,model,arrive,start,end,infer_s,wait_s,total_s,cold_start,success,correct\r\n"
        b"1,q1,gsm8k,SIMPLE,keyword,llama3.2_1B,0.0,0.0,0.333,0.333,0.0,0.333,False,True,True\r\n"
        b"2,q0,math,COMPLEX,distilbert,gemma3_27B,0.25,95.25,95.917,0.667,95.0,95.667,True,False,\r\n"
    )


# --- run_load with a stand-in engine: the grid, the files and the progress lines --------------------


def stand_in_result(seed: int, n: int) -> SimulationResult:
    """A run of n SIMPLE queries on llama3.2_1B whose numbers depend on the seed; even queries are correct."""
    records = [
        QueryRecord(
            order=n - i, id=f"q{i}", benchmark="b", tier=Tier.SIMPLE, stage=Stage.KEYWORD, model="llama3.2_1B",
            arrive=float(i), infer_s=1.0, success=True, correct=i % 2 == 0, cold_start=i == 0, worker=i,
            start=float(i + seed), end=float(i + seed) + 1.0, wait_s=float(seed), total_s=seed + 1.0,
        )
        for i in range(n)
    ]  # fmt: skip
    usage = SpinSummary(
        gpu_hours=seed + 0.25,
        busy_gpu_hours=0.5,
        gpu_utilization=0.4,
        cold_starts=1,
        cold_start_rate=1 / n,
        per_model={m: ModelUsage(0.0, 0.0, 0.0, 0) for m in MODELS},
    )
    return SimulationResult(records=records, usage=usage, makespan_s=3600.0 * (seed + 1))


def test_run_load_runs_policies_then_seeds_and_traces_the_first_seed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, progress: pytest.LogCaptureFixture, make_tiny_trace: MakeTinyTrace
) -> None:
    data, tiers = make_tiny_trace(n=2)
    settings = SimulationSettings(workers=4)
    calls = []

    def stand_in(
        policy: Policy, seed: int, data_: TraceData, tiers_: dict[str, TierAssignment], settings_: SimulationSettings
    ) -> SimulationResult:
        assert data_ is data
        assert tiers_ is tiers
        assert settings_ is settings
        calls.append((policy, seed))
        return stand_in_result(seed, len(data_.queries))

    monkeypatch.setattr(experiment, "simulate", stand_in)
    out = run_load([Policy.STATIC, Policy.UNAWARE], [5, 2], settings, data, tiers, tmp_path, write_queries=True)

    assert out == tmp_path / "closed-4"
    assert calls == [(Policy.STATIC, 5), (Policy.STATIC, 2), (Policy.UNAWARE, 5), (Policy.UNAWARE, 2)]
    assert sorted(p.name for p in out.iterdir()) == sorted(
        [*OUTPUT_FILES, "queries_static.csv.gz", "queries_unaware.csv.gz"]
    )
    summary = read_rows(out / "summary.csv")
    assert [(r["load"], r["policy"], r["seed"], r["gpu_hours"]) for r in summary] == [
        ("closed-4", "static", "5", "5.25"),
        ("closed-4", "static", "2", "2.25"),
        ("closed-4", "unaware", "5", "5.25"),
        ("closed-4", "unaware", "2", "2.25"),
    ]
    per_model = read_rows(out / "per_model.csv")
    assert [(r["policy"], r["seed"], r["model"]) for r in per_model] == [
        (p, s, spec.label) for p in ("static", "unaware") for s in ("5", "2") for spec in MODELS.values()
    ]
    cold = read_rows(out / "cold_starts_by_tier.csv")
    assert [(r["policy"], r["seed"], r["tier"]) for r in cold] == [
        (p, s, t) for p in ("static", "unaware") for s in ("5", "2") for t in TIER_ORDER
    ]
    # The trace is the run with seeds[0] = 5: every query waited 5 s.
    with gzip.open(out / "queries_unaware.csv.gz", "rt", encoding="utf-8", newline="") as f:
        trace = list(csv.DictReader(f))
    assert [(r["order"], r["id"], r["wait_s"]) for r in trace] == [("1", "q1", "5.0"), ("2", "q0", "5.0")]

    def line(policy: str, seed: int, gpu_hours: str, latency: str) -> str:
        return (
            f"closed-4 {policy.ljust(24)} seed {seed}: {gpu_hours} GPU-h, util 40.0%, cold starts 1 (50.00%), "
            f"latency {latency}s, accuracy 50.0%"
        )

    assert messages(progress) == [
        line("static", 5, "5.25", "6.00"),
        line("static", 2, "2.25", "3.00"),
        line("unaware", 5, "5.25", "6.00"),
        line("unaware", 2, "2.25", "3.00"),
        f"Wrote {out}",
    ]


# --- run_load on the tiny trace, through the engine --------------------------------------------------


def test_run_load_on_the_tiny_trace(
    tmp_path: Path, progress: pytest.LogCaptureFixture, make_tiny_trace: MakeTinyTrace
) -> None:
    data, tiers = make_tiny_trace()
    settings = SimulationSettings(workers=4)
    seeds = [1, 0]
    out = run_load(list(Policy), seeds, settings, data, tiers, tmp_path, write_queries=True)

    assert out == tmp_path / "closed-4"
    assert sorted(p.name for p in out.iterdir()) == sorted([*OUTPUT_FILES, *(f"queries_{p}.csv.gz" for p in Policy)])
    summary = read_rows(out / "summary.csv")
    assert [(r["load"], r["policy"], r["seed"]) for r in summary] == [
        ("closed-4", p.value, str(s)) for p in Policy for s in seeds
    ]
    for row in summary:
        assert (row["queries"], row["scored_queries"], row["execution_success_pct"]) == ("40", "40", "100.0")
        assert (row["accuracy_pct"], row["share_simple_pct"], row["share_complex_pct"]) == ("50.0", "100.0", "0.0")
    assert all(r["cold_starts"] == "0" for r in summary if r["policy"] == "static")
    assert all(int(r["cold_starts"]) >= 1 for r in summary if r["policy"] != "static")
    per_model = read_rows(out / "per_model.csv")
    assert [r["model"] for r in per_model] == [spec.label for spec in MODELS.values()] * 8
    cold = read_rows(out / "cold_starts_by_tier.csv")
    assert [r["tier"] for r in cold] == list(TIER_ORDER) * 8
    assert all(r["routed_queries"] == ("40" if r["tier"] == Tier.SIMPLE else "0") for r in cold)

    # Each trace is the run with the first seed listed (1), which differs from the seed-0 run.
    for policy in Policy:
        first, other = tmp_path / "expected" / "first.csv.gz", tmp_path / "expected" / "other.csv.gz"
        write_query_trace(first, simulate(policy, seeds[0], data, tiers, settings).records)
        write_query_trace(other, simulate(policy, seeds[1], data, tiers, settings).records)
        written = gzip.decompress((out / f"queries_{policy}.csv.gz").read_bytes())
        assert written == gzip.decompress(first.read_bytes())
        assert written != gzip.decompress(other.read_bytes())

    lines = messages(progress)
    assert len(lines) == 2 * len(Policy) + 1
    assert lines[0].startswith(f"closed-4 {'pick-and-spin'.ljust(24)} seed 1: ")
    assert lines[-1] == f"Wrote {out}"


def test_run_load_and_write_overview_over_two_loads(tmp_path: Path, make_tiny_trace: MakeTinyTrace) -> None:
    data, tiers = make_tiny_trace()
    closed = SimulationSettings(workers=4)
    run_load(list(Policy), [0, 1], closed, data, tiers, tmp_path)
    poisson = replace(closed, arrival_rate=0.5)
    out = run_load([Policy.STATIC, Policy.PICK_AND_SPIN], [0], poisson, data, tiers, tmp_path)
    assert out == tmp_path / "poisson-0.5qps"
    assert sorted(p.name for p in out.iterdir()) == OUTPUT_FILES  # no traces without write_queries

    overview = write_overview(tmp_path)
    assert overview == tmp_path / "overview.csv"
    rows = read_rows(overview)
    assert [(r["load"], r["policy"], r["seeds"]) for r in rows] == [
        ("closed-4", "pick-and-spin", "2"),
        ("closed-4", "pick-and-spin-observed", "2"),
        ("closed-4", "static", "2"),
        ("closed-4", "unaware", "2"),
        ("poisson-0.5qps", "pick-and-spin", "1"),
        ("poisson-0.5qps", "static", "1"),
    ]
    assert [r["gpu_hours_vs_static_pct"] for r in rows if r["policy"] == "static"] == ["0.0", "0.0"]


# --- write_overview --------------------------------------------------------------------------------


def summary_rows(load: str, policy: Policy, gpu_hours: list[float]) -> list[dict[str, object]]:
    """summary.csv rows of one policy, one per seed; every averaged metric but gpu_hours is 1.0."""
    return [
        {"load": load, "policy": policy, "seed": seed, **dict.fromkeys(OVERVIEW_METRICS, 1.0), "gpu_hours": g}
        for seed, g in enumerate(gpu_hours)
    ]


def test_write_overview_averages_the_rounded_summaries_of_every_load(tmp_path: Path) -> None:
    poisson = summary_rows("poisson-0.5qps", Policy.UNAWARE, [6e-05, 6e-05, 0.0])
    closed = summary_rows("closed-4", Policy.STATIC, [2.0, 4.0]) + summary_rows(
        "closed-4", Policy.PICK_AND_SPIN, [1 / 3, 1 / 3]
    )
    write_csv(tmp_path / "poisson-0.5qps" / "summary.csv", poisson)
    write_csv(tmp_path / "closed-4" / "summary.csv", closed)
    # Only <root>/*/summary.csv counts.
    write_csv(tmp_path / "closed-4" / "old" / "summary.csv", summary_rows("closed-4", Policy.STATIC, [99.0]))

    path = write_overview(tmp_path)
    assert path == tmp_path / "overview.csv"
    expected = (
        b"load,policy,seeds,makespan_h,gpu_hours,gpu_hours_sd,gpu_hours_vs_static_pct,gpu_utilization_pct,"
        b"cold_starts,cold_start_rate_pct,queries_waiting_for_load_pct,latency_mean_s,latency_p95_s,accuracy_pct\r\n"
        # 1/3 was written as 0.3333: 100 * (0.3333 / 3.0 - 1) = -88.89.
        b"closed-4,pick-and-spin,2,1.0,0.3333,0.0,-88.89,1.0,1.0,1.0,1.0,1.0,1.0,1.0\r\n"
        b"closed-4,static,2,1.0,3.0,1.4142,0.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0\r\n"
        # The summary holds 0.0001, 0.0001 and 0.0, whose mean rounds to 0.0001; averaging the unrounded
        # 6e-05, 6e-05 and 0.0 would round to 0.0.
        b"poisson-0.5qps,unaware,3,1.0,0.0001,0.0001,,1.0,1.0,1.0,1.0,1.0,1.0,1.0\r\n"
    )
    assert path.read_bytes() == expected
    write_overview(tmp_path)  # overview.csv itself is never read back as a summary
    assert path.read_bytes() == expected
