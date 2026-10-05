"""The CSV rows of a simulated run (pickspin.simulation.metrics), on hand-built records.

The columns, their order and the formulas are those of the v1.1.0 simulator. Several expected values
are exact floats that pin the shape of an expression: 100 * 1 / 3 is 33.333333333333336 while
1 / 3 * 100 is 33.33333333333333, and statistics.fmean([9.0, 0.3, 0.3]) is 3.1999999999999997 while
statistics.mean gives 3.2. Rewriting a formula changes the rounded CSVs.
"""

import statistics
from collections.abc import Mapping

from pickspin.config import MODELS, TIER_ORDER, Tier
from pickspin.pick.classifier import Stage
from pickspin.simulation.engine import QueryRecord, SimulationResult
from pickspin.simulation.metrics import (
    OVERVIEW_METRICS,
    cold_by_tier_rows,
    overview_rows,
    per_model_rows,
    percentile,
    summary_row,
)
from pickspin.simulation.policies import Policy
from pickspin.spin.lifecycle import ModelUsage, SpinSummary

SUMMARY_COLUMNS = [
    "load", "policy", "seed", "queries", "makespan_h", "gpu_hours", "busy_gpu_hours", "gpu_utilization_pct",
    "cold_starts", "cold_start_rate_pct", "queries_waiting_for_load_pct", "latency_mean_s", "latency_median_s",
    "latency_p95_s", "execution_success_pct", "accuracy_pct", "scored_queries", "share_simple_pct",
    "share_medium_pct", "share_complex_pct",
]  # fmt: skip
PER_MODEL_COLUMNS = [
    "load", "policy", "seed", "model", "tier", "queries", "share_pct", "accuracy_pct", "latency_mean_s",
    "cold_starts", "gpu_hours", "busy_gpu_hours",
]  # fmt: skip
COLD_BY_TIER_COLUMNS = ["load", "policy", "seed", "tier", "routed_queries", "cold_starts", "conditional_rate_pct"]
OVERVIEW_COLUMNS = [
    "load", "policy", "seeds", "makespan_h", "gpu_hours", "gpu_hours_sd", "gpu_hours_vs_static_pct",
    "gpu_utilization_pct", "cold_starts", "cold_start_rate_pct", "queries_waiting_for_load_pct", "latency_mean_s",
    "latency_p95_s", "accuracy_pct",
]  # fmt: skip


def make_record(
    order: int,
    model: str,
    *,
    tier: Tier = Tier.SIMPLE,
    total_s: float = 1.0,
    wait_s: float = 0.0,
    success: bool = True,
    correct: bool | None = True,
    cold_start: bool = False,
) -> QueryRecord:
    """A finished query that arrived at 0 s, waited wait_s for its model and took total_s end to end."""
    return QueryRecord(
        order=order,
        id=f"q{order}",
        benchmark="b",
        tier=tier,
        stage=Stage.KEYWORD,
        model=model,
        arrive=0.0,
        infer_s=total_s - wait_s,
        success=success,
        correct=correct,
        cold_start=cold_start,
        worker=0,
        start=wait_s,
        end=total_s,
        wait_s=wait_s,
        total_s=total_s,
    )


# Per-model accounting with values distinct for every model: model i (in MODELS order) holds
# (i + 1) / 8 GPU-hours, busy for (i + 1) / 16, after i cold starts.
PER_MODEL_USAGE = {
    m: ModelUsage(gpu_hours=(i + 1) / 8, busy_gpu_hours=(i + 1) / 16, loading_gpu_hours=0.0, cold_starts=i)
    for i, m in enumerate(MODELS)
}


def make_result(
    records: list[QueryRecord], *, makespan_s: float = 3600.0, per_model: Mapping[str, ModelUsage] = PER_MODEL_USAGE
) -> SimulationResult:
    usage = SpinSummary(
        gpu_hours=2.5,
        busy_gpu_hours=0.5,
        gpu_utilization=0.2,
        cold_starts=4,
        cold_start_rate=0.25,
        per_model=dict(per_model),
    )
    return SimulationResult(records=records, usage=usage, makespan_s=makespan_s)


# --- percentile ------------------------------------------------------------------------------------


def test_percentile_is_the_nearest_rank_without_interpolation() -> None:
    assert percentile([3.0, 1.0, 2.0], 0.5) == 2.0
    assert percentile([4.0, 1.0, 3.0, 2.0], 0.5) == 3.0  # the upper median, not 2.5
    assert percentile([5.0], 0.95) == 5.0
    assert percentile([2.0, 1.0], 0.0) == 1.0
    assert percentile([2.0, 1.0], 1.0) == 2.0  # the index is capped at n - 1


def test_percentile_index_is_int_of_p_times_n() -> None:
    values = [float(v) for v in range(1, 101)]
    assert percentile(values, 0.95) == 96.0  # sorted(values)[int(0.95 * 100)] = sorted(values)[95]
    assert percentile(reversed(values), 0.95) == 96.0  # any iterable, in any order
    assert percentile((float(v) for v in range(1, 21)), 0.95) == 20.0  # int(0.95 * 20) = 19, the largest
    assert percentile([float(v) for v in range(1, 22)], 0.95) == 20.0  # int(19.95) = 19


# --- summary_row -----------------------------------------------------------------------------------


def test_summary_row_formulas() -> None:
    records = [
        make_record(1, "llama3.2_1B", total_s=9.0, wait_s=1.0, correct=True),
        make_record(2, "qwen2.5_7B", tier=Tier.MEDIUM, total_s=0.3, correct=False),
        # A SIMPLE query on a COMPLEX model: the shares count the classifier's tier, not the model's.
        make_record(3, "gemma3_27B", total_s=0.3, success=False, correct=None),
    ]
    row = summary_row("closed-4", Policy.UNAWARE, 7, make_result(records, makespan_s=5400.0))
    assert list(row) == SUMMARY_COLUMNS
    assert row == {
        "load": "closed-4",
        "policy": Policy.UNAWARE,
        "seed": 7,
        "queries": 3,
        "makespan_h": 1.5,
        "gpu_hours": 2.5,
        "busy_gpu_hours": 0.5,
        "gpu_utilization_pct": 20.0,
        "cold_starts": 4,
        "cold_start_rate_pct": 25.0,
        "queries_waiting_for_load_pct": 33.333333333333336,  # 100 * 1 / 3
        "latency_mean_s": 3.1999999999999997,  # statistics.fmean
        "latency_median_s": 0.3,
        "latency_p95_s": 9.0,
        "execution_success_pct": 66.66666666666667,  # 100 * 2 / 3
        "accuracy_pct": 50.0,  # 1 of the 2 scored queries; the unscored one is left out
        "scored_queries": 2,
        "share_simple_pct": 66.66666666666667,
        "share_medium_pct": 33.333333333333336,
        "share_complex_pct": 0.0,
    }
    assert statistics.mean([9.0, 0.3, 0.3]) == 3.2  # which is why the mean must stay fmean


def test_summary_row_latency_median_and_p95_of_an_even_number_of_queries() -> None:
    records = [make_record(i, "llama3.2_1B", total_s=t) for i, t in enumerate([8.0, 1.0, 4.0, 2.0], start=1)]
    row = summary_row("closed-4", Policy.PICK_AND_SPIN, 0, make_result(records))
    assert row["latency_median_s"] == 3.0  # statistics.median averages the two middle values
    assert row["latency_p95_s"] == 8.0  # the nearest rank: sorted(total)[int(0.95 * 4)] = sorted(total)[3]
    assert percentile([8.0, 1.0, 4.0, 2.0], 0.5) == 4.0  # which is not the median


def test_summary_row_has_twenty_columns_ending_with_the_tier_shares() -> None:
    row = summary_row("poisson-0.25qps", Policy.STATIC, 0, make_result([make_record(1, "gemma2_2B")]))
    assert len(row) == 20
    assert list(row)[-3:] == [f"share_{t.lower()}_pct" for t in TIER_ORDER]
    assert [row[c] for c in ("share_simple_pct", "share_medium_pct", "share_complex_pct")] == [100.0, 0.0, 0.0]


def test_summary_row_shares_and_rates_are_exact_percentages() -> None:
    records = [make_record(i, "llama3.2_1B", tier=TIER_ORDER[i % 3], wait_s=float(i % 2)) for i in range(1, 12)]
    row = summary_row("closed-250", Policy.PICK_AND_SPIN, 0, make_result(records))
    # 11 queries: four MEDIUM (i = 1, 4, 7, 10), four COMPLEX (2, 5, 8, 11), three SIMPLE (3, 6, 9).
    assert row["share_simple_pct"] == 100 * 3 / 11 == 27.272727272727273
    assert row["share_medium_pct"] == row["share_complex_pct"] == 100 * 4 / 11 == 36.36363636363637
    assert row["queries_waiting_for_load_pct"] == 100 * 6 / 11 == 54.54545454545455
    assert 6 / 11 * 100 != 54.54545454545455  # the other association rounds differently


# --- per_model_rows --------------------------------------------------------------------------------


def test_per_model_rows_one_row_per_model_in_catalog_order() -> None:
    records = [
        make_record(1, "qwen2.5_7B", tier=Tier.MEDIUM, total_s=2.0, correct=True),
        make_record(2, "qwen2.5_7B", tier=Tier.MEDIUM, total_s=4.0, correct=None),
        make_record(3, "gemma3_27B", tier=Tier.COMPLEX, total_s=9.0, correct=None),
    ]
    rows = per_model_rows("poisson-4qps", Policy.PICK_AND_SPIN_OBSERVED, 3, make_result(records))
    assert len(rows) == 9
    assert [list(r) for r in rows] == [PER_MODEL_COLUMNS] * 9
    assert [r["model"] for r in rows] == [spec.label for spec in MODELS.values()]
    assert [r["tier"] for r in rows] == [spec.tier for spec in MODELS.values()]
    by_key = dict(zip(MODELS, rows))
    assert by_key["qwen2.5_7B"] == {
        "load": "poisson-4qps",
        "policy": Policy.PICK_AND_SPIN_OBSERVED,
        "seed": 3,
        "model": "Qwen2.5-7B",
        "tier": Tier.MEDIUM,
        "queries": 2,
        "share_pct": 66.66666666666667,  # 100 * 2 / 3
        "accuracy_pct": 100.0,  # of the one scored query
        "latency_mean_s": 3.0,  # end to end, statistics.fmean
        "cold_starts": 4,
        "gpu_hours": 5 / 8,
        "busy_gpu_hours": 5 / 16,
    }
    # Served, but no judge label: no accuracy.
    assert by_key["gemma3_27B"]["accuracy_pct"] == ""
    assert by_key["gemma3_27B"]["latency_mean_s"] == 9.0


def test_per_model_rows_leave_accuracy_and_latency_empty_for_unused_models() -> None:
    rows = per_model_rows("closed-4", Policy.STATIC, 0, make_result([make_record(1, "gemma2_9B", tier=Tier.MEDIUM)]))
    for m, row in zip(MODELS, rows):
        served = (row["queries"], row["share_pct"], row["accuracy_pct"], row["latency_mean_s"])
        assert served == ((1, 100.0, 100.0, 1.0) if m == "gemma2_9B" else (0, 0.0, "", ""))
        # The GPU accounting comes from Spin's summary whether or not the model served a query.
        usage = PER_MODEL_USAGE[m]
        assert (row["cold_starts"], row["gpu_hours"], row["busy_gpu_hours"]) == (
            usage.cold_starts,
            usage.gpu_hours,
            usage.busy_gpu_hours,
        )


# --- cold_by_tier_rows -----------------------------------------------------------------------------


def test_cold_by_tier_rows_count_the_tier_of_the_routed_model() -> None:
    records = [
        make_record(1, "llama3.2_1B", cold_start=True),
        make_record(2, "llama3.2_3B"),
        make_record(3, "gemma2_2B"),
        make_record(4, "qwen2.5_7B", tier=Tier.SIMPLE, cold_start=True),  # counted under MEDIUM, its model's tier
    ]
    rows = cold_by_tier_rows("closed-250", Policy.PICK_AND_SPIN, 2, records)
    assert [list(r) for r in rows] == [COLD_BY_TIER_COLUMNS] * 3
    assert all((r["load"], r["policy"], r["seed"]) == ("closed-250", Policy.PICK_AND_SPIN, 2) for r in rows)
    assert [(r["tier"], r["routed_queries"], r["cold_starts"], r["conditional_rate_pct"]) for r in rows] == [
        (Tier.SIMPLE, 3, 1, 33.333333333333336),  # 100 * 1 / 3
        (Tier.MEDIUM, 1, 1, 100.0),
        (Tier.COMPLEX, 0, 0, 0.0),
    ]
    assert [r["tier"] for r in rows] == list(TIER_ORDER)
    assert type(rows[2]["conditional_rate_pct"]) is float  # an empty tier is written as 0.0, not 0
    assert type(rows[0]["cold_starts"]) is int


# --- overview_rows ---------------------------------------------------------------------------------


def summary_strings(load: str, policy: str, seed: int, **values: str) -> dict[str, str]:
    """A summary.csv row as csv.DictReader returns it: every value a string, every column present."""
    row = dict.fromkeys(SUMMARY_COLUMNS, "1.0") | {"load": load, "policy": policy, "seed": str(seed)}
    return row | values


def test_overview_rows_average_each_load_and_policy_over_seeds() -> None:
    rows = overview_rows(
        [
            summary_strings("poisson-1qps", "unaware", 0, gpu_hours="0.5"),
            summary_strings("closed-4", "static", 0, gpu_hours="2.0"),
            summary_strings("closed-4", "pick-and-spin", 0, gpu_hours="1.0", latency_mean_s="9.0", cold_starts="12"),
            summary_strings("closed-4", "pick-and-spin", 1, gpu_hours="2.0", latency_mean_s="0.3", cold_starts="13"),
            summary_strings("closed-4", "static", 1, gpu_hours="4.0"),
            summary_strings("closed-4", "pick-and-spin", 2, gpu_hours="3.0", latency_mean_s="0.3", cold_starts="15"),
        ]
    )
    # Sorted by (load, policy), whatever the input order.
    assert [(r["load"], r["policy"]) for r in rows] == [
        ("closed-4", "pick-and-spin"),
        ("closed-4", "static"),
        ("poisson-1qps", "unaware"),
    ]
    # gpu_hours_sd and gpu_hours_vs_static_pct come right after gpu_hours.
    assert [list(r) for r in rows] == [OVERVIEW_COLUMNS] * 3
    assert OVERVIEW_COLUMNS[3:5] + OVERVIEW_COLUMNS[7:] == list(OVERVIEW_METRICS)
    pick_and_spin, static, _ = rows
    assert pick_and_spin["seeds"] == 3
    assert pick_and_spin["gpu_hours"] == 2.0
    assert pick_and_spin["gpu_hours_sd"] == 1.0  # statistics.stdev, the sample standard deviation
    assert pick_and_spin["gpu_hours_vs_static_pct"] == -33.333333333333336  # 100 * (2.0 / 3.0 - 1)
    assert pick_and_spin["latency_mean_s"] == 3.1999999999999997  # statistics.fmean of the three seeds
    assert pick_and_spin["cold_starts"] == 13.333333333333334
    assert pick_and_spin["makespan_h"] == 1.0
    assert static["seeds"] == 2
    assert static["gpu_hours"] == 3.0
    assert static["gpu_hours_sd"] == 1.4142135623730951 == statistics.stdev([2.0, 4.0])
    assert static["gpu_hours_vs_static_pct"] == 0.0


def test_overview_rows_single_seed_and_no_static_run() -> None:
    (row,) = overview_rows([summary_strings("poisson-1qps", "unaware", 0, gpu_hours="0.5")])
    assert row["seeds"] == 1
    assert row["gpu_hours"] == 0.5
    assert row["gpu_hours_sd"] == 0.0  # statistics.stdev needs two seeds
    assert row["gpu_hours_vs_static_pct"] == ""  # no static run under this load
    assert all(type(row[c]) is float for c in OVERVIEW_METRICS)


def test_overview_rows_compare_with_the_static_run_of_the_same_load_only() -> None:
    rows = overview_rows(
        [
            summary_strings("closed-4", "static", 0, gpu_hours="4.0"),
            summary_strings("closed-8", "unaware", 0, gpu_hours="1.0"),
            summary_strings("closed-4", "unaware", 0, gpu_hours="1.0"),
        ]
    )
    assert [(r["load"], r["policy"], r["gpu_hours_vs_static_pct"]) for r in rows] == [
        ("closed-4", "static", 0.0),
        ("closed-4", "unaware", -75.0),
        ("closed-8", "unaware", ""),
    ]
