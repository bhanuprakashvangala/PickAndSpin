"""Turn simulated runs into CSV rows.

Each row is an ordered dict whose key order is the CSV header. summary_row describes one run,
per_model_rows and cold_by_tier_rows break it down by model and by tier, and overview_rows averages
the rounded summary rows read back from disk over the seeds of every (load, policy).

The columns and formulas are those of the v1.1.0 simulator, and the arithmetic is kept expression
for expression (100 * x / n, never x / n * 100), because the rounded CSVs must not change. Means
use statistics.fmean, medians statistics.median, the overview's spread statistics.stdev, and the
95th percentile the nearest-rank rule of percentile().
"""

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Final

from pickspin.config import MODELS, TIER_ORDER
from pickspin.simulation.engine import QueryRecord, SimulationResult
from pickspin.simulation.policies import Policy

# The summary columns that overview.csv averages over seeds, in column order. gpu_hours is followed
# by its sample standard deviation and by its difference from the static policy under the same load.
OVERVIEW_METRICS: Final[tuple[str, ...]] = (
    "makespan_h",
    "gpu_hours",
    "gpu_utilization_pct",
    "cold_starts",
    "cold_start_rate_pct",
    "queries_waiting_for_load_pct",
    "latency_mean_s",
    "latency_p95_s",
    "accuracy_pct",
)


def percentile(xs: Iterable[float], p: float) -> float:
    """Return the p-quantile of xs by the nearest-rank rule, without interpolation.

    This is sorted(xs)[min(n - 1, int(p * n))] for n values: p = 0.5 gives the upper median of an
    even number of values, and p = 1.0 the largest value. xs must not be empty.
    """
    values = sorted(xs)
    return values[min(len(values) - 1, int(p * len(values)))]


def summary_row(load: str, policy: Policy, seed: int, result: SimulationResult) -> dict[str, object]:
    """Return the summary.csv row of one run.

    Latencies are end to end (total_s, which includes any wait for a load). accuracy_pct counts
    only the queries the judge labelled on the model they were routed to (scored_queries), and the
    share_<tier>_pct columns give the queries of each tier as a percentage of all queries.
    """
    records = result.records
    usage = result.usage
    n = len(records)
    labels = [r.correct for r in records if r.correct is not None]  # the judge labels of the scored queries
    total = [r.total_s for r in records]
    row: dict[str, object] = {
        "load": load,
        "policy": policy,
        "seed": seed,
        "queries": n,
        "makespan_h": result.makespan_s / 3600,
        "gpu_hours": usage.gpu_hours,
        "busy_gpu_hours": usage.busy_gpu_hours,
        "gpu_utilization_pct": 100 * usage.gpu_utilization,
        "cold_starts": usage.cold_starts,
        "cold_start_rate_pct": 100 * usage.cold_start_rate,
        "queries_waiting_for_load_pct": 100 * sum(r.wait_s > 0 for r in records) / n,
        "latency_mean_s": statistics.fmean(total),
        "latency_median_s": statistics.median(total),
        "latency_p95_s": percentile(total, 0.95),
        "execution_success_pct": 100 * sum(r.success for r in records) / n,
        "accuracy_pct": 100 * sum(labels) / len(labels),
        "scored_queries": len(labels),
    }
    for tier in TIER_ORDER:
        row[f"share_{tier.lower()}_pct"] = 100 * sum(r.tier == tier for r in records) / n
    return row


def per_model_rows(load: str, policy: Policy, seed: int, result: SimulationResult) -> list[dict[str, object]]:
    """Return the per_model.csv rows of one run, one per model in MODELS order.

    A model that served no query has an empty latency_mean_s, and one with no judge-labelled query an
    empty accuracy_pct. The GPU-hours and cold starts come from Spin's accounting of the run.
    """
    records = result.records
    rows: list[dict[str, object]] = []
    for m, spec in MODELS.items():
        rs = [r for r in records if r.model == m]
        labels = [r.correct for r in rs if r.correct is not None]
        usage = result.usage.per_model[m]
        rows.append(
            {
                "load": load,
                "policy": policy,
                "seed": seed,
                "model": spec.label,
                "tier": spec.tier,
                "queries": len(rs),
                "share_pct": 100 * len(rs) / len(records),
                "accuracy_pct": 100 * sum(labels) / len(labels) if labels else "",
                "latency_mean_s": statistics.fmean(r.total_s for r in rs) if rs else "",
                "cold_starts": usage.cold_starts,
                "gpu_hours": usage.gpu_hours,
                "busy_gpu_hours": usage.busy_gpu_hours,
            }
        )
    return rows


def cold_by_tier_rows(load: str, policy: Policy, seed: int, records: Sequence[QueryRecord]) -> list[dict[str, object]]:
    """Return the cold_starts_by_tier.csv rows of one run, one per tier in TIER_ORDER.

    A query counts toward the tier of the model it was routed to (not the tier the classifier gave
    it). conditional_rate_pct is the percentage of those queries that triggered a cold start, or 0.0
    when no query went to the tier.
    """
    rows: list[dict[str, object]] = []
    for tier in TIER_ORDER:
        rs = [r for r in records if MODELS[r.model].tier == tier]
        cs = sum(r.cold_start for r in rs)
        rows.append(
            {
                "load": load,
                "policy": policy,
                "seed": seed,
                "tier": tier,
                "routed_queries": len(rs),
                "cold_starts": cs,
                "conditional_rate_pct": 100 * cs / len(rs) if rs else 0.0,
            }
        )
    return rows


def overview_rows(summary_rows: Iterable[Mapping[str, str]]) -> list[dict[str, object]]:
    """Return the overview.csv rows: the mean over seeds of each (load, policy), sorted by (load, policy).

    summary_rows are summary.csv rows as read back from disk, so every value is the rounded string
    that was written, never the full-precision float. Each row has load, policy, the number of seeds,
    then the fmean of every OVERVIEW_METRICS column. Right after gpu_hours come gpu_hours_sd, the
    sample standard deviation over seeds (0.0 for a single seed), and gpu_hours_vs_static_pct,
    100 * (mean / static mean - 1) against the static policy under the same load, or '' when that
    load has no static run.
    """
    groups: defaultdict[tuple[str, str], list[Mapping[str, str]]] = defaultdict(list)
    for r in summary_rows:
        groups[(r["load"], r["policy"])].append(r)
    static = {
        load: statistics.fmean(float(r["gpu_hours"]) for r in rs)
        for (load, policy), rs in groups.items()
        if policy == Policy.STATIC
    }
    rows: list[dict[str, object]] = []
    for (load, policy), rs in sorted(groups.items()):
        row: dict[str, object] = {"load": load, "policy": policy, "seeds": len(rs)}
        for column in OVERVIEW_METRICS:
            vals = [float(r[column]) for r in rs]
            mean = statistics.fmean(vals)
            row[column] = mean
            if column == "gpu_hours":
                row["gpu_hours_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row["gpu_hours_vs_static_pct"] = 100 * (mean / static[load] - 1) if load in static else ""
        rows.append(row)
    return rows
