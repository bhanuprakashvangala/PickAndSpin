"""Run the policy x seed grid of one load and write its outputs.

run_load writes summary.csv, per_model.csv and cold_starts_by_tier.csv to <out>/<load>/, where <load>
is closed-<workers> or poisson-<rate>qps, plus a per-query trace queries_<policy>.csv.gz for the first
seed when asked. write_overview rebuilds <out>/overview.csv from every summary.csv under <out>,
including those of earlier runs. Progress goes through logging, one line per run.

The files are byte for byte those of the v1.1.0 simulator. CSVs are UTF-8 in the csv module's default
dialect (CRLF line ends, minimal quoting), floats are rounded with round() and written in their
shortest form, and enum values such as the policy and the tier are written as their plain strings.
"""

import csv
import gzip
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

from pickspin.simulation.engine import QueryRecord, SimulationSettings, simulate
from pickspin.simulation.inputs import TierAssignment, TraceData
from pickspin.simulation.metrics import cold_by_tier_rows, overview_rows, per_model_rows, summary_row
from pickspin.simulation.policies import Policy

log = logging.getLogger(__name__)

# The columns of a per-query trace, in order: QueryRecord fields without the worker.
QUERY_TRACE_COLUMNS: Final[tuple[str, ...]] = (
    "order",
    "id",
    "benchmark",
    "tier",
    "stage",
    "model",
    "arrive",
    "start",
    "end",
    "infer_s",
    "wait_s",
    "total_s",
    "cold_start",
    "success",
    "correct",
)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write rows to a CSV file whose header is the first row's keys, in order.

    Floats are rounded to 4 decimal places, so 0.5 is written as 0.5 and 1/3 as 0.3333; every other
    value, such as True, 3 or '', is written unchanged. The parent directory is created if needed.
    Raises ValueError if there are no rows.
    """
    if not rows:
        raise ValueError(f"no rows to write to {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


def write_query_trace(path: Path, records: Sequence[QueryRecord]) -> None:
    """Write the per-query trace of one run as gzipped CSV, in dispatch order.

    The header is QUERY_TRACE_COLUMNS and the rows are sorted by order, the position in which the
    queries were dispatched. Floats are rounded to 3 decimal places, and a query without a judge label
    has an empty 'correct'. The parent directory is created if needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(QUERY_TRACE_COLUMNS)
        for record in sorted(records, key=lambda r: r.order):
            values = [getattr(record, column) for column in QUERY_TRACE_COLUMNS]
            writer.writerow([round(v, 3) if isinstance(v, float) else v for v in values])


def run_load(
    policies: Sequence[Policy],
    seeds: Sequence[int],
    settings: SimulationSettings,
    data: TraceData,
    tiers: Mapping[str, TierAssignment],
    out_root: Path,
    *,
    write_queries: bool = False,
) -> Path:
    """Simulate every policy with every seed under one load, write the CSVs and return their directory.

    The directory is out_root / settings.load_name. Policies are the outer loop and seeds the inner
    one, which is also the order of the rows in each CSV. Each run logs one line at INFO. With
    write_queries, the per-query trace of each policy's run with seeds[0] is written as
    queries_<policy>.csv.gz.
    """
    load = settings.load_name
    out = out_root / load
    summary: list[dict[str, object]] = []
    per_model: list[dict[str, object]] = []
    cold_tier: list[dict[str, object]] = []
    for policy in policies:
        for seed in seeds:
            result = simulate(policy, seed, data, tiers, settings)
            row = summary_row(load, policy, seed, result)
            summary.append(row)
            per_model += per_model_rows(load, policy, seed, result)
            cold_tier += cold_by_tier_rows(load, policy, seed, result.records)
            log.info(
                "%s %-24s seed %s: %.2f GPU-h, util %.1f%%, cold starts %s (%.2f%%), latency %.2fs, accuracy %.1f%%",
                load,
                policy,
                seed,
                row["gpu_hours"],
                row["gpu_utilization_pct"],
                row["cold_starts"],
                row["cold_start_rate_pct"],
                row["latency_mean_s"],
                row["accuracy_pct"],
            )
            if write_queries and seed == seeds[0]:
                write_query_trace(out / f"queries_{policy}.csv.gz", result.records)
    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_model.csv", per_model)
    write_csv(out / "cold_starts_by_tier.csv", cold_tier)
    log.info("Wrote %s", out)
    return out


def write_overview(root: Path) -> Path:
    """Rebuild root/overview.csv from every root/*/summary.csv and return its path.

    Every load directory under root counts, including those written by earlier invocations. The
    summaries are read back from disk, so the overview averages their rounded values.
    """
    rows: list[dict[str, str]] = []
    paths = sorted(root.glob("*/summary.csv"))
    for path in paths:
        with path.open(encoding="utf-8", newline="") as f:
            rows += csv.DictReader(f)
    out = root / "overview.csv"
    write_csv(out, overview_rows(rows))
    log.debug("Averaged %d summary files into %s", len(paths), out)
    return out
