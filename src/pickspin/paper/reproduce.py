"""Reproduce the paper's tables and figures from results/traces/ and check them against the paper.

reproduce() writes the four table CSVs, the figures and verification.csv, and builds the 84
paper-versus-reproduced checks in their published order. It returns a report and neither prints nor
exits; the command line does that.

Each reproduced value is rounded the way the paper prints it, and a check matches when both strings
are equal once thousands separators and the 'x' of ratios are removed. The CSV files are written by
csv.writer with its default dialect (CRLF line ends, minimal quoting), from strings formatted here.
"""

import csv
import logging
import statistics
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from pickspin.paper.analysis import (
    ModelBaseline,
    RoutingTable,
    convergence,
    load_paper_traces,
    routing_table,
    size_class_totals,
    static_summary,
)
from pickspin.paper.constants import (
    BENCHMARK_COLUMNS,
    DISPLAY_NAME,
    PAPER_FIG2A_TOK_S,
    PAPER_FIG2B_LATENCY_S,
    PAPER_MODELS,
    PAPER_TABLE1,
    ROUTED_DISPLAY_NAME,
    SIZE_CLASS,
)
from pickspin.paper.figures import make_figures

log = logging.getLogger(__name__)


def _normalise(value: str) -> str:
    """Remove thousands separators and the 'x' of ratios, so '31,019' equals '31019' and '10x' equals '10'."""
    return value.replace(",", "").replace("x", "")


@dataclass(frozen=True, slots=True)
class Check:
    """One published value next to the reproduced one."""

    claim: str
    paper: str
    reproduced: str

    @property
    def matches(self) -> bool:
        """True if the values are equal once ',' and 'x' are removed from both."""
        return _normalise(self.paper) == _normalise(self.reproduced)


def _verdict(check: Check) -> str:
    """The match column of verification.csv and of the report: 'yes' or 'NO'."""
    return "yes" if check.matches else "NO"


def build_checks(
    *,
    n_queries: int,
    static_runs: int,
    summary: Mapping[str, ModelBaseline],
    table: RoutingTable,
    sizes: Mapping[str, int],
    rates: Mapping[str, Sequence[float]],
) -> list[Check]:
    """Return the 84 checks in the order they are published.

    These are 14 static-baseline values (Sec. VII-A), four values per model of Table I, 16 values of
    Table I's last rows, the oracle analysis, Table II and Fig. 3, then the nine bars of Fig. 2a and
    the nine of Fig. 2b. n_queries is the size of the routed run, static_runs the number of
    static-baseline runs of the nine models, sizes the output of size_class_totals and rates the Fig. 3
    series of convergence. Every model must have at least one counted static run and one scored routed
    query, and every rate series at least one point.
    """
    s = summary
    per_model = table.per_model
    q7, g27 = s["qwen2.5_7B"], s["gemma3_27B"]
    tps = [s[m].tok_s for m in PAPER_MODELS]
    lat = [s[m].latency_s for m in PAPER_MODELS]
    judged_total = table.judged_total
    unjudged = n_queries - judged_total
    complex_share = 100 * (per_model["qwen2.5_14B"].n + per_model["gemma3_27B"].n) / n_queries

    # Each check is (claim, paper value, reproduced value), rounded the way the paper prints it.
    checks = [
        Check("Queries across 8 benchmarks", "31,019", f"{n_queries:,}"),
        Check("Static baseline runs (9 models x queries)", "279,171", f"{static_runs:,}"),
        Check("Total inference runs", "310,190", f"{static_runs + n_queries:,}"),
        Check("Throughput, Llama-3.2-1B (tok/s)", "45.8", f"{tps[0]:.1f}"),
        Check("Throughput, Gemma-3-27B (tok/s)", "4.6", f"{tps[-1]:.1f}"),
        Check("Throughput ratio small/large", "10x", f"{tps[0] / tps[-1]:.0f}x"),
        Check("Latency, Llama-3.2-1B (s)", "1.51", f"{lat[0]:.2f}"),
        Check("Latency, Gemma-3-27B (s)", "21.66", f"{lat[-1]:.2f}"),
        Check("Latency ratio large/small", "14x", f"{lat[-1] / lat[0]:.0f}x"),
        Check("Accuracy, Qwen2.5-14B (%)", "62.6", f"{s['qwen2.5_14B'].accuracy:.1f}"),
        Check("Accuracy, Gemma-3-27B (%)", "56.6", f"{g27.accuracy:.1f}"),
        Check("Accuracy, Qwen2.5-7B (%)", "56.3", f"{q7.accuracy:.1f}"),
        Check("Latency, Qwen2.5-7B (s)", "4.83", f"{q7.latency_s:.2f}"),
        Check("Latency ratio Gemma-3-27B / Qwen2.5-7B", "4.5x", f"{g27.latency_s / q7.latency_s:.1f}x"),
    ]
    for m, (n, pct, acc, latency) in PAPER_TABLE1.items():
        d = per_model[m]
        # The paper prints shares below 1% with two decimals.
        pct_fmt = f"{100 * d.n / n_queries:.2f}" if pct < 1 else f"{100 * d.n / n_queries:.1f}"
        checks += [
            Check(f"Table I {ROUTED_DISPLAY_NAME[m]}: queries", f"{n:,}", f"{d.n:,}"),
            Check(f"Table I {ROUTED_DISPLAY_NAME[m]}: % of queries", f"{pct}", pct_fmt),
            Check(f"Table I {ROUTED_DISPLAY_NAME[m]}: accuracy (%)", f"{acc}", f"{100 * d.correct / d.n:.1f}"),
            Check(f"Table I {ROUTED_DISPLAY_NAME[m]}: latency (s)", f"{latency}", f"{statistics.mean(d.lat):.2f}"),
        ]
    checks += [
        Check("Table I unscored queries (no judge label)", "2,635", f"{unjudged:,}"),
        Check("Table I unscored share (%)", "8.5", f"{100 * unjudged / n_queries:.1f}"),
        Check("Table I overall accuracy (%)", "49.7", f"{100 * table.ps_ok / table.with_any:.1f}"),
        Check("Table I overall latency (s)", "23.4", f"{statistics.mean(table.any_lat):.1f}"),
        Check("Oracle accuracy (%)", "86.8", f"{100 * table.oracle_ok / table.with_any:.1f}"),
        Check(
            "Queries no model solves (%)", "13.2", f"{100 * (table.with_any - table.oracle_ok) / table.with_any:.1f}"
        ),
        Check("Share of oracle ceiling captured (%)", "57.2", f"{100 * table.ps_ok / table.oracle_ok:.1f}"),
        Check("Share of queries routed to the 14B and 27B models (%)", "3.2", f"{complex_share:.1f}"),
        Check("Table II scored routed queries, 1-3B models", "5,342", f"{sizes['1-3B']:,}"),
        Check("Table II scored routed queries, 7-9B models", "22,062", f"{sizes['7-9B']:,}"),
        Check("Table II scored routed queries, 14-27B models", "980", f"{sizes['14-27B']:,}"),
        Check("Table II scored routed queries, total", "28,384", f"{judged_total:,}"),
        Check("Fig. 3 rate at 31,000 queries, Llama-3.1-8B (%)", "46.1", f"{rates['llama3.1_8B'][-1]:.1f}"),
        Check("Fig. 3 rate at 31,000 queries, Qwen2.5-7B (%)", "27.8", f"{rates['qwen2.5_7B'][-1]:.1f}"),
        Check("Fig. 3 rate at 31,000 queries, Llama-3.2-1B (%)", "19.6", f"{rates['llama3.2_1B'][-1]:.1f}"),
        Check("Fig. 3 rate at 31,000 queries, Gemma-2-9B (%)", "2.5", f"{rates['gemma2_9B'][-1]:.1f}"),
    ]
    for m, paper in zip(PAPER_MODELS, PAPER_FIG2A_TOK_S):
        checks.append(Check(f"Fig. 2a throughput {DISPLAY_NAME[m]} (tok/s)", paper, f"{s[m].tok_s:.1f}"))
    for m, paper in zip(PAPER_MODELS, PAPER_FIG2B_LATENCY_S):
        checks.append(Check(f"Fig. 2b latency {DISPLAY_NAME[m]} (s)", paper, f"{s[m].latency_s:.1f}"))
    return checks


def write_table(path: Path, header: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    """Write a CSV table, creating its directory if needed.

    The file is UTF-8 and uses the csv module's default dialect, so lines end with CRLF on every
    platform. Values are written with str(), so floats should arrive already formatted.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def write_tables(
    out_dir: Path,
    *,
    summary: Mapping[str, ModelBaseline],
    per_bench: Mapping[str, Mapping[str, float]],
    table: RoutingTable,
    n_queries: int,
    sizes: Mapping[str, int],
) -> None:
    """Write the static-baseline, Table I and Table II CSVs to out_dir.

    The files are static_baseline_summary.csv, static_baseline_per_benchmark_accuracy.csv,
    table1_routing.csv and table2_scored_routed_queries_by_size.csv. A benchmark a model has no counted
    run for gets accuracy 0.0, and a Table I model without scored queries gets '-' for accuracy and
    latency.
    """
    write_table(
        out_dir / "static_baseline_summary.csv",
        ["model", "size_class", "runs", "runs_with_output", "accuracy_pct", "tokens_per_s", "latency_s"],
        [
            [
                DISPLAY_NAME[m],
                SIZE_CLASS[m],
                s.runs,
                s.counted,
                f"{s.accuracy:.1f}",
                f"{s.tok_s:.1f}",
                f"{s.latency_s:.2f}",
            ]
            for m, s in summary.items()
        ],
    )
    write_table(
        out_dir / "static_baseline_per_benchmark_accuracy.csv",
        ["model", *BENCHMARK_COLUMNS],
        [[DISPLAY_NAME[m]] + [f"{per_bench[m].get(b, 0):.1f}" for b in BENCHMARK_COLUMNS] for m in PAPER_MODELS],
    )

    judged_total = table.judged_total
    unjudged = n_queries - judged_total
    t1: list[list[object]] = []
    for m in PAPER_MODELS:
        d = table.per_model[m]
        t1.append(
            [
                ROUTED_DISPLAY_NAME[m],
                d.n,
                f"{100 * d.n / n_queries:.2f}",
                f"{100 * d.correct / d.n:.1f}" if d.n else "-",
                f"{statistics.mean(d.lat):.2f}" if d.lat else "-",
            ]
        )
    t1.append(["Unscored (no judge label)", unjudged, f"{100 * unjudged / n_queries:.1f}", "-", "-"])
    t1.append(
        [
            "Total",
            n_queries,
            "100",
            f"{100 * table.ps_ok / table.with_any:.1f}",
            f"{statistics.mean(table.any_lat):.1f}",
        ]
    )
    write_table(out_dir / "table1_routing.csv", ["model", "queries", "pct", "accuracy_pct", "latency_s"], t1)

    write_table(
        out_dir / "table2_scored_routed_queries_by_size.csv",
        ["model_size", "scored_routed_queries"],
        [[t, n] for t, n in sizes.items()] + [["Total", judged_total]],
    )


def write_verification(path: Path, checks: Sequence[Check]) -> None:
    """Write every check with 'yes' or 'NO' to a CSV file."""
    write_table(
        path,
        ["claim", "paper", "reproduced", "match"],
        [[c.claim, c.paper, c.reproduced, _verdict(c)] for c in checks],
    )


def format_report(checks: Sequence[Check]) -> str:
    """Return the comparison table followed by the 'N/84 numbers match the paper.' line.

    The claim column is as wide as the longest claim, the paper and reproduced columns are
    right-aligned to 20 characters, and a blank line comes before the count. There is no trailing
    newline.
    """
    width = max((len(c.claim) for c in checks), default=0)
    lines = [f"{'claim':<{width}}  {'paper':>20}  {'reproduced':>20}  match"]
    lines += [f"{c.claim:<{width}}  {c.paper:>20}  {c.reproduced:>20}  {_verdict(c)}" for c in checks]
    n_ok = sum(c.matches for c in checks)
    lines += ["", f"{n_ok}/{len(checks)} numbers match the paper."]
    return "\n".join(lines)


@dataclass(frozen=True)
class ReproductionReport:
    """The checks of a reproduction and where its outputs went."""

    checks: tuple[Check, ...]
    out_dir: Path
    figures_written: bool

    @property
    def matched(self) -> int:
        """Number of checks that match the paper."""
        return sum(c.matches for c in self.checks)

    @property
    def all_match(self) -> bool:
        """True if every check matches the paper."""
        return self.matched == len(self.checks)


def reproduce(traces_dir: Path, out_dir: Path) -> ReproductionReport:
    """Compute every table, figure and check from the traces in traces_dir and write them to out_dir.

    Writes the four table CSVs and verification.csv to out_dir and the three figures to
    out_dir/figures (skipped, with a warning, if matplotlib is missing). Only the three trace files are
    read; the released data under data/ is not needed.
    """
    traces = load_paper_traces(traces_dir)
    static_runs = sum(len(traces.static.get(m, [])) for m in PAPER_MODELS)
    n_queries = len(traces.routed)
    log.debug(
        "Read %d static-baseline runs of the nine models, %d judge labels and %d routed queries from %s",
        static_runs,
        len(traces.judged),
        n_queries,
        traces_dir,
    )

    summary, per_bench = static_summary(traces)
    table = routing_table(traces.routed, traces.judged)
    sizes = size_class_totals(table)
    write_tables(out_dir, summary=summary, per_bench=per_bench, table=table, n_queries=n_queries, sizes=sizes)

    xs, rates = convergence(traces.routed)
    figures_written = make_figures(summary, xs, rates, out_dir / "figures")

    checks = build_checks(
        n_queries=n_queries, static_runs=static_runs, summary=summary, table=table, sizes=sizes, rates=rates
    )
    write_verification(out_dir / "verification.csv", checks)
    log.debug("Wrote the tables, verification.csv%s to %s", " and the figures" if figures_written else "", out_dir)
    return ReproductionReport(checks=tuple(checks), out_dir=out_dir, figures_written=figures_written)
