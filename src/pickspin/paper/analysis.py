"""Analysis of the released traces for the paper's Sec. VII.

Loads the traces with the reproduction's own parsing rules (a judge label counts as correct only when
is_correct is '1') and computes the static-baseline summary (Sec. VII-A, Fig. 2), Table I with the
oracle analysis (Sec. VII-B), the Table II totals by model size and the Fig. 3 convergence series.
Apart from loading, every function is pure.

The arithmetic is kept exactly as published: throughput is toks / (ms / 1000.0), means use
statistics.mean (which is exact), and a share is 100 * part / whole. Changing the shape of these
expressions can change the last digit that the tables print.
"""

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from pickspin.data import read_csv_gz
from pickspin.paper.constants import (
    CONVERGENCE_MODELS,
    CONVERGENCE_STEP,
    PAPER_MODELS,
    SIZE_CLASS,
    canonical_model,
)


@dataclass(frozen=True)
class PaperTraces:
    """The released traces: static-baseline rows per model, judge labels, and the routed run in order.

    static holds the rows of all eleven static-baseline models (including llama3_70B and kimi_1T_MoE),
    keyed by model in a defaultdict(list). judged maps (query id, model) to True when the judge marked
    the answer correct. routed is the routed run sorted by its order column.
    """

    static: Mapping[str, list[dict[str, str]]]
    judged: Mapping[tuple[str, str], bool]
    routed: list[dict[str, str]]


def load_paper_traces(traces_dir: Path) -> PaperTraces:
    """Read static_baseline.csv.gz, judgments.csv.gz and pick_spin_routed.csv.gz from traces_dir.

    Every value stays a string except the judge label, which is True only when is_correct is '1'. If a
    (query id, model) pair is judged twice, the last row wins.
    """
    static: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for r in read_csv_gz(traces_dir / "static_baseline.csv.gz"):
        static[r["model"]].append(r)
    judged: dict[tuple[str, str], bool] = {}
    for r in read_csv_gz(traces_dir / "judgments.csv.gz"):
        judged[(r["id"], r["model"])] = r["is_correct"] == "1"
    routed = sorted(read_csv_gz(traces_dir / "pick_spin_routed.csv.gz"), key=lambda r: int(r["order"]))
    return PaperTraces(static=static, judged=judged, routed=routed)


@dataclass(frozen=True, slots=True)
class ModelBaseline:
    """One model's static-baseline figures: run counts, judged accuracy (%), tokens/s and latency (s)."""

    runs: int
    counted: int
    accuracy: float
    tok_s: float
    latency_s: float


def static_summary(traces: PaperTraces) -> tuple[dict[str, ModelBaseline], dict[str, dict[str, float]]]:
    """Return each model's static-baseline figures and its accuracy (%) per benchmark.

    Both results follow PAPER_MODELS order. A run counts if it returned at least one token with a
    positive latency, and accuracy is the share of counted runs that the judge marked correct; a run
    without a judge label counts as wrong. The per-benchmark accuracies keep the order in which each
    benchmark first appears among the model's counted runs.
    """
    summary: dict[str, ModelBaseline] = {}
    per_bench: dict[str, dict[str, float]] = {}
    for m in PAPER_MODELS:
        runs = traces.static.get(m, [])
        tps: list[float] = []
        lat: list[float] = []
        correct, total = 0, 0
        bench: defaultdict[str, list[int]] = defaultdict(lambda: [0, 0])  # benchmark -> [correct, counted]
        for r in runs:
            ms, toks = float(r["latency_ms"]), int(r["completion_tokens"])
            if ms <= 0 or toks <= 0:
                continue
            tps.append(toks / (ms / 1000.0))
            lat.append(ms)
            total += 1
            ok = traces.judged.get((r["id"], m)) is True
            correct += ok
            bench[r["benchmark"]][0] += ok
            bench[r["benchmark"]][1] += 1
        summary[m] = ModelBaseline(
            runs=len(runs),
            counted=total,
            accuracy=100 * correct / total,
            tok_s=statistics.mean(tps),
            latency_s=statistics.mean(lat) / 1000,
        )
        per_bench[m] = {b: 100 * c / n for b, (c, n) in bench.items()}
    return summary, per_bench


@dataclass(slots=True)
class RoutedModelStats:
    """Scored routed queries of one model: count, correct ones and their latencies."""

    n: int = 0
    correct: int = 0
    lat: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RoutingTable:
    """Table I and the oracle analysis of the routed run.

    per_model has an entry for every model of PAPER_MODELS, in that order, even if no query routed to
    it was scored. with_any counts the routed queries that have a judge label for at least one of the
    nine models, and any_lat holds their latencies in run order (the Total row of Table I). Of those,
    oracle_ok counts the queries that at least one model answered correctly, and ps_ok the ones that
    Pick and Spin answered correctly.
    """

    per_model: dict[str, RoutedModelStats]
    with_any: int
    oracle_ok: int
    ps_ok: int
    any_lat: list[float]

    @property
    def judged_total(self) -> int:
        """Number of routed queries scored with a judge label."""
        return sum(d.n for d in self.per_model.values())


def routing_table(routed: Sequence[Mapping[str, str]], judged: Mapping[tuple[str, str], bool]) -> RoutingTable:
    """Score the routed run with the judge's labels (Table I and the oracle analysis).

    A routed query is scored with the static-baseline judge label of the model slot it was routed to
    (see canonical_model). Queries whose slot has no label stay unscored.
    """
    # Every slot has an entry; a scored model outside the nine slots is added after them and still
    # counts towards judged_total.
    per_model: defaultdict[str, RoutedModelStats] = defaultdict(
        RoutedModelStats, {m: RoutedModelStats() for m in PAPER_MODELS}
    )
    with_any, oracle_ok, ps_ok = 0, 0, 0
    any_lat: list[float] = []  # latency of every query that has at least one label (Total row of Table I)
    for r in routed:
        qid, m = r["id"], canonical_model(r["model"])
        labels = {x: judged[(qid, x)] for x in PAPER_MODELS if (qid, x) in judged}
        mine = judged.get((qid, m))
        if mine is not None:
            d = per_model[m]
            d.n += 1
            d.correct += mine
            d.lat.append(float(r["total_latency"]))
        if labels:
            with_any += 1
            any_lat.append(float(r["total_latency"]))
            if any(labels.values()):
                oracle_ok += 1
                ps_ok += bool(mine)
    return RoutingTable(per_model=dict(per_model), with_any=with_any, oracle_ok=oracle_ok, ps_ok=ps_ok, any_lat=any_lat)


def convergence(
    routed: Sequence[Mapping[str, str]],
    track: Sequence[str] = CONVERGENCE_MODELS,
    step: int = CONVERGENCE_STEP,
) -> tuple[list[float], dict[str, list[float]]]:
    """Return the Fig. 3 series: thousands of queries processed, and each tracked model's selection rate (%).

    A point is taken every `step` queries; the rate is the model's cumulative share of the queries
    routed so far. Models are counted by the raw keys of the routed run, without canonical_model.
    """
    xs: list[float] = []
    ys: dict[str, list[float]] = {m: [] for m in track}
    counts: Counter[str] = Counter()
    for i, r in enumerate(routed, 1):
        counts[r["model"]] += 1
        if i % step == 0:
            xs.append(i / 1000)
            for m in track:
                ys[m].append(100 * counts[m] / i)
    return xs, ys


def size_class_totals(table: RoutingTable) -> dict[str, int]:
    """Return the scored routed queries per model size class (Table II).

    The classes appear in the order of their first model in PAPER_MODELS: 1-3B, 7-9B, then 14-27B.
    """
    totals: defaultdict[str, int] = defaultdict(int)
    for m in PAPER_MODELS:
        totals[SIZE_CLASS[m]] += table.per_model[m].n
    return dict(totals)
