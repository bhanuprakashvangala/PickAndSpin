"""The simulator's inputs: recorded outcomes per (query, model), judge labels and the cached tiers.

load_trace_data reads, for every (query, model) pair of the models in pickspin.config.MODELS, the
success and latency recorded in the static baseline and the judge's correctness label. The simulator
counts a flag as true when it is 'True', 'true' or '1', and an empty latency as 0.0 seconds. Rows of
other models are skipped, and when a pair appears twice the later row wins.

The hybrid classifier's tier for every query is cached in data/query_tiers.csv.gz. DistilBERT
classifies the queries again only when asked to, or when the cache is missing.
"""

import csv
import gzip
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias

from pickspin.config import MODELS, Tier
from pickspin.data import Query, load_queries, read_csv_gz
from pickspin.errors import ConfigError
from pickspin.pick.classifier import HybridClassifier, Stage

log = logging.getLogger(__name__)

# A query's tier and the classifier stage that assigned it.
TierAssignment: TypeAlias = tuple[Tier, Stage]


@dataclass(frozen=True)
class TraceData:
    """The recorded data a simulation replays.

    queries are in file order. runs maps (query id, model) to (success, inference seconds) and correct
    maps (query id, model) to the judge's label; a pair the judge did not score has no entry.
    """

    queries: list[Query]
    runs: dict[tuple[str, str], tuple[bool, float]]
    correct: dict[tuple[str, str], bool]


def load_trace_data(queries_path: Path, traces_dir: Path) -> TraceData:
    """Read the queries, the static-baseline outcomes and the judge labels.

    The outcomes come from traces_dir/static_baseline.csv.gz and the labels from
    traces_dir/judgments.csv.gz; only the models in MODELS are kept. Latencies are recorded in
    milliseconds and returned in seconds.
    """
    queries = load_queries(queries_path)
    runs: dict[tuple[str, str], tuple[bool, float]] = {}
    for r in read_csv_gz(traces_dir / "static_baseline.csv.gz"):
        if r["model"] in MODELS:
            ms = r["latency_ms"]
            # Written exactly as in v1.1.0: ms / 1000 and ms * 0.001 differ in the last bit for many values.
            runs[(r["id"], r["model"])] = (_flag(r["success"]), float(ms) / 1000 if ms else 0.0)
    correct = {
        (r["id"], r["model"]): _flag(r["is_correct"])
        for r in read_csv_gz(traces_dir / "judgments.csv.gz")
        if r["model"] in MODELS
    }
    return TraceData(queries=queries, runs=runs, correct=correct)


def _flag(value: str) -> bool:
    """The simulator's reading of a boolean trace column: 'True', 'true' and '1' are true."""
    return value in ("True", "true", "1")


def read_tier_cache(path: Path) -> dict[str, TierAssignment]:
    """Read the cached (tier, stage) of every query id; a value that is not a tier or stage raises ValueError."""
    return {r["id"]: (Tier(r["tier"]), Stage(r["stage"])) for r in read_csv_gz(path)}


def write_tier_cache(path: Path, queries: Sequence[Query], labels: Sequence[TierAssignment]) -> None:
    """Write the (tier, stage) of every query to the gzipped tier cache, in query order.

    labels[i] belongs to queries[i]. The columns are id, tier and stage.
    """
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tier", "stage"])
        for q, (t, s) in zip(queries, labels):
            w.writerow([q.id, t, s])


def query_tiers(
    queries: Sequence[Query], cache: Path, *, reclassify: bool = False, model_dir: Path | None = None
) -> dict[str, TierAssignment]:
    """Return the hybrid classifier's (tier, stage) for every query, from the cache when possible.

    The cache is read when it exists, unless reclassify is set. Otherwise the queries are classified
    with the keyword lists and the DistilBERT saved in model_dir, which needs the [classifier] extra,
    and the cache is rewritten. Raises ConfigError when the queries must be classified but model_dir
    is None.
    """
    if cache.exists() and not reclassify:
        return read_tier_cache(cache)
    if model_dir is None:
        reason = "reclassifying was requested" if reclassify else f"there is no tier cache at {cache}"
        raise ConfigError(f"{reason}, so the queries must be classified, but no DistilBERT model directory was given")
    log.info("Classifying %s queries (keyword lists, then DistilBERT) ...", f"{len(queries):,}")
    labels = HybridClassifier.from_pretrained(model_dir).classify_many([q.query for q in queries])
    write_tier_cache(cache, queries, labels)
    return {q.id: label for q, label in zip(queries, labels)}


def describe_stages(n_queries: int, tiers: Mapping[str, TierAssignment]) -> str:
    """Return a one-line count of the queries each classifier stage assigned, most frequent first.

    The format is '<n> queries; classifier stages: <stage> <count> (<percent>%), ...', with thousands
    separators and percentages of n_queries to one decimal place.
    """
    stages = Counter(s for _, s in tiers.values())
    return f"{n_queries:,} queries; classifier stages: " + ", ".join(
        f"{k} {v:,} ({100 * v / n_queries:.1f}%)" for k, v in stages.most_common()
    )
