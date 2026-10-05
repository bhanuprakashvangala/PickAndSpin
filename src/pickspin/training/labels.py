"""Complexity labels for training DistilBERT, derived from the judge's results.

Each query gets the smallest label-generation model group that the judge marks correct on it:

- SIMPLE: at least one small model (1-3B) is correct
- MEDIUM: no small model is correct, but a medium model (7-14B) is
- COMPLEX: only a large model (Gemma-3-27B, Llama-3-70B, Kimi-K2) is correct, or no model is

These groups are not the routing tiers of pickspin.config: Qwen2.5-14B is in the medium group, and the
large group includes two models outside the routing pool. The labelled queries are shuffled with a
fixed seed and split 80/20 into data/classifier/train.jsonl and val.jsonl, with the label counts in
label_stats.json.

The files are written in text mode, so their line ends are the platform's (CRLF on Windows, where the
released split was made), and their JSON is ASCII-only.
"""

import json
import logging
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from pickspin.config import TIER_ORDER, Tier
from pickspin.data import load_queries, read_csv_gz

log = logging.getLogger(__name__)

SMALL_MODELS: Final[tuple[str, ...]] = ("llama3.2_1B", "llama3.2_3B", "gemma2_2B", "qwen2.5_1.5B")
MEDIUM_MODELS: Final[tuple[str, ...]] = ("llama3.1_8B", "gemma2_9B", "qwen2.5_7B", "qwen2.5_14B")
LARGE_MODELS: Final[tuple[str, ...]] = ("gemma3_27B", "llama3_70B", "kimi_1T_MoE")

SPLIT_SEED: Final = 42
TRAIN_FRACTION: Final = 0.8

_RULE: Final = "=" * 60


def load_judgments(path: Path) -> dict[tuple[str, str], bool]:
    """Read the judge's label of every (query id, model) pair from judgments.csv.gz.

    A pair is correct when its is_correct field is '1'. Every model in the file is kept, including
    the two large models outside the routing pool.
    """
    judged = {(row["id"], row["model"]): row["is_correct"] == "1" for row in read_csv_gz(path)}
    log.info("Loaded %d accuracy entries", len(judged))
    return judged


def complexity_label(query_id: str, judged: Mapping[tuple[str, str], bool]) -> Tier:
    """Return the complexity label of one query from the judge's labels.

    A model without a label for the query counts as incorrect. A query that only a large model
    answers correctly and a query that no model answers correctly are both COMPLEX.
    """
    if any(judged.get((query_id, model), False) for model in SMALL_MODELS):
        return Tier.SIMPLE
    if any(judged.get((query_id, model), False) for model in MEDIUM_MODELS):
        return Tier.MEDIUM
    return Tier.COMPLEX


def _write_jsonl(path: Path, records: Sequence[Mapping[str, str]]) -> None:
    """Write one JSON object per line (ASCII-only JSON, platform line ends)."""
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def generate_labeled_dataset(judgments_csv: Path, queries_file: Path, output_dir: Path) -> dict[str, Any]:
    """Label every query, write the train/validation split and label_stats.json, and return the stats.

    Queries are labelled in the order of the queries file (a repeated id keeps its first position and
    its last text). The labelled queries are shuffled with random.Random(SPLIT_SEED), and the first
    int(n * TRAIN_FRACTION) become the training split. label_stats.json lists every label for the
    whole set and for every benchmark, with a count of 0 where a label does not occur.
    """
    judged = load_judgments(judgments_csv)
    queries = {q.id: q for q in load_queries(queries_file)}
    log.info("Loaded %d unique queries", len(queries))

    records: list[dict[str, str]] = []
    label_counts: defaultdict[Tier, int] = defaultdict(int)
    benchmark_counts: defaultdict[str, defaultdict[Tier, int]] = defaultdict(lambda: defaultdict(int))
    for query_id, query in queries.items():
        label = complexity_label(query_id, judged)
        records.append({"id": query_id, "text": query.query, "label": label, "benchmark": query.benchmark})
        label_counts[label] += 1
        benchmark_counts[query.benchmark][label] += 1

    # The distribution report reads its counts from the defaultdicts whether or not INFO is logged,
    # and must keep doing so (never behind log.isEnabledFor): each lookup adds a count of 0 for a label
    # that does not occur, and label_stats.json lists those zeros after the labels that do occur.
    total = len(records)
    log.info("\n%s\nLABEL DISTRIBUTION\n%s", _RULE, _RULE)
    for label in TIER_ORDER:
        count = label_counts[label]
        pct = count / total * 100
        log.info("%s: %s (%.1f%%)", label, f"{count:,}", pct)

    log.info("\n%s\nDISTRIBUTION BY BENCHMARK\n%s", _RULE, _RULE)
    for benchmark in sorted(benchmark_counts):
        log.info("\n%s:", benchmark)
        for label in TIER_ORDER:
            count = benchmark_counts[benchmark][label]
            log.info("  %s: %d", label, count)

    random.Random(SPLIT_SEED).shuffle(records)
    split = int(len(records) * TRAIN_FRACTION)
    train_data = records[:split]
    val_data = records[split:]

    log.info("\n%s\nTRAIN/VAL SPLIT\n%s", _RULE, _RULE)
    log.info("Training samples: %s", f"{len(train_data):,}")
    log.info("Validation samples: %s", f"{len(val_data):,}")

    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    stats_path = output_dir / "label_stats.json"
    _write_jsonl(train_path, train_data)
    _write_jsonl(val_path, val_data)

    stats: dict[str, Any] = {
        "total_samples": total,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "label_distribution": dict(label_counts),
        "benchmark_distribution": {k: dict(v) for k, v in benchmark_counts.items()},
        "model_tiers": {
            "small": list(SMALL_MODELS),
            "medium": list(MEDIUM_MODELS),
            "large": list(LARGE_MODELS),
        },
    }
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    log.info("\nSaved:\n  - %s\n  - %s\n  - %s", train_path, val_path, stats_path)
    return stats
