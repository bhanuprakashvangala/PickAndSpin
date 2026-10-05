"""Complexity labels and the seeded train/validation split of the DistilBERT training data."""

import csv
import gzip
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import pytest

from pickspin.config import MODELS, TIERS, Tier
from pickspin.training.labels import (
    LARGE_MODELS,
    MEDIUM_MODELS,
    SMALL_MODELS,
    SPLIT_SEED,
    TRAIN_FRACTION,
    complexity_label,
    generate_labeled_dataset,
    load_judgments,
)

# Synthetic inputs: (id, benchmark, query text) in file order. q04 appears twice; as in the old
# script, its second text wins and its first position stays. Every label is SIMPLE or COMPLEX, mbpp
# has only COMPLEX and gsm8k only SIMPLE, so zero counts must be filled in.
QUERIES = [
    ("q00", "mbpp", "Write a function that reverses a list."),
    ("q01", "arc", "Which gas do plants absorb?"),
    ("q02", "mbpp", "Implement quicksort."),
    ("q03", "gsm8k", "Wie viele \u00c4pfel sind \u00fcbrig? \u20ac 3 \u00f7 1"),
    ("q04", "arc", "First text of q04"),
    ("q05", "gsm8k", "What is 2 + 2?"),
    ("q06", "arc", "Name the largest planet."),
    ("q07", "mbpp", 'Write a regex for "e-mail" addresses.'),
    ("q08", "gsm8k", "How many legs do 3 spiders have?"),
    ("q04", "arc", "Second text of q04"),
    ("q09", "arc", "Is the sky blue?"),
    ("q10", "gsm8k", "Solve x^2 = 4."),
]

# (model, id, benchmark, is_correct) in the column order of results/traces/judgments.csv.gz.
JUDGMENTS = [
    ("kimi_1T_MoE", "q00", "mbpp", "1"),
    ("llama3.2_1B", "q00", "mbpp", "0"),
    ("llama3.2_1B", "q01", "arc", "1"),
    ("gemma2_2B", "q03", "gsm8k", "1"),
    ("qwen2.5_7B", "q03", "gsm8k", "1"),
    ("qwen2.5_1.5B", "q04", "arc", "True"),  # only '1' counts as correct
    ("gemma3_27B", "q04", "arc", "1"),
    ("llama3.2_3B", "q05", "gsm8k", "1"),
    *[(m, "q06", "arc", "0") for m in ("llama3.2_1B", "qwen2.5_1.5B", "llama3.1_8B", "qwen2.5_14B")],
    ("llama3_70B", "q06", "arc", "1"),
    ("qwen2.5_14B", "q07", "mbpp", "0"),
    ("gemma3_27B", "q07", "mbpp", "0"),
    ("qwen2.5_1.5B", "q08", "gsm8k", "1"),
    ("llama3.2_1B", "q09", "arc", "1"),
    ("kimi_1T_MoE", "q09", "arc", "1"),
    ("gemma2_2B", "q10", "gsm8k", "1"),
    ("gemma2_27B", "q02", "mbpp", "1"),  # a model in no label group is ignored
    ("llama3.2_1B", "zz", "arc", "1"),  # a query that is not in the queries file is ignored
]

# Produced by src/classifier/generate_labels.py at tag v1.1.0 from the inputs above (line ends
# normalised to LF): the ids of train.jsonl and val.jsonl in file order, and label_stats.json.
EXPECTED_TRAIN_IDS = ["q07", "q03", "q02", "q08", "q05", "q06", "q09", "q04"]
EXPECTED_VAL_IDS = ["q00", "q01", "q10"]
EXPECTED_STATS_JSON = """\
{
  "total_samples": 11,
  "train_samples": 8,
  "val_samples": 3,
  "label_distribution": {
    "COMPLEX": 5,
    "SIMPLE": 6,
    "MEDIUM": 0
  },
  "benchmark_distribution": {
    "mbpp": {
      "COMPLEX": 3,
      "SIMPLE": 0,
      "MEDIUM": 0
    },
    "arc": {
      "SIMPLE": 2,
      "COMPLEX": 2,
      "MEDIUM": 0
    },
    "gsm8k": {
      "SIMPLE": 4,
      "MEDIUM": 0,
      "COMPLEX": 0
    }
  },
  "model_tiers": {
    "small": [
      "llama3.2_1B",
      "llama3.2_3B",
      "gemma2_2B",
      "qwen2.5_1.5B"
    ],
    "medium": [
      "llama3.1_8B",
      "gemma2_9B",
      "qwen2.5_7B",
      "qwen2.5_14B"
    ],
    "large": [
      "gemma3_27B",
      "llama3_70B",
      "kimi_1T_MoE"
    ]
  }
}"""


def write_inputs(directory: Path) -> tuple[Path, Path]:
    """Write QUERIES and JUDGMENTS as queries.jsonl.gz and judgments.csv.gz; return (judgments, queries)."""
    queries = directory / "queries.jsonl.gz"
    with gzip.open(queries, "wt", encoding="utf-8") as f:
        for query_id, benchmark, text in QUERIES:
            record = {"id": query_id, "benchmark": benchmark, "query": text, "ground_truth": "", "query_type": "t"}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    judgments = directory / "judgments.csv.gz"
    with gzip.open(judgments, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "id", "benchmark", "is_correct"])
        writer.writerows(JUDGMENTS)
    return judgments, queries


@pytest.fixture
def labelled(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """Run generate_labeled_dataset on the synthetic inputs; return (output dir, returned stats)."""
    judgments, queries = write_inputs(tmp_path)
    out = tmp_path / "out" / "classifier"
    return out, generate_labeled_dataset(judgments, queries, out)


def read_lines(path: Path) -> list[str]:
    """Return the lines of a text file written with platform line ends, checking those line ends."""
    raw = path.read_bytes().decode("ascii")
    assert raw.endswith(os.linesep)
    if os.linesep == "\r\n":
        assert "\n" not in raw.replace("\r\n", "")
    return raw.split(os.linesep)[:-1]


def test_label_groups_are_pinned_and_are_not_the_routing_tiers() -> None:
    assert SMALL_MODELS == ("llama3.2_1B", "llama3.2_3B", "gemma2_2B", "qwen2.5_1.5B")
    assert MEDIUM_MODELS == ("llama3.1_8B", "gemma2_9B", "qwen2.5_7B", "qwen2.5_14B")
    assert LARGE_MODELS == ("gemma3_27B", "llama3_70B", "kimi_1T_MoE")
    assert (SPLIT_SEED, TRAIN_FRACTION) == (42, 0.8)
    # The small group is the SIMPLE tier; the medium group adds Qwen2.5-14B, a COMPLEX-tier model;
    # the large group holds the remaining catalog model and two models outside the routing pool.
    assert set(SMALL_MODELS) == set(TIERS[Tier.SIMPLE])
    assert set(MEDIUM_MODELS) == {*TIERS[Tier.MEDIUM], "qwen2.5_14B"}
    assert [m for m in LARGE_MODELS if m in MODELS] == ["gemma3_27B"]
    assert set(SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS) >= set(MODELS)


@pytest.mark.parametrize("model", SMALL_MODELS)
def test_any_small_model_correct_is_simple(model: str) -> None:
    judged = {("q", m): True for m in (model, "llama3.1_8B", "gemma3_27B")}
    assert complexity_label("q", judged) is Tier.SIMPLE


@pytest.mark.parametrize("model", MEDIUM_MODELS)
def test_a_medium_model_correct_without_a_small_one_is_medium(model: str) -> None:
    judged = {**{("q", m): False for m in SMALL_MODELS}, ("q", model): True, ("q", "kimi_1T_MoE"): True}
    assert complexity_label("q", judged) is Tier.MEDIUM


def test_only_qwen_14b_correct_is_medium() -> None:
    assert complexity_label("q", {("q", "qwen2.5_14B"): True}) is Tier.MEDIUM


@pytest.mark.parametrize(
    "judged",
    [
        {("q", "gemma3_27B"): True},
        {("q", "llama3_70B"): True},
        {("q", "kimi_1T_MoE"): True},
        {},
        {("q", m): False for m in SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS},
        {("other", "llama3.2_1B"): True, ("q", "gemma2_27B"): True},
    ],
    ids=["gemma3_27B", "llama3_70B", "kimi_1T_MoE", "no labels", "none correct", "other query or model"],
)
def test_only_large_or_no_model_correct_is_complex(judged: dict[tuple[str, str], bool]) -> None:
    assert complexity_label("q", judged) is Tier.COMPLEX


def test_load_judgments_reads_every_model_and_only_one_means_correct(tmp_path: Path) -> None:
    judgments, _ = write_inputs(tmp_path)
    judged = load_judgments(judgments)
    assert len(judged) == len(JUDGMENTS)
    assert judged[("q00", "kimi_1T_MoE")] is True
    assert judged[("q06", "llama3_70B")] is True
    assert judged[("q04", "qwen2.5_1.5B")] is False  # 'True' is not '1'
    assert judged[("q00", "llama3.2_1B")] is False
    assert judged[("q02", "gemma2_27B")] is True


def test_seeded_shuffle_matches_the_old_global_seed_and_shuffle() -> None:
    items = [f"q{i}" for i in range(1000)]
    state = random.getstate()
    try:
        random.seed(SPLIT_SEED)
        old = items.copy()
        random.shuffle(old)
    finally:
        random.setstate(state)
    new = items.copy()
    random.Random(SPLIT_SEED).shuffle(new)
    assert new == old


def test_generate_labeled_dataset_matches_the_v1_1_0_output(labelled: tuple[Path, dict[str, Any]]) -> None:
    out, stats = labelled
    train = [json.loads(line) for line in read_lines(out / "train.jsonl")]
    val = [json.loads(line) for line in read_lines(out / "val.jsonl")]
    assert [r["id"] for r in train] == EXPECTED_TRAIN_IDS
    assert [r["id"] for r in val] == EXPECTED_VAL_IDS
    stats_text = (out / "label_stats.json").read_bytes().decode("ascii")
    assert stats_text.replace(os.linesep, "\n") == EXPECTED_STATS_JSON
    assert json.loads(stats_text) == stats


def test_records_and_split(labelled: tuple[Path, dict[str, Any]]) -> None:
    out, stats = labelled
    train_lines = read_lines(out / "train.jsonl")
    val_lines = read_lines(out / "val.jsonl")
    records = [json.loads(line) for line in train_lines + val_lines]
    n = len({query_id for query_id, _, _ in QUERIES})
    assert len(train_lines) == int(n * 0.8) == stats["train_samples"] == 8
    assert len(val_lines) == n - int(n * 0.8) == stats["val_samples"] == 3
    assert stats["total_samples"] == n
    assert all(list(r) == ["id", "text", "label", "benchmark"] for r in records)
    # json.dumps defaults: ASCII-only lines, non-ASCII text escaped.
    assert all(line == json.dumps(r) for line, r in zip(train_lines + val_lines, records))
    by_id = {r["id"]: r for r in records}
    assert by_id["q03"]["text"] == QUERIES[3][2]
    assert "\\u00c4pfel" in (train_lines + val_lines)[[r["id"] for r in records].index("q03")]
    assert by_id["q04"]["text"] == "Second text of q04"
    assert [by_id[q]["label"] for q in ("q00", "q02", "q04", "q06", "q07")] == ["COMPLEX"] * 5
    assert [by_id[q]["label"] for q in ("q01", "q03", "q05", "q08", "q09", "q10")] == ["SIMPLE"] * 6


def test_label_stats_layout(labelled: tuple[Path, dict[str, Any]]) -> None:
    out, stats = labelled
    assert list(stats) == [
        "total_samples",
        "train_samples",
        "val_samples",
        "label_distribution",
        "benchmark_distribution",
        "model_tiers",
    ]
    assert stats["model_tiers"] == {
        "small": list(SMALL_MODELS),
        "medium": list(MEDIUM_MODELS),
        "large": list(LARGE_MODELS),
    }
    text = (out / "label_stats.json").read_bytes().decode("ascii")
    assert text.startswith("{" + os.linesep + '  "total_samples": 11,')
    assert text.endswith("}")  # no trailing newline


@pytest.mark.parametrize("level", [logging.INFO, logging.CRITICAL + 1], ids=["logged", "logging disabled"])
def test_absent_labels_get_zero_counts_whatever_the_log_level(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, level: int
) -> None:
    # The distribution report fills in the zeros, and they must appear even when nothing is logged.
    caplog.set_level(level, logger="pickspin.training.labels")
    judgments, queries = write_inputs(tmp_path)
    stats = generate_labeled_dataset(judgments, queries, tmp_path / "out")
    assert list(stats["label_distribution"].items()) == [("COMPLEX", 5), ("SIMPLE", 6), ("MEDIUM", 0)]
    assert {b: list(c.items()) for b, c in stats["benchmark_distribution"].items()} == {
        "mbpp": [("COMPLEX", 3), ("SIMPLE", 0), ("MEDIUM", 0)],
        "arc": [("SIMPLE", 2), ("COMPLEX", 2), ("MEDIUM", 0)],
        "gsm8k": [("SIMPLE", 4), ("MEDIUM", 0), ("COMPLEX", 0)],
    }
    assert list(stats["benchmark_distribution"]) == ["mbpp", "arc", "gsm8k"]  # first appearance, not sorted
    messages = [r.getMessage() for r in caplog.records]
    if level == logging.INFO:
        assert "Loaded 21 accuracy entries" in messages
        assert "Loaded 11 unique queries" in messages
        assert "MEDIUM: 0 (0.0%)" in messages
    else:
        assert messages == []


def test_the_global_random_state_is_left_alone(tmp_path: Path) -> None:
    judgments, queries = write_inputs(tmp_path)
    state = random.getstate()
    generate_labeled_dataset(judgments, queries, tmp_path / "out")
    assert random.getstate() == state
