"""The DistilBERT training labels regenerate exactly from the released judge labels and queries.

generate_labeled_dataset runs once on the released data, into a temporary directory. Its label_stats.json
must match the SHA-256 in tests/data/label_stats.sha256 (taken with LF line ends) and agree with the stats the
function returns, and its train/validation split must equal
the local data/classifier/{train,val}.jsonl when those git-ignored files are present. The files are
written in text mode, so their line ends are the platform's; comparisons normalise line ends on both
sides, as a checkout may hold the committed file with either.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

from pickspin.paths import Paths
from pickspin.training.labels import generate_labeled_dataset

SPLIT_LINES = {"train.jsonl": 24_815, "val.jsonl": 6_204}  # the released 80/20 split of 31,019 queries
RECORD_KEYS = ["id", "text", "label", "benchmark"]
LABEL_STATS_SHA256 = Path(__file__).parents[1] / "data" / "label_stats.sha256"


@pytest.fixture(scope="module")
def labels(released_data: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    """The output directory and returned stats of one run on the released data."""
    paths = Paths(released_data)
    out = tmp_path_factory.mktemp("classifier")
    stats = generate_labeled_dataset(paths.traces / "judgments.csv.gz", paths.queries, out)
    return out, stats


def normalised(path: Path) -> bytes:
    """The file's bytes with CRLF line ends turned into LF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_label_stats_match_the_recorded_digest(labels):
    out, stats = labels
    produced = out / "label_stats.json"

    assert json.loads(produced.read_text(encoding="utf-8")) == stats
    # The same text as the released file: key order, indent=2 and no newline at the end.
    expected = LABEL_STATS_SHA256.read_text(encoding="utf-8").split()[0]
    assert hashlib.sha256(normalised(produced)).hexdigest() == expected


def test_split_files_have_the_released_sizes_and_platform_line_ends(labels):
    out, _ = labels
    for name, lines in SPLIT_LINES.items():
        data = (out / name).read_bytes()
        assert data.count(b"\n") == lines, name
        assert data.count(os.linesep.encode()) == lines, f"{name} does not use the platform's line ends"
        assert data.isascii(), f"{name} is not ASCII-only JSON"
    with (out / "val.jsonl").open(encoding="utf-8") as f:
        first = json.loads(next(f))
    assert list(first) == RECORD_KEYS
    stats = (out / "label_stats.json").read_bytes()
    assert stats.count(b"\n") == stats.count(os.linesep.encode())
    assert stats.endswith(b"}")


def test_split_files_equal_the_local_ones(labels, released_data):
    local = Paths(released_data).classifier_data
    missing = [name for name in SPLIT_LINES if not (local / name).is_file()]
    if missing:
        pytest.skip(f"no local {' or '.join(missing)} in {local} (git-ignored; `pickspin classifier labels` writes it)")
    out, _ = labels
    for name in SPLIT_LINES:
        assert normalised(out / name) == normalised(local / name), f"{name} differs from {local / name}"
    if (local / "label_stats.json").is_file():
        assert normalised(out / "label_stats.json") == normalised(local / "label_stats.json")
