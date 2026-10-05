"""The benchmark query record and the two readers that several workflows share.

Domain-specific parsing, such as which values count as true or which models to keep, stays with each
consumer.
"""

from __future__ import annotations

import csv
import gzip
import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class Query:
    """One benchmark prompt from data/queries.jsonl.gz."""

    id: str
    benchmark: str
    query: str
    ground_truth: str
    query_type: str

    @classmethod
    def from_json(cls, record: Mapping[str, Any]) -> Query:
        """Build a Query from one decoded line of the queries file; a missing key raises KeyError."""
        return cls(
            id=record["id"],
            benchmark=record["benchmark"],
            query=record["query"],
            ground_truth=record["ground_truth"],
            query_type=record["query_type"],
        )


def load_queries(path: Path) -> list[Query]:
    """Read every query of a gzipped JSON-lines file, in file order.

    The file is read line by line: query texts may contain characters such as U+2028 that
    str.splitlines() would treat as line breaks.
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [Query.from_json(json.loads(line)) for line in f]


def read_csv_gz(path: Path) -> Iterator[dict[str, str]]:
    """Yield the rows of a gzipped CSV file as dicts keyed by the header, in file order."""
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)
