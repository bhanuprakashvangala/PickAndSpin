"""The released traces have the properties that the parsing rules rely on.

Each workflow parses the traces in its own way, and the rules stay at their call sites:

- the simulator counts a flag as true when it is 'True', 'true' or '1', reads an empty latency_ms as
  0.0 s, keeps only the nine routing models, and lets the last row win when a (query id, model) pair
  repeats;
- the paper reproduction and the training labels count a judge label as correct only when it is '1'
  and keep all eleven models, and the reproduction converts every latency_ms to a number.

The rules agree on the released traces only because every flag is '0' or '1', no pair repeats and no
latency is empty. These tests pin those facts, and the shapes the workflows assume: every query run on
every model, unique query ids, a routed run that orders every query once, and a tier cache whose
keyword stage matches the keyword lists.
"""

import gzip
import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pickspin.config import MODELS
from pickspin.data import read_csv_gz
from pickspin.paper.constants import PAPER_MODELS, canonical_model
from pickspin.paths import Paths
from pickspin.pick.classifier import Stage, keyword_tier
from pickspin.simulation.inputs import read_tier_cache
from pickspin.training.labels import LARGE_MODELS, MEDIUM_MODELS, SMALL_MODELS

N_QUERIES = 31_019
QUERY_KEYS = ("id", "benchmark", "query", "ground_truth", "query_type")
# The static baseline ran the nine routing models and two larger ones that only the labels use.
EXTRA_MODELS = ("llama3_70B", "kimi_1T_MoE")


@dataclass(frozen=True)
class Traces:
    """The released traces as rows of strings, and the queries as decoded JSON objects, in file order."""

    static: list[dict[str, str]]
    judgments: list[dict[str, str]]
    routed: list[dict[str, str]]
    queries: list[dict[str, Any]]

    @property
    def query_ids(self) -> list[str]:
        """The query ids in file order."""
        return [q["id"] for q in self.queries]


@pytest.fixture(scope="module")
def traces(released_data: Path) -> Traces:
    """The released traces and queries, read once for the module."""
    paths = Paths(released_data)
    with gzip.open(paths.queries, "rt", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    return Traces(
        static=list(read_csv_gz(paths.traces / "static_baseline.csv.gz")),
        judgments=list(read_csv_gz(paths.traces / "judgments.csv.gz")),
        routed=list(read_csv_gz(paths.traces / "pick_spin_routed.csv.gz")),
        queries=queries,
    )


def repeated_pairs(rows: Iterable[dict[str, str]]) -> list[tuple[str, str]]:
    """The (query id, model) pairs that occur more than once."""
    return [pair for pair, n in Counter((r["id"], r["model"]) for r in rows).items() if n > 1]


def test_flags_are_only_0_and_1(traces):
    assert {r["success"] for r in traces.static} <= {"0", "1"}
    assert {r["is_correct"] for r in traces.judgments} <= {"0", "1"}


def test_no_query_model_pair_repeats(traces):
    assert not repeated_pairs(traces.static)
    assert not repeated_pairs(traces.judgments)


def test_every_static_baseline_latency_and_token_count_is_a_number(traces):
    assert not [(r["id"], r["model"]) for r in traces.static if not r["latency_ms"]]
    # The reproduction parses every row with float() and int().
    assert all(float(r["latency_ms"]) >= 0 for r in traces.static)
    assert all(int(r["completion_tokens"]) >= 0 for r in traces.static)


def test_static_baseline_runs_every_query_on_eleven_models(traces):
    eleven = set(MODELS) | set(EXTRA_MODELS)
    assert set(SMALL_MODELS + MEDIUM_MODELS + LARGE_MODELS) == eleven  # the label groups use all of them
    assert Counter(r["model"] for r in traces.static) == dict.fromkeys(eleven, N_QUERIES)
    assert {r["id"] for r in traces.static} == set(traces.query_ids)


def test_judge_labels_are_for_the_eleven_models_and_known_queries(traces):
    assert {r["model"] for r in traces.judgments} <= set(MODELS) | set(EXTRA_MODELS)
    assert {r["id"] for r in traces.judgments} <= set(traces.query_ids)


def test_queries_have_unique_ids_and_the_five_keys(traces):
    assert len(traces.queries) == N_QUERIES
    assert len(set(traces.query_ids)) == N_QUERIES
    assert {tuple(q) for q in traces.queries} == {QUERY_KEYS}
    assert all(isinstance(value, str) for q in traces.queries for value in q.values())


def test_routed_run_orders_every_query_once(traces):
    orders = [int(r["order"]) for r in traces.routed]
    assert sorted(orders) == list(range(N_QUERIES))
    assert len(traces.routed) == N_QUERIES
    assert {r["id"] for r in traces.routed} == set(traces.query_ids)
    # Scored by model slot: the 27B slot was recorded as gemma2_27B.
    assert {canonical_model(r["model"]) for r in traces.routed} <= set(PAPER_MODELS)
    assert all(float(r["total_latency"]) >= 0 for r in traces.routed)


def test_tier_cache_covers_every_query_in_file_order(traces, released_data):
    cache = Paths(released_data).tier_cache
    assert [r["id"] for r in read_csv_gz(cache)] == traces.query_ids
    tiers = read_tier_cache(cache)  # raises ValueError on an unknown tier or stage
    # Made with the trained DistilBERT, so no query fell back to the default tier.
    assert {stage for _, stage in tiers.values()} <= {Stage.KEYWORD, Stage.DISTILBERT}


def test_tier_cache_keyword_stage_matches_the_keyword_lists(traces, released_data):
    tiers = read_tier_cache(Paths(released_data).tier_cache)
    mismatches = []
    for q in traces.queries:
        tier, stage = tiers[q["id"]]
        expected = keyword_tier(q["query"])
        if (stage is Stage.KEYWORD and tier != expected) or (stage is not Stage.KEYWORD and expected is not None):
            mismatches.append((q["id"], tier, stage, expected))
    assert not mismatches
