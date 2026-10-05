"""The fine-tuned DistilBERT reproduces the released tier cache, data/query_tiers.csv.gz.

The cache holds the hybrid classifier's (tier, stage) for every query, from one classify_many call over
all of them. These tests classify samples of it again with HybridClassifier.from_pretrained. Keyword
rows must come back exactly. DistilBERT rows may differ in a near tie, because a sample is batched
differently from the full run and the padding of a batch can flip an argmax, so at least 99% of them
must agree; every one must still come from the DistilBERT stage.

Slow, and skipped unless torch, transformers and models/distilbert-complexity-classifier are present
(the [classifier] extra and a trained or copied model).
"""

from pathlib import Path

import pytest

from pickspin.config import Tier
from pickspin.data import load_queries, read_csv_gz
from pickspin.paths import Paths
from pickspin.pick.classifier import HybridClassifier, Stage

pytestmark = pytest.mark.slow

KEYWORD_ROWS = 1_000
DISTILBERT_ROWS = 256
MIN_AGREEMENT = 0.99


@pytest.fixture(scope="module")
def classifier(released_data: Path) -> HybridClassifier:
    """The hybrid classifier with the fine-tuned DistilBERT, loaded once for the module."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    model_dir = Paths(released_data).classifier_model
    if not (model_dir / "config.json").is_file():
        pytest.skip(f"no trained DistilBERT in {model_dir}")
    return HybridClassifier.from_pretrained(model_dir)


@pytest.fixture(scope="module")
def cached(released_data: Path) -> list[tuple[str, Tier, Stage]]:
    """(query text, tier, stage) for every row of the tier cache, in file order."""
    paths = Paths(released_data)
    text = {q.id: q.query for q in load_queries(paths.queries)}
    return [(text[r["id"]], Tier(r["tier"]), Stage(r["stage"])) for r in read_csv_gz(paths.tier_cache)]


def first_rows(cached: list[tuple[str, Tier, Stage]], stage: Stage, n: int) -> list[tuple[str, Tier, Stage]]:
    """The first n rows of the cache that the given stage classified."""
    rows = [row for row in cached if row[2] is stage][:n]
    assert len(rows) == n, f"the tier cache has only {len(rows)} {stage} rows"
    return rows


def test_keyword_rows_are_reproduced_exactly(classifier, cached):
    rows = first_rows(cached, Stage.KEYWORD, KEYWORD_ROWS)
    assert classifier.classify_many([text for text, _, _ in rows]) == [(tier, stage) for _, tier, stage in rows]


def test_distilbert_rows_are_reproduced(classifier, cached):
    rows = first_rows(cached, Stage.DISTILBERT, DISTILBERT_ROWS)
    predicted = classifier.classify_many([text for text, _, _ in rows])
    assert [stage for _, stage in predicted] == [Stage.DISTILBERT] * len(rows)
    agree = sum(tier == cached_tier for (tier, _), (_, cached_tier, _) in zip(predicted, rows))
    assert agree >= MIN_AGREEMENT * len(rows), f"only {agree} of {len(rows)} tiers agree with the cache"
