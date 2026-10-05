"""Pick's hybrid classifier (Eq. 1): the keyword lists, their precedence, and what reaches stage 2."""

import hashlib
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from pickspin.config import TIER_ORDER, Tier
from pickspin.pick import distilbert
from pickspin.pick.classifier import KEYWORDS, HybridClassifier, Stage, keyword_tier

# sha256 of the keyword lists in src/pickspin/config.py at tag v1.1.0: one UTF-8 line per tier, in
# TIER_ORDER, of '<tier>' TAB '|'.join(phrases), joined by newlines.
KEYWORDS_SHA256 = "96d1777fee0774885a5918e5f1931c8c247e31316014cdc986a4f5a6f7b2a07f"


class RecordingStage2:
    """A stage-2 predictor that records each call and predicts the same tier name for every query."""

    def __init__(self, tier: str = "COMPLEX") -> None:
        self.tier = tier
        self.seen: list[list[str]] = []

    def predict(self, queries: Sequence[str]) -> list[str]:
        self.seen.append(list(queries))
        return [self.tier] * len(queries)


class FirstWordStage2(RecordingStage2):
    """Predicts the tier named by each query's first word, so every query can get a different tier."""

    def predict(self, queries: Sequence[str]) -> list[str]:
        super().predict(queries)
        return [q.split()[0].upper() for q in queries]


def test_keyword_lists_and_precedence() -> None:
    assert keyword_tier("Prove that x > 0. What is x?") == Tier.COMPLEX
    assert keyword_tier("Calculate how many apples are left") == Tier.MEDIUM
    assert keyword_tier("What is the capital of France?") == Tier.SIMPLE
    assert keyword_tier("Finish the story") is None


def test_distilbert_only_sees_unmatched_queries() -> None:
    stub = RecordingStage2()
    clf = HybridClassifier(stub)
    assert clf.classify("What is 2 + 2?") == (Tier.SIMPLE, Stage.KEYWORD)
    assert clf.classify("Finish the story") == (Tier.COMPLEX, Stage.DISTILBERT)
    assert clf.classify_many(["define entropy", "foo", "bar"]) == [
        (Tier.SIMPLE, Stage.KEYWORD),
        (Tier.COMPLEX, Stage.DISTILBERT),
        (Tier.COMPLEX, Stage.DISTILBERT),
    ]
    # classify sends its one query; classify_many sends exactly the unmatched queries, in one call.
    assert stub.seen == [["Finish the story"], ["foo", "bar"]]


def test_keyword_lists_are_the_published_ones() -> None:
    assert tuple(KEYWORDS) == TIER_ORDER
    assert all(type(phrases) is tuple for phrases in KEYWORDS.values())
    text = "\n".join(f"{tier}\t" + "|".join(KEYWORDS[tier]) for tier in TIER_ORDER)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == KEYWORDS_SHA256
    # Only the query is lowercased, so a phrase with a capital letter could never match.
    assert all(phrase == phrase.lower() for phrases in KEYWORDS.values() for phrase in phrases)
    with pytest.raises(TypeError):
        KEYWORDS[Tier.SIMPLE] = ()  # type: ignore[index]


@pytest.mark.parametrize(
    ("query", "tier"),
    [
        ("PROVE THAT 2 IS PRIME", Tier.COMPLEX),
        ("The function is undefined at 0", Tier.SIMPLE),
        ("Redesign the logo", Tier.COMPLEX),
        ("Solve x + 1 = 2. Is it true?", Tier.MEDIUM),
        ("What is the best design?", Tier.COMPLEX),
        ("Explain why we compute it", Tier.COMPLEX),
        ("Prove it, then calculate what is left", Tier.COMPLEX),
        ("what  is", None),
        ("", None),
    ],
    ids=[
        "case-insensitive",
        "inside-a-word",
        "inside-a-word-complex",
        "medium-before-simple",
        "complex-before-simple",
        "complex-before-medium",
        "all-three-lists",
        "phrases-match-exactly",
        "empty",
    ],
)
def test_keyword_tier_is_a_lowercase_substring_match(query: str, tier: Tier | None) -> None:
    assert keyword_tier(query) == tier


def test_without_stage2_unmatched_queries_get_the_fallback_tier() -> None:
    clf = HybridClassifier(None)
    assert clf.fallback_tier is Tier.MEDIUM
    assert clf.classify("Finish the story") == (Tier.MEDIUM, Stage.DEFAULT)
    assert clf.classify("What is 2 + 2?") == (Tier.SIMPLE, Stage.KEYWORD)
    assert clf.classify_many(["Finish the story", "Prove it", "foo"]) == [
        (Tier.MEDIUM, Stage.DEFAULT),
        (Tier.COMPLEX, Stage.KEYWORD),
        (Tier.MEDIUM, Stage.DEFAULT),
    ]
    simple = HybridClassifier(None, fallback_tier=Tier.SIMPLE)
    assert simple.classify("foo") == (Tier.SIMPLE, Stage.DEFAULT)
    assert simple.classify_many(["foo", "bar"]) == [(Tier.SIMPLE, Stage.DEFAULT)] * 2


def test_classify_many_keeps_the_input_order() -> None:
    stage2 = FirstWordStage2()
    queries = ["medium a", "Prove b", "complex c", "list the d", "simple e", "How much f"]
    assert HybridClassifier(stage2).classify_many(queries) == [
        (Tier.MEDIUM, Stage.DISTILBERT),
        (Tier.COMPLEX, Stage.KEYWORD),
        (Tier.COMPLEX, Stage.DISTILBERT),
        (Tier.SIMPLE, Stage.KEYWORD),
        (Tier.SIMPLE, Stage.DISTILBERT),
        (Tier.MEDIUM, Stage.KEYWORD),
    ]
    assert stage2.seen == [["medium a", "complex c", "simple e"]]


def test_classify_many_does_not_call_stage2_when_every_query_matches() -> None:
    stage2 = RecordingStage2()
    clf = HybridClassifier(stage2)
    assert clf.classify_many(["Prove it", "What is it?"]) == [
        (Tier.COMPLEX, Stage.KEYWORD),
        (Tier.SIMPLE, Stage.KEYWORD),
    ]
    assert clf.classify_many([]) == []
    assert stage2.seen == []


def test_results_are_tier_and_stage_members() -> None:
    clf = HybridClassifier(RecordingStage2("SIMPLE"))
    results = [
        clf.classify("foo"),
        clf.classify("Prove it"),
        *clf.classify_many(["bar", "define x"]),
        HybridClassifier(None).classify("baz"),
        HybridClassifier(None, fallback_tier="COMPLEX").classify("baz"),  # type: ignore[arg-type]
    ]
    assert [(type(tier), type(stage)) for tier, stage in results] == [(Tier, Stage)] * len(results)


def test_stage2_must_return_a_known_tier_for_every_query() -> None:
    with pytest.raises(ValueError, match="'LARGE' is not a valid Tier"):
        HybridClassifier(RecordingStage2("LARGE")).classify("foo")

    class SkipsAQuery(RecordingStage2):
        def predict(self, queries: Sequence[str]) -> list[str]:
            return super().predict(queries)[1:]

    with pytest.raises(ValueError, match="stage 2 must predict one tier per query, but gave 1 for 2"):
        HybridClassifier(SkipsAQuery()).classify_many(["foo", "bar"])


def test_from_pretrained_uses_distilbert_as_stage2(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    created = []

    class FakeDistilBertTier(RecordingStage2):
        def __init__(self, model_dir: Path, **options: object) -> None:
            super().__init__()
            created.append((model_dir, options))

    monkeypatch.setattr(distilbert, "DistilBertTier", FakeDistilBertTier)
    clf = HybridClassifier.from_pretrained(tmp_path, device="cpu", batch_size=8)
    assert isinstance(clf.stage2, FakeDistilBertTier)
    assert clf.fallback_tier is Tier.MEDIUM
    assert clf.classify("foo") == (Tier.COMPLEX, Stage.DISTILBERT)
    HybridClassifier.from_pretrained(tmp_path)
    assert created == [(tmp_path, {"device": "cpu", "batch_size": 8}), (tmp_path, {"device": None, "batch_size": 64})]


def test_importing_the_classifier_does_not_load_stage2() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = sys.modules['transformers'] = None  # importing either now fails\n"
        "import pickspin.pick.classifier\n"
        "assert 'pickspin.pick.distilbert' not in sys.modules, 'pickspin.pick.classifier imported stage 2'\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
