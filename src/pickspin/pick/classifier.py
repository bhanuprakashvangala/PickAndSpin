"""The hybrid complexity classifier of Pick (Sec. IV-A, Eq. 1).

Stage 1 matches three keyword lists as lowercase substrings, checking the COMPLEX list first, then
MEDIUM, then SIMPLE. Queries that match no list go to stage 2, a predictor such as the fine-tuned
DistilBERT (pickspin.pick.distilbert), which returns argmax_tau P(tau | q; theta). Without a stage-2
predictor those queries get a fallback tier instead, which is only meant for running without the
trained model.

Importing this module never loads torch: HybridClassifier.from_pretrained imports the DistilBERT
stage only when it is called.
"""

from __future__ import annotations

import enum
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final, Protocol

from pickspin.config import TIER_ORDER, Tier

log = logging.getLogger(__name__)


class Stage(enum.StrEnum):
    """The classifier stage that assigned a query's tier."""

    KEYWORD = "keyword"
    DISTILBERT = "distilbert"
    DEFAULT = "default"


# Stage 1 of the hybrid classifier (Sec. IV-A): three keyword lists matched as lowercase
# substrings, COMPLEX first, then MEDIUM, then SIMPLE. Queries with no match go to stage 2.
KEYWORDS: Final[Mapping[Tier, tuple[str, ...]]] = MappingProxyType(
    {
        Tier.SIMPLE: (
            "what is",
            "define",
            "who is",
            "when was",
            "where is",
            "true or false",
            "which of",
            "select the",
            "name the",
            "list the",
            "is it true",
            "yes or no",
        ),
        Tier.MEDIUM: (
            "calculate",
            "how many",
            "how much",
            "solve",
            "compute",
            "write a function",
            "write a python function",
            "find the value",
            "what will be",
            "complete the",
        ),
        Tier.COMPLEX: (
            "prove",
            "derive",
            "implement step by step",
            "analyze",
            "explain why",
            "compare and contrast",
            "design",
            "evaluate",
            "synthesize",
            "critique",
            "justify",
            "hypothesize",
            "formulate",
        ),
    }
)


def keyword_tier(query: str) -> Tier | None:
    """Return the first tier, in the order COMPLEX, MEDIUM, SIMPLE, whose list matches the query, or None.

    A list matches when one of its phrases occurs anywhere in the lowercased query, even inside a
    longer word ('undefined' contains 'define').
    """
    q = query.lower()
    for tier in reversed(TIER_ORDER):
        if any(k in q for k in KEYWORDS[tier]):
            return tier
    return None


class TierPredictor(Protocol):
    """Stage 2 of the hybrid classifier: predicts the tier name of each query.

    predict returns one tier name (a Tier, or its value such as 'MEDIUM') per query, in input order.
    """

    def predict(self, queries: Sequence[str]) -> Sequence[str]: ...


class HybridClassifier:
    """Assigns a (tier, stage) to queries: the keyword lists first, then the stage-2 predictor.

    stage2 classifies the queries that no keyword list matches. With stage2=None those queries get
    fallback_tier with Stage.DEFAULT instead, which is only meant for running without the trained
    model. from_pretrained() builds the classifier with the fine-tuned DistilBERT as stage 2.
    """

    stage2: TierPredictor | None
    fallback_tier: Tier

    def __init__(self, stage2: TierPredictor | None, *, fallback_tier: Tier = Tier.MEDIUM) -> None:
        self.stage2 = stage2
        self.fallback_tier = Tier(fallback_tier)

    @classmethod
    def from_pretrained(cls, model_dir: Path, *, device: str | None = None, batch_size: int = 64) -> HybridClassifier:
        """Return a classifier whose stage 2 is the fine-tuned DistilBERT saved in model_dir.

        This needs the [classifier] extra (torch and transformers). The device defaults to CUDA when it
        is available and to the CPU otherwise. Raises MissingDependencyError without the extra, and
        DataNotFoundError when model_dir holds no model.
        """
        # Imported here, not at the top, so that importing pickspin.pick never loads torch.
        from pickspin.pick.distilbert import DistilBertTier

        return cls(DistilBertTier(model_dir, device=device, batch_size=batch_size))

    def classify(self, query: str) -> tuple[Tier, Stage]:
        """Return the tier of one query and the stage that assigned it.

        A query that no keyword list matches goes to stage 2 on its own, as a batch of one.
        """
        tier = keyword_tier(query)
        if tier is not None:
            return tier, Stage.KEYWORD
        if self.stage2 is None:
            return self.fallback_tier, Stage.DEFAULT
        return Tier(self.stage2.predict([query])[0]), Stage.DISTILBERT

    def classify_many(self, queries: Sequence[str]) -> list[tuple[Tier, Stage]]:
        """Classify many queries with one stage-2 call over the queries that no keyword list matches.

        Stage 2 receives all the unmatched queries in a single predict() call, in their original
        relative order, and is not called at all when every query matches a keyword list. This is the
        batched form for offline use. Its stage-2 tier for a query can differ from classify()'s,
        because the batch a query lands in changes its padding, which can flip a near tie.
        """
        keyword = [keyword_tier(q) for q in queries]
        todo = [i for i, tier in enumerate(keyword) if tier is None]
        if todo and self.stage2 is not None:
            log.debug("Stage 2 classifies the %d of %d queries that match no keyword list", len(todo), len(queries))
            predicted = self.stage2.predict([queries[i] for i in todo])
            if len(predicted) != len(todo):
                raise ValueError(f"stage 2 must predict one tier per query, but gave {len(predicted)} for {len(todo)}")
            rest = {i: (Tier(t), Stage.DISTILBERT) for i, t in zip(todo, predicted)}
        else:
            rest = {i: (self.fallback_tier, Stage.DEFAULT) for i in todo}
        return [(tier, Stage.KEYWORD) if tier is not None else rest[i] for i, tier in enumerate(keyword)]
