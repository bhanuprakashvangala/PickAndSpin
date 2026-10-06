"""The Pick router (Sec. IV): classify a query, then select a model of its tier.

route() classifies a query with the hybrid classifier and then selects a model; select() serves a tier
that is already known, such as the simulator's cached tiers. Selection scores every candidate with the
latency Spin reports for it, under one of three latency signals:

- spin: inference latency plus the cold-start penalty of a model that is not warm (the paper's design)
- observed: mean observed total latency, cold-start waits included
- inference: mean inference latency only, ignoring Spin's lifecycle state
"""

import random
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from pickspin.config import DEFAULT_ROUTING, TIERS, RoutingParams, Tier
from pickspin.pick.classifier import HybridClassifier, Stage
from pickspin.pick.sampler import ThompsonSampler
from pickspin.spin.lifecycle import LatencySignal, ModelState, Spin


@dataclass(frozen=True, slots=True)
class RouteDecision:
    """Where a query goes: its tier, the classifier stage that set it, the model and its Eq. 4 score."""

    tier: Tier
    stage: Stage | None
    model: str
    score: float


class Pick:
    """Routes queries with route() or select(); update() records the outcome of each query.

    signal is the latency Pick scores on, given as a LatencySignal or its value ('spin', 'observed' or
    'inference'); any other value raises ValueError. classifier may be None when every query's tier is
    known in advance, as in the simulator, which calls select() directly. The sampler's lock guards the
    posteriors, so the live runner's worker threads can share one Pick.

    With prefer_warm, a query goes to one of its tier's models that is WARM or LOADING whenever there
    is one, so a cold start happens only when the tier has no model up; the gateway uses this. Without
    it (the paper's method), every model of the tier is a candidate.
    """

    classifier: HybridClassifier | None
    spin: Spin
    signal: LatencySignal
    sampler: ThompsonSampler
    prefer_warm: bool

    def __init__(
        self,
        classifier: HybridClassifier | None,
        spin: Spin,
        signal: LatencySignal | str = LatencySignal.SPIN,
        *,
        rng: random.Random | None = None,
        tiers: Mapping[Tier, Sequence[str]] = TIERS,
        params: RoutingParams = DEFAULT_ROUTING,
        models: Iterable[str] | None = None,
        prefer_warm: bool = False,
    ) -> None:
        try:
            self.signal = LatencySignal(signal)
        except ValueError:
            raise ValueError(f"unknown latency signal {signal!r}") from None
        self.classifier = classifier
        self.spin = spin
        self.prefer_warm = prefer_warm
        if models is None:
            self.sampler = ThompsonSampler(tiers=tiers, params=params, rng=rng)
        else:
            self.sampler = ThompsonSampler(models, tiers=tiers, params=params, rng=rng)

    def route(self, query: str, now: float) -> RouteDecision:
        """Classify the query, then select a model for its tier at time now.

        Raises RuntimeError if this Pick was created without a classifier.
        """
        if self.classifier is None:
            raise RuntimeError("this Pick has no classifier: call select() with the query's tier instead")
        tier, stage = self.classifier.classify(query)
        return self.select(tier, now, stage)

    def select(self, tier: Tier, now: float, stage: Stage | None = None) -> RouteDecision:
        """Select a model for a query whose tier is already known.

        Every candidate of the tier is scored with spin.latency_estimate(model, now, signal); with
        prefer_warm the candidates are the tier's models that are up, if any. The stage is not used for
        the choice; it is only recorded in the decision.
        """
        candidates = None
        if self.prefer_warm:
            up = [m for m in self.sampler.tiers[tier] if self.spin.status(m) is not ModelState.COLD]
            candidates = up or None
        model, score = self.sampler.select(tier, lambda m: self.spin.latency_estimate(m, now, self.signal), candidates)
        return RouteDecision(tier=tier, stage=stage, model=model, score=score)

    def update(self, model: str, tier: Tier, success: bool) -> None:
        """Record whether a query of the tier succeeded on the model."""
        self.sampler.update(model, tier, success)
