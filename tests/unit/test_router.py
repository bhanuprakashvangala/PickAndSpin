"""The Pick router (Sec. IV): classify a query, then select a model of its tier.

select() scores the tier's candidates with the latency Spin reports under the chosen signal; route()
first asks the classifier for the tier.
"""

import dataclasses
import random
from collections.abc import Callable

import pytest

from pickspin.config import DEFAULT_ROUTING, MODELS, TIERS, RoutingParams, Tier
from pickspin.pick import HybridClassifier, Pick, RouteDecision, Stage
from pickspin.spin import LatencySignal, Spin


class StubClassifier:
    """Puts every query in the same tier and records the queries it was asked about."""

    def __init__(self, tier: Tier, stage: Stage) -> None:
        self.tier = tier
        self.stage = stage
        self.seen: list[str] = []

    def classify(self, query: str) -> tuple[Tier, Stage]:
        self.seen.append(query)
        return self.tier, self.stage


class RecordingSpin:
    """Stands in for Spin: answers latency_estimate from a table and records each call's arguments."""

    def __init__(self, latencies: dict[str, float | None] | None = None) -> None:
        self.latencies = latencies or {}
        self.calls: list[tuple[str, float, LatencySignal | str]] = []

    def latency_estimate(
        self, model: str, now: float, signal: LatencySignal | str = LatencySignal.SPIN
    ) -> float | None:
        self.calls.append((model, now, signal))
        return self.latencies.get(model)


def test_eq4_routes_around_a_cold_model(fixed_beta: type[random.Random], serve: Callable[..., float]) -> None:
    spin = Spin(scale_to_zero=True, now=0.0)
    serve(spin, "qwen2.5_7B", 0.0, 1.0, 3.0)
    serve(spin, "llama3.1_8B", 0.0, 1.0, 4.0)
    assert spin.stop("qwen2.5_7B", 1000.0)  # the faster 7B model is cold again, 8B is warm
    tiers = {Tier.MEDIUM: ("qwen2.5_7B", "llama3.1_8B")}
    aware = Pick(None, spin, "spin", rng=fixed_beta(), tiers=tiers)
    unaware = Pick(None, spin, "inference", rng=fixed_beta(), tiers=tiers)
    for p in (aware, unaware):
        p.sampler.n.update({"qwen2.5_7B": 1, "llama3.1_8B": 1})
    assert aware.select(Tier.MEDIUM, 1001.0).model == "llama3.1_8B"
    assert unaware.select(Tier.MEDIUM, 1001.0).model == "qwen2.5_7B"


@pytest.mark.parametrize("signal", list(LatencySignal))
def test_select_asks_spin_about_every_candidate_at_now(fixed_beta: type[random.Random], signal: LatencySignal) -> None:
    spin = RecordingSpin({"qwen2.5_7B": 8.0, "llama3.1_8B": 2.0})
    pick = Pick(None, spin, signal.value, rng=fixed_beta())
    decision = pick.select(Tier.MEDIUM, 12.5, Stage.KEYWORD)
    assert spin.calls == [(m, 12.5, signal) for m in TIERS[Tier.MEDIUM]]
    assert all(type(s) is LatencySignal for _, _, s in spin.calls)
    # l_norm: qwen2.5_7B 1 - 8/8 = 0, llama3.1_8B 1 - 2/8 = 0.75, gemma2_9B 0.5 (no latency yet)
    assert decision.tier is Tier.MEDIUM
    assert decision.stage is Stage.KEYWORD
    assert decision.model == "llama3.1_8B"
    assert decision.score == pytest.approx(0.7 * 0.5 + 0.3 * 0.75 + 0.1)
    assert pick.select(Tier.MEDIUM, 13.0).stage is None


def test_route_classifies_once_then_selects() -> None:
    classifier = StubClassifier(Tier.COMPLEX, Stage.DISTILBERT)
    spin = RecordingSpin({"qwen2.5_14B": 30.0, "gemma3_27B": 20.0})
    routed = Pick(classifier, spin, rng=random.Random(11))
    selected = Pick(None, spin, rng=random.Random(11))
    queries = ["Finish the story", "Tell me more", "Finish the story", "And then?"]
    for i, query in enumerate(queries):
        now = 10.0 * i
        decision = routed.route(query, now)
        assert classifier.seen == queries[: i + 1]
        assert decision == selected.select(classifier.tier, now, classifier.stage)
        for pick in (routed, selected):
            pick.update(decision.model, decision.tier, i % 2 == 0)
    assert routed.sampler.model_ab == selected.sampler.model_ab
    assert routed.sampler.rng.getstate() == selected.sampler.rng.getstate()


def test_route_uses_the_hybrid_classifier() -> None:
    pick = Pick(HybridClassifier(None), RecordingSpin(), rng=random.Random(0))
    keyword = pick.route("What is the capital of France?", 0.0)
    assert (keyword.tier, keyword.stage) == (Tier.SIMPLE, Stage.KEYWORD)
    assert keyword.model in TIERS[Tier.SIMPLE]
    fallback = pick.route("Finish the story", 1.0)
    assert (fallback.tier, fallback.stage) == (Tier.MEDIUM, Stage.DEFAULT)
    assert fallback.model in TIERS[Tier.MEDIUM]


def test_route_needs_a_classifier() -> None:
    pick = Pick(None, RecordingSpin())
    with pytest.raises(RuntimeError, match="no classifier"):
        pick.route("What is 2 + 2?", 0.0)


def test_unknown_latency_signals_are_rejected() -> None:
    with pytest.raises(ValueError, match="unknown latency signal 'bogus'"):
        Pick(None, Spin(), "bogus")


@pytest.mark.parametrize("signal", [*LatencySignal, *(s.value for s in LatencySignal)])
def test_the_latency_signal_is_kept_as_the_enum(signal: LatencySignal | str) -> None:
    assert Pick(None, RecordingSpin(), signal).signal is LatencySignal(signal)
    assert Pick(None, RecordingSpin()).signal is LatencySignal.SPIN


def test_update_records_the_outcome_in_the_sampler() -> None:
    pick = Pick(None, RecordingSpin(), rng=random.Random(0))
    pick.update("gemma3_27B", Tier.COMPLEX, False)
    pick.update("gemma3_27B", Tier.COMPLEX, True)
    assert pick.sampler.model_ab["gemma3_27B"] == [2.0, 2.0]
    assert pick.sampler.tier_ab[Tier.COMPLEX] == [2.0, 2.0]
    assert pick.sampler.n["gemma3_27B"] == 2


def test_the_sampler_gets_the_tiers_params_and_rng() -> None:
    rng = random.Random(5)
    tiers = {Tier.SIMPLE: ("gemma2_2B", "llama3.2_1B")}
    params = RoutingParams(alpha_prior=2.0, beta_prior=3.0)
    pick = Pick(None, RecordingSpin(), rng=rng, tiers=tiers, params=params)
    assert pick.sampler.rng is rng
    assert pick.sampler.tiers is tiers
    assert pick.sampler.params is params
    assert list(pick.sampler.model_ab) == list(MODELS)
    assert pick.sampler.tier_ab == {Tier.SIMPLE: [2.0, 3.0]}
    default = Pick(None, RecordingSpin())
    assert default.sampler.tiers is TIERS
    assert default.sampler.params is DEFAULT_ROUTING


def test_route_decisions_are_frozen_records() -> None:
    decision = RouteDecision(tier=Tier.SIMPLE, stage=None, model="llama3.2_1B", score=0.5)
    assert [f.name for f in dataclasses.fields(RouteDecision)] == ["tier", "stage", "model", "score"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        decision.model = "gemma2_2B"  # type: ignore[misc]


def test_prefer_warm_routes_to_a_model_that_is_up_and_starts_one_only_when_none_is() -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0)
    simple = list(TIERS[Tier.SIMPLE])
    warm = simple[1]
    spin.request(warm, 0.0)
    spin.loaded(warm, 1.0)
    spin.start(warm, 1.0)
    spin.finish(warm, 2.0, infer_s=1.0, total_s=2.0)

    # The paper's Pick still tries the cold models of the tier, whose latency is unknown.
    explore = Pick(None, spin, rng=random.Random(0))
    assert {explore.select(Tier.SIMPLE, 3.0).model for _ in range(50)} - {warm}

    # With prefer_warm every query of the tier goes to the model that is up ...
    pick = Pick(None, spin, rng=random.Random(0), prefer_warm=True)
    assert {pick.select(Tier.SIMPLE, 3.0).model for _ in range(50)} == {warm}
    # ... and a tier with no model up chooses among all of its models.
    assert pick.select(Tier.COMPLEX, 3.0).model in TIERS[Tier.COMPLEX]


def test_prefer_warm_opens_the_choice_again_when_a_load_is_overdue() -> None:
    spin = Spin(cooldown_s=300, scale_to_zero=True, now=0.0, load_estimate=lambda model, now: 100.0)
    stuck = TIERS[Tier.COMPLEX][0]
    spin.request(stuck, 0.0)  # loading, expected to be ready at 100 s
    pick = Pick(None, spin, rng=random.Random(0), prefer_warm=True)
    assert {pick.select(Tier.COMPLEX, 150.0).model for _ in range(20)} == {stuck}  # within twice the estimate
    assert {pick.select(Tier.COMPLEX, 250.0).model for _ in range(50)} == set(TIERS[Tier.COMPLEX])  # overdue
