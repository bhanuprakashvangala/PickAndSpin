"""Thompson sampling and the Eq. 4 score of Pick (Sec. IV, Eqs. 2-4).

Seeded runs depend on the exact order of select()'s random draws, so that order is tested as a
contract, together with the Eq. 4 arithmetic evaluated on the recorded draws.
"""

import math
import random
from collections.abc import Callable

import pytest

from pickspin.config import DEFAULT_ROUTING, MODELS, TIERS, RoutingParams, Tier
from pickspin.pick import Pick, ThompsonSampler
from pickspin.spin import Spin

Event = tuple[object, ...]


class RecordingRandom(random.Random):
    """A random.Random that logs the arguments of every Beta draw into a shared event list.

    The draws are the real ones; their results are kept in `draws`.
    """

    events: list[Event]
    draws: list[float]

    def betavariate(self, alpha: float, beta: float) -> float:
        self.events.append(("betavariate", alpha, beta))
        value = super().betavariate(alpha, beta)
        self.draws.append(value)
        return value


def recording_random(seed: int, events: list[Event]) -> RecordingRandom:
    """Return a RecordingRandom seeded with seed that logs into events."""
    rng = RecordingRandom(seed)
    rng.events = events
    rng.draws = []
    return rng


class ScriptedBeta(random.Random):
    """A random.Random whose Beta samples are taken, in order, from `samples`."""

    samples: list[float]

    def betavariate(self, alpha: float, beta: float) -> float:
        return self.samples.pop(0)


def scripted_beta(*samples: float) -> ScriptedBeta:
    """Return a ScriptedBeta that returns the given Beta samples in order."""
    rng = ScriptedBeta()
    rng.samples = list(samples)
    return rng


def test_eq4_score(fixed_beta: type[random.Random], serve: Callable[..., float]) -> None:
    spin = Spin(scale_to_zero=False, now=0.0)
    serve(spin, "qwen2.5_7B", 0.0, 0.0, 2.0)
    serve(spin, "llama3.1_8B", 0.0, 0.0, 8.0)
    pick = Pick(None, spin, "spin", rng=fixed_beta(), tiers={Tier.MEDIUM: ("qwen2.5_7B", "llama3.1_8B")})
    pick.sampler.n.update({"qwen2.5_7B": 3, "llama3.1_8B": 3})
    model, score = pick.sampler.select(Tier.MEDIUM, lambda m: spin.latency_estimate(m, 10.0))
    # S = 0.7 * 0.5 + 0.3 * (1 - 2/8) + 0.1 / sqrt(4)
    assert model == "qwen2.5_7B"
    assert score == pytest.approx(0.35 + 0.225 + 0.05)


def test_select_draws_in_the_documented_order() -> None:
    events: list[Event] = []
    rng = recording_random(2024, events)
    sampler = ThompsonSampler(rng=rng)
    outcomes = [
        ("qwen2.5_7B", True),
        ("qwen2.5_7B", False),
        ("llama3.1_8B", True),
        ("gemma2_9B", False),
        ("gemma2_9B", False),
        ("llama3.2_1B", True),  # a SIMPLE model: changes nothing in the MEDIUM tier
    ]
    for model, success in outcomes:
        sampler.update(model, MODELS[model].tier, success)
    assert events == []  # update() makes no random draws
    latencies = {"qwen2.5_7B": 2.5, "llama3.1_8B": None, "gemma2_9B": 4.0}

    def latency_of(model: str) -> float | None:
        events.append(("latency_of", model))
        return latencies[model]

    model, score = sampler.select(Tier.MEDIUM, latency_of)

    # First every candidate's latency, then the tier draw, then one draw per candidate, in TIERS order.
    assert events == [
        ("latency_of", "qwen2.5_7B"),
        ("latency_of", "llama3.1_8B"),
        ("latency_of", "gemma2_9B"),
        ("betavariate", 3.0, 4.0),  # tier MEDIUM: the prior plus 2 successes and 3 failures
        ("betavariate", 2.0, 2.0),  # qwen2.5_7B
        ("betavariate", 2.0, 1.0),  # llama3.1_8B
        ("betavariate", 1.0, 3.0),  # gemma2_9B
    ]
    assert sampler.tier_ab[Tier.MEDIUM] == [3.0, 4.0]
    assert all(type(arg) is float for event in events[3:] for arg in event[1:])

    # select() makes no other draws: a fresh generator making the same four draws ends in the same state.
    reference = random.Random(2024)
    for alpha, beta in [(3.0, 4.0), (2.0, 2.0), (2.0, 1.0), (1.0, 3.0)]:
        reference.betavariate(alpha, beta)
    assert rng.getstate() == reference.getstate()

    # The result is Eq. 4 on the recorded draws, computed with the same expressions, bit for bit.
    mu_tier, *mu = rng.draws
    w, lam, eps = 0.3, 0.3, 0.1
    norms = [1.0 - 2.5 / 4.0, 0.5, 1.0 - 4.0 / 4.0]
    counts = [2, 1, 2]
    scores = [
        (1 - lam) * ((1 - w) * mu_m + w * mu_tier) + lam * norm + eps / math.sqrt(n + 1)
        for mu_m, norm, n in zip(mu, norms, counts, strict=True)
    ]
    best = scores.index(max(scores))
    assert model == TIERS[Tier.MEDIUM][best]
    assert score == scores[best]


def test_select_weighs_the_terms_with_the_routing_parameters() -> None:
    params = RoutingParams(tier_weight=0.2, latency_weight=0.5, exploration_bonus=0.05)
    rng = scripted_beta(0.9, 0.3, 0.8)  # the tier's sample, then qwen2.5_7B's and llama3.1_8B's
    sampler = ThompsonSampler(tiers={Tier.MEDIUM: ("qwen2.5_7B", "llama3.1_8B")}, params=params, rng=rng)
    sampler.n["llama3.1_8B"] = 3
    model, score = sampler.select(Tier.MEDIUM, {"qwen2.5_7B": 1.0, "llama3.1_8B": 4.0}.__getitem__)
    # qwen2.5_7B:  0.5 * (0.8 * 0.3 + 0.2 * 0.9) + 0.5 * (1 - 1/4) + 0.05 / sqrt(1) = 0.635
    # llama3.1_8B: 0.5 * (0.8 * 0.8 + 0.2 * 0.9) + 0.5 * (1 - 4/4) + 0.05 / sqrt(4) = 0.435
    assert model == "qwen2.5_7B"
    assert score == pytest.approx(0.635)
    assert rng.samples == []


def test_equal_scores_go_to_the_first_candidate(fixed_beta: type[random.Random]) -> None:
    sampler = ThompsonSampler(rng=fixed_beta())
    for tier in Tier:
        assert sampler.select(tier, lambda m: None)[0] == TIERS[tier][0]
        assert sampler.select(tier, lambda m: 2.0)[0] == TIERS[tier][0]
    # A smaller exploration bonus for qwen2.5_7B leaves llama3.1_8B and gemma2_9B tied for the best score.
    sampler.n["qwen2.5_7B"] = 3
    model, score = sampler.select(Tier.MEDIUM, lambda m: None)
    assert model == "llama3.1_8B"
    assert score == pytest.approx(0.7 * 0.5 + 0.3 * 0.5 + 0.1)


def test_update_counts_outcomes_for_the_model_and_its_tier() -> None:
    sampler = ThompsonSampler(rng=random.Random(0))
    state = sampler.rng.getstate()
    sampler.update("gemma2_9B", Tier.MEDIUM, True)
    assert sampler.model_ab["gemma2_9B"] == [2.0, 1.0]
    assert sampler.tier_ab[Tier.MEDIUM] == [2.0, 1.0]
    assert sampler.n["gemma2_9B"] == 1
    sampler.update("gemma2_9B", Tier.MEDIUM, False)
    sampler.update("qwen2.5_7B", Tier.MEDIUM, False)
    assert sampler.model_ab["gemma2_9B"] == [2.0, 2.0]
    assert sampler.model_ab["qwen2.5_7B"] == [1.0, 2.0]
    assert sampler.tier_ab[Tier.MEDIUM] == [2.0, 3.0]
    assert sampler.n == {m: {"gemma2_9B": 2, "qwen2.5_7B": 1}.get(m, 0) for m in MODELS}
    assert all(sampler.model_ab[m] == [1.0, 1.0] for m in MODELS if m not in ("gemma2_9B", "qwen2.5_7B"))
    assert sampler.tier_ab[Tier.SIMPLE] == [1.0, 1.0]
    assert sampler.tier_ab[Tier.COMPLEX] == [1.0, 1.0]
    assert all(type(x) is float for ab in (*sampler.model_ab.values(), *sampler.tier_ab.values()) for x in ab)
    assert sampler.rng.getstate() == state


@pytest.mark.parametrize(
    "latencies",
    [(None, None, None), (0.0, 0.0, 0.0), (None, 0.0, -1.0)],
    ids=["none-known", "all-zero", "none-positive"],
)
def test_l_norm_is_one_half_without_a_positive_latency(
    fixed_beta: type[random.Random], latencies: tuple[float | None, ...]
) -> None:
    sampler = ThompsonSampler(rng=fixed_beta())
    latency = dict(zip(TIERS[Tier.MEDIUM], latencies, strict=True))
    model, score = sampler.select(Tier.MEDIUM, latency.__getitem__)
    # Every candidate scores 0.7 * 0.5 + 0.3 * 0.5 + 0.1 / sqrt(1), so the first one wins.
    assert model == "qwen2.5_7B"
    assert score == pytest.approx(0.6)


def test_a_model_without_a_latency_gets_one_half(fixed_beta: type[random.Random]) -> None:
    sampler = ThompsonSampler(rng=fixed_beta())
    latency = {"qwen2.5_7B": 8.0, "llama3.1_8B": None, "gemma2_9B": 6.0}
    model, score = sampler.select(Tier.MEDIUM, latency.__getitem__)
    # l_norm: qwen2.5_7B 1 - 8/8 = 0, llama3.1_8B 0.5 (no latency yet), gemma2_9B 1 - 6/8 = 0.25
    assert model == "llama3.1_8B"
    assert score == pytest.approx(0.7 * 0.5 + 0.3 * 0.5 + 0.1)


def test_stats_lists_the_models_with_outcomes_in_catalog_order() -> None:
    sampler = ThompsonSampler(rng=random.Random(0))
    assert sampler.stats() == {}
    sampler.update("qwen2.5_7B", Tier.MEDIUM, True)
    sampler.update("qwen2.5_7B", Tier.MEDIUM, False)
    sampler.update("gemma2_2B", Tier.SIMPLE, True)
    stats = sampler.stats()
    assert stats == {"gemma2_2B": {"n": 1, "success_rate": 2 / 3}, "qwen2.5_7B": {"n": 2, "success_rate": 0.5}}
    assert list(stats) == ["gemma2_2B", "qwen2.5_7B"]


def test_every_posterior_starts_at_the_prior() -> None:
    params = RoutingParams(alpha_prior=2.0, beta_prior=5.0)
    tiers = {Tier.MEDIUM: ("qwen2.5_7B", "gemma2_9B")}
    sampler = ThompsonSampler(models=iter(["qwen2.5_7B", "gemma2_9B"]), tiers=tiers, params=params)
    assert sampler.model_ab == {"qwen2.5_7B": [2.0, 5.0], "gemma2_9B": [2.0, 5.0]}
    assert sampler.tier_ab == {Tier.MEDIUM: [2.0, 5.0]}
    assert sampler.n == {"qwen2.5_7B": 0, "gemma2_9B": 0}
    assert sampler.tiers is tiers
    assert sampler.params is params
    assert isinstance(sampler.rng, random.Random)

    default = ThompsonSampler()
    assert default.tiers is TIERS
    assert default.params is DEFAULT_ROUTING
    assert list(default.model_ab) == list(default.n) == list(MODELS)
    assert list(default.tier_ab) == list(Tier)
    assert all(ab == [1.0, 1.0] for ab in (*default.model_ab.values(), *default.tier_ab.values()))


def test_a_tier_without_candidates_cannot_be_selected() -> None:
    sampler = ThompsonSampler(tiers={Tier.MEDIUM: ()}, rng=random.Random(0))
    with pytest.raises(ValueError, match="no model of tier MEDIUM"):
        sampler.select(Tier.MEDIUM, lambda m: None)
