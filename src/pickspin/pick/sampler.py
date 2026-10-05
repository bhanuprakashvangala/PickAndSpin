"""Thompson sampling for Pick (Sec. IV, Eqs. 2-4).

Within a tier each model m keeps a Beta(alpha_m, beta_m) posterior of its success rate (Eq. 2),
blended with the tier's posterior (Hybrid Tier-Model Estimation, Eq. 3):

    mu_HTS(m) = (1 - w) * mu_m + w * mu_tau,   mu_m ~ Beta(alpha_m, beta_m), mu_tau ~ Beta(alpha_tau, beta_tau)

and the model with the highest score is selected (Eq. 4):

    S(m) = (1 - lambda) * mu_HTS(m) + lambda * L_norm(m) + epsilon / sqrt(n_m + 1)
    L_norm(m) = 1 - L(m) / max_{m' in tier} L(m')

Models without a latency observation get L_norm = 0.5, and so does every model when no candidate has a
positive latency. When scores are equal, the candidate listed first in the tier wins.

Seeded runs depend on the order of the random draws in select(), which is part of its contract: first
latency_of for every candidate, then one draw for the tier, then one draw per candidate, in the tier's
candidate order. The arithmetic is written exactly as above, with the same grouping and math.sqrt,
because the golden tests compare seeded runs with v1.1.0 bit for bit. One lock guards every method.
"""

import math
import random
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence

from pickspin.config import DEFAULT_ROUTING, MODELS, TIERS, RoutingParams, Tier


class ThompsonSampler:
    """Beta posteriors per model and per tier, and the Eq. 4 selection.

    model_ab[m] and tier_ab[t] hold the [alpha, beta] of each posterior: the prior plus one for every
    recorded success or failure. n[m] counts the outcomes recorded for model m (n_m in Eq. 4). They are
    public so that tests can set them.
    """

    tiers: Mapping[Tier, Sequence[str]]
    params: RoutingParams
    rng: random.Random
    model_ab: dict[str, list[float]]
    tier_ab: dict[Tier, list[float]]
    n: dict[str, int]

    def __init__(
        self,
        models: Iterable[str] = MODELS,
        tiers: Mapping[Tier, Sequence[str]] = TIERS,
        params: RoutingParams = DEFAULT_ROUTING,
        rng: random.Random | None = None,
    ) -> None:
        self.tiers = tiers
        self.params = params
        self.rng = random.Random() if rng is None else rng
        self._lock = threading.Lock()
        keys = list(models)
        a, b = params.alpha_prior, params.beta_prior
        self.model_ab = {m: [a, b] for m in keys}
        self.tier_ab = {t: [a, b] for t in tiers}
        self.n = {m: 0 for m in keys}

    def select(self, tier: Tier, latency_of: Callable[[str], float | None]) -> tuple[str, float]:
        """Return (model, score) for a query of the tier; latency_of(m) is m's latency, or None if unknown.

        latency_of is called once for every candidate, before any random draw and with the lock held.
        Under Pick it asks Spin, so the locks are always taken in the order sampler, then Spin. Raises
        ValueError if the tier has no candidate that can be selected.
        """
        w, lam, eps = self.params.tier_weight, self.params.latency_weight, self.params.exploration_bonus
        candidates = self.tiers[tier]
        with self._lock:
            lat = {m: latency_of(m) for m in candidates}
            known = [v for v in lat.values() if v is not None]
            max_lat = max(known) if known else 0.0
            mu_tier = self.rng.betavariate(*self.tier_ab[tier])
            best: str | None = None
            best_score = -math.inf
            for m in candidates:
                mu_hts = (1 - w) * self.rng.betavariate(*self.model_ab[m]) + w * mu_tier
                latency = lat[m]
                l_norm = 0.5 if latency is None or max_lat <= 0 else 1.0 - latency / max_lat
                score = (1 - lam) * mu_hts + lam * l_norm + eps / math.sqrt(self.n[m] + 1)
                if score > best_score:
                    best, best_score = m, score
        if best is None:
            raise ValueError(f"no model of tier {tier} can be selected from {list(candidates)}")
        return best, best_score

    def update(self, model: str, tier: Tier, success: bool) -> None:
        """Count one outcome: alpha grows on success and beta on failure, for the model and its tier."""
        with self._lock:
            k = 0 if success else 1
            self.model_ab[model][k] += 1
            self.tier_ab[tier][k] += 1
            self.n[model] += 1

    def stats(self) -> dict[str, dict[str, float]]:
        """Return the number of queries and the posterior mean success rate of every model used so far."""
        with self._lock:
            return {
                m: {"n": self.n[m], "success_rate": a / (a + b)} for m, (a, b) in self.model_ab.items() if self.n[m]
            }
