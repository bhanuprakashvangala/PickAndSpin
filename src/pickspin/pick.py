"""Pick: complexity-aware routing (Sec. IV).

A query is classified into a tier by the hybrid classifier (Eq. 1). Within the tier each
model m keeps a Beta(alpha_m, beta_m) posterior of its success rate (Eq. 2), blended with
the tier's posterior (Hybrid Tier-Model Estimation, Eq. 3):

    mu_HTS(m) = (1 - w) * mu_m + w * mu_tau,       mu_m ~ Beta(alpha_m, beta_m), mu_tau ~ Beta(alpha_tau, beta_tau)

and the model with the highest score is selected (Eq. 4):

    S(m) = (1 - lambda) * mu_HTS(m) + lambda * L_norm(m) + epsilon / sqrt(n_m + 1)
    L_norm(m) = 1 - L(m) / max_{m' in tier} L(m')

L(m) is the latency that Spin reports for m (spin.Spin.latency_estimate), which includes the
cold-start penalty of a model that is not warm. Models without a latency observation get
L_norm = 0.5.
"""

import math
import random
import threading

from config import MODELS, ROUTING, TIERS


class ThompsonSampler:
    def __init__(self, models=MODELS, tiers=TIERS, params=ROUTING, rng=None):
        self.tiers = tiers
        self.p = params
        self.rng = rng or random.Random()
        self.lock = threading.Lock()
        a, b = params["alpha_prior"], params["beta_prior"]
        self.model_ab = {m: [a, b] for m in models}
        self.tier_ab = {t: [a, b] for t in tiers}
        self.n = {m: 0 for m in models}

    def select(self, tier, latency_of):
        """Return (model, score). latency_of(m) gives Spin's latency for m, or None if unknown."""
        w, lam, eps = self.p["tier_weight"], self.p["latency_weight"], self.p["exploration_bonus"]
        candidates = self.tiers[tier]
        with self.lock:
            lat = {m: latency_of(m) for m in candidates}
            known = [v for v in lat.values() if v is not None]
            max_lat = max(known) if known else 0.0
            mu_tier = self.rng.betavariate(*self.tier_ab[tier])
            best, best_score = None, -math.inf
            for m in candidates:
                mu_hts = (1 - w) * self.rng.betavariate(*self.model_ab[m]) + w * mu_tier
                if lat[m] is None or max_lat <= 0:
                    l_norm = 0.5
                else:
                    l_norm = 1.0 - lat[m] / max_lat
                score = (1 - lam) * mu_hts + lam * l_norm + eps / math.sqrt(self.n[m] + 1)
                if score > best_score:
                    best, best_score = m, score
            return best, best_score

    def update(self, model, tier, success):
        """alpha <- alpha + 1[success], beta <- beta + 1[failure], for the model and its tier."""
        with self.lock:
            k = 0 if success else 1
            self.model_ab[model][k] += 1
            self.tier_ab[tier][k] += 1
            self.n[model] += 1

    def stats(self):
        with self.lock:
            return {m: {"n": self.n[m], "success_rate": a / (a + b)}
                    for m, (a, b) in self.model_ab.items() if self.n[m]}


class Pick:
    """route(query, now) -> {"tier", "stage", "model", "score"}; update() after each query.

    `spin` provides the lifecycle-aware latency (Spin.latency_estimate). `latency_signal`
    selects what Pick scores on:
      "spin"      inference latency plus the cold-start penalty of a model that is not warm
                  (the paper's design, default),
      "observed"  mean observed total latency, cold-start waits included,
      "inference" mean inference latency only (Pick ignores Spin's lifecycle state).
    """

    def __init__(self, classifier, spin, latency_signal="spin", rng=None, tiers=TIERS):
        if latency_signal not in ("spin", "observed", "inference"):
            raise ValueError(f"unknown latency signal {latency_signal!r}")
        self.classifier = classifier
        self.spin = spin
        self.latency_signal = latency_signal
        self.sampler = ThompsonSampler(tiers=tiers, rng=rng)

    def route(self, query, now, tier=None, stage=None):
        if tier is None:
            tier, stage = self.classifier.classify(query)
        model, score = self.sampler.select(
            tier, lambda m: self.spin.latency_estimate(m, now, self.latency_signal))
        return {"tier": tier, "stage": stage, "model": model, "score": score}

    def update(self, model, tier, success):
        self.sampler.update(model, tier, success)
