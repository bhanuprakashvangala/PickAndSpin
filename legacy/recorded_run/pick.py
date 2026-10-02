"""
PICK Component: Intelligent Model Selection

1. Complexity classification with keyword rules
2. Thompson Sampling with Hybrid Estimation (HTS)
3. Latency-Aware Scoring
4. Confidence-Based Escalation
"""

import random
import threading
from config import MODELS, TIERS, ROUTING


class ThompsonSampler:
    """
    Thompson Sampling with Hybrid Tier-Model Estimation (HTS)

    For each model m, maintains Beta(α_m, β_m) distribution.
    For each tier τ, maintains Beta(α_τ, β_τ) distribution.

    Selection score:
        S(m) = (1-λ)·μ_HTS + λ·L_norm + ε/√(n+1)

    Where:
        μ_HTS = (1-w)·μ_model + w·μ_tier  (hybrid blend)
        L_norm = normalized latency score (lower latency = higher score)
        ε/√(n+1) = exploration bonus
    """

    def __init__(self):
        self.lock = threading.Lock()

        # Model-level statistics
        self.model_stats = {
            m: {
                "alpha": ROUTING["alpha_prior"],
                "beta": ROUTING["beta_prior"],
                "count": 0,
                "latency_sum": 0.0,
            }
            for m in MODELS
        }

        # Tier-level statistics
        self.tier_stats = {
            t: {
                "alpha": ROUTING["alpha_prior"],
                "beta": ROUTING["beta_prior"],
            }
            for t in TIERS
        }

    def sample_beta(self, alpha, beta):
        """Sample from Beta distribution"""
        return random.betavariate(alpha, beta)

    def select_model(self, tier):
        """
        Select best model from tier using Thompson Sampling + HTS

        Returns: (model_name, score)
        """
        candidates = TIERS[tier]
        w = ROUTING["tier_weight"]
        λ = ROUTING["latency_weight"]
        ε = ROUTING["exploration_bonus"]

        with self.lock:
            # Sample tier-level estimate
            tier_sample = self.sample_beta(
                self.tier_stats[tier]["alpha"],
                self.tier_stats[tier]["beta"]
            )

            # Calculate max latency for normalization
            latencies = []
            for m in candidates:
                s = self.model_stats[m]
                if s["count"] > 0:
                    latencies.append(s["latency_sum"] / s["count"])
            max_latency = max(latencies) if latencies else 1.0

            best_model = None
            best_score = -1

            for model in candidates:
                stats = self.model_stats[model]

                # Sample model-level estimate
                model_sample = self.sample_beta(stats["alpha"], stats["beta"])

                # HTS: Blend model and tier estimates
                # μ_HTS = (1-w)·μ_model + w·μ_tier
                hts_sample = (1 - w) * model_sample + w * tier_sample

                # Latency score (lower latency = higher score)
                if stats["count"] > 0:
                    avg_latency = stats["latency_sum"] / stats["count"]
                    latency_score = 1.0 - (avg_latency / max_latency) if max_latency > 0 else 0.5
                else:
                    latency_score = 0.5  # Neutral for unexplored models

                # Exploration bonus (UCB-style)
                exploration = ε / (stats["count"] + 1) ** 0.5

                # Combined score: S(m) = (1-λ)·μ_HTS + λ·L_norm + exploration
                score = (1 - λ) * hts_sample + λ * latency_score + exploration

                if score > best_score:
                    best_score = score
                    best_model = model

            return best_model, best_score

    def update(self, model, tier, success, latency):
        """Update statistics after observing query outcome"""
        with self.lock:
            # Update model stats
            if success:
                self.model_stats[model]["alpha"] += 1
            else:
                self.model_stats[model]["beta"] += 1
            self.model_stats[model]["count"] += 1
            self.model_stats[model]["latency_sum"] += latency

            # Update tier stats
            if success:
                self.tier_stats[tier]["alpha"] += 1
            else:
                self.tier_stats[tier]["beta"] += 1

    def get_stats(self):
        """Return current statistics for reporting"""
        with self.lock:
            return {
                m: {
                    "count": s["count"],
                    "success_rate": s["alpha"] / (s["alpha"] + s["beta"]),
                    "avg_latency": s["latency_sum"] / s["count"] if s["count"] > 0 else 0,
                }
                for m, s in self.model_stats.items()
                if s["count"] > 0
            }


class Classifier:
    """
    Keyword-based complexity classifier (under 1 ms per query).

    Returns a tier and a confidence; queries without a keyword match get MEDIUM with
    confidence 0.6. No learned classifier is called: self.distilbert is a hook for one.
    """

    # Keywords for each tier
    SIMPLE_KEYWORDS = [
        "what is", "define", "who is", "when was", "where is",
        "true or false", "which of", "select the", "name the",
        "list the", "is it true", "yes or no"
    ]

    COMPLEX_KEYWORDS = [
        "analyze", "explain why", "compare and contrast", "prove",
        "derive", "implement", "design", "evaluate", "synthesize",
        "critique", "justify", "hypothesize", "formulate"
    ]

    def __init__(self):
        self.distilbert = None  # hook for a learned classifier; unused

    def classify_keywords(self, query):
        """
        Keyword-based classification with confidence
        Returns: (tier, confidence)
        """
        q_lower = query.lower()

        # Check for complex keywords
        complex_matches = sum(1 for k in self.COMPLEX_KEYWORDS if k in q_lower)
        if complex_matches >= 2:
            return "COMPLEX", 0.9
        if complex_matches == 1:
            return "COMPLEX", 0.7

        # Check for simple keywords
        simple_matches = sum(1 for k in self.SIMPLE_KEYWORDS if k in q_lower)
        if simple_matches >= 2:
            return "SIMPLE", 0.9
        if simple_matches == 1:
            return "SIMPLE", 0.7

        # Default to medium with lower confidence
        return "MEDIUM", 0.6

    def classify(self, query):
        """
        Keyword classification. Returns: (tier, confidence)
        """
        tier, confidence = self.classify_keywords(query)

        # If keyword confidence is high enough, use it
        if confidence >= 0.7:
            return tier, confidence

        # Low-confidence queries keep the keyword result (no learned fallback is called)
        return tier, confidence


class PickRouter:
    """
    Complete PICK component combining:
    - Keyword complexity classification
    - Thompson Sampling with HTS
    - Confidence-based Escalation
    """

    def __init__(self):
        self.classifier = Classifier()
        self.sampler = ThompsonSampler()

    def route(self, query):
        """
        Route query to optimal model

        Returns: {
            "tier": str,
            "model": str,
            "confidence": float,
            "score": float
        }
        """
        # Step 1: Classify query complexity
        tier, confidence = self.classifier.classify(query)

        # Step 2: Select model via Thompson Sampling
        model, score = self.sampler.select_model(tier)

        return {
            "tier": tier,
            "model": model,
            "confidence": confidence,
            "score": score,
        }

    def maybe_escalate(self, tier, response_confidence):
        """
        Check if escalation needed based on response confidence

        Returns: (should_escalate, next_tier, next_model)
        """
        δ = ROUTING["confidence_threshold"]

        if response_confidence >= δ:
            return False, None, None

        if tier == "COMPLEX":
            return False, None, None  # Already at highest tier

        # Escalate to next tier
        next_tier = "MEDIUM" if tier == "SIMPLE" else "COMPLEX"
        next_model, _ = self.sampler.select_model(next_tier)

        return True, next_tier, next_model

    def update(self, model, tier, success, latency):
        """Update Thompson Sampling statistics"""
        self.sampler.update(model, tier, success, latency)

    def get_stats(self):
        """Get current model statistics"""
        return self.sampler.get_stats()
