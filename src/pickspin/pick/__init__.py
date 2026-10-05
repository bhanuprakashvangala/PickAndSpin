"""Pick: complexity-aware routing (Sec. IV of the paper).

The hybrid classifier puts each query into a tier: keyword lists first, then a stage-2 predictor such
as the fine-tuned DistilBERT (Eq. 1). Within the tier, Thompson sampling draws each model's success
rate from a Beta posterior blended with the tier's posterior (Eqs. 2-3), and Pick selects the model
with the highest score (Eq. 4), which uses the latency Spin reports.

This package re-exports the pure-Python API. It never imports pickspin.pick.distilbert, so importing
it never loads torch.
"""

from .classifier import KEYWORDS, HybridClassifier, Stage, TierPredictor, keyword_tier
from .router import Pick, RouteDecision
from .sampler import ThompsonSampler

__all__ = [
    "KEYWORDS",
    "HybridClassifier",
    "Pick",
    "RouteDecision",
    "Stage",
    "ThompsonSampler",
    "TierPredictor",
    "keyword_tier",
]
