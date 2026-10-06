"""Pick and Spin configuration (Sections IV-VI of the paper).

The paper's fixed configuration as immutable typed data: the query tiers, the nine-model pool in
paper order (Sec. VI-A), the models of each tier, the routing parameters of Eqs. 2-4, and Spin's
cooldown and the storage bandwidth of the cold-start model (Eq. 5).

This module holds no paths, environment reads, I/O or keyword lists. Paths are defined in
pickspin.paths, the keyword lists in pickspin.pick.classifier, and endpoints are read from a JSON
file (see deploy/endpoints.example.json).

The insertion order of MODELS is part of the behaviour. It sets the order of the candidates within
a tier, and with it the order of Pick's random draws and how ties are broken. It also sets the order
of Spin's per-model state, which decides the order of scale-downs and of the GPU-hour sums, and the
order of the rows in per_model.csv.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final


class Tier(enum.StrEnum):
    """Complexity tier of a query (Sec. IV-A). Members are declared in tier order, smallest first."""

    SIMPLE = "SIMPLE"
    MEDIUM = "MEDIUM"
    COMPLEX = "COMPLEX"


TIER_ORDER: Final[tuple[Tier, ...]] = tuple(Tier)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """One model of the pool: its key, display label, tier, Hugging Face id and cold-start figures."""

    key: str
    label: str
    tier: Tier
    hf_id: str
    weight_gb: int
    cold_start_s: int
    gpus: int = 1


# Nine models from three families in three tiers (Sec. VI-A). weight_gb is the size of the
# bf16 weights that a cold start loads from storage (Eq. 5); cold_start_s is the cold-start
# time stated for each model (Sec. V-B and Table II: 32-38 s small, 42-48 s medium,
# 65-95 s large).
_SPECS: Final[tuple[ModelSpec, ...]] = (
    # key, label, tier, hf_id, weight_gb, cold_start_s, gpus
    ModelSpec("llama3.2_1B", "Llama-3.2-1B", Tier.SIMPLE, "meta-llama/Llama-3.2-1B-Instruct", 2, 32, 1),
    ModelSpec("qwen2.5_1.5B", "Qwen2.5-1.5B", Tier.SIMPLE, "Qwen/Qwen2.5-1.5B-Instruct", 3, 35, 1),
    ModelSpec("gemma2_2B", "Gemma-2-2B", Tier.SIMPLE, "google/gemma-2-2b-it", 4, 38, 1),
    ModelSpec("llama3.2_3B", "Llama-3.2-3B", Tier.SIMPLE, "meta-llama/Llama-3.2-3B-Instruct", 6, 32, 1),
    ModelSpec("qwen2.5_7B", "Qwen2.5-7B", Tier.MEDIUM, "Qwen/Qwen2.5-7B-Instruct", 14, 48, 1),
    ModelSpec("llama3.1_8B", "Llama-3.1-8B", Tier.MEDIUM, "meta-llama/Llama-3.1-8B-Instruct", 16, 42, 1),
    ModelSpec("gemma2_9B", "Gemma-2-9B", Tier.MEDIUM, "google/gemma-2-9b-it", 18, 48, 1),
    ModelSpec("qwen2.5_14B", "Qwen2.5-14B", Tier.COMPLEX, "Qwen/Qwen2.5-14B-Instruct", 28, 65, 1),
    ModelSpec("gemma3_27B", "Gemma-3-27B", Tier.COMPLEX, "google/gemma-3-27b-it", 54, 95, 1),
)

# The model pool by key, read-only and in paper order.
MODELS: Final[Mapping[str, ModelSpec]] = MappingProxyType({s.key: s for s in _SPECS})


def tiers_of(catalog: Mapping[str, ModelSpec]) -> Mapping[Tier, tuple[str, ...]]:
    """Return the keys of each tier's models in catalog order: the candidates Pick chooses between."""
    return MappingProxyType({t: tuple(k for k, s in catalog.items() if s.tier is t) for t in Tier})


# The paper's tiers, in MODELS order.
TIERS: Final[Mapping[Tier, tuple[str, ...]]] = tiers_of(MODELS)


@dataclass(frozen=True, slots=True)
class RoutingParams:
    """Pick's routing parameters (Eqs. 2-4)."""

    alpha_prior: float = 1.0  # Beta(1, 1) prior for every model and tier (Eq. 2)
    beta_prior: float = 1.0
    tier_weight: float = 0.3  # w in Eq. 3
    latency_weight: float = 0.3  # lambda in Eq. 4
    exploration_bonus: float = 0.1  # epsilon in Eq. 4


@dataclass(frozen=True, slots=True)
class SpinParams:
    """Spin's lifecycle parameters."""

    cooldown_s: float = 300.0  # T_cooldown: a WARM model idle this long is scaled to zero
    storage_gbps: float = 1.2  # aggregate bandwidth of the shared weight volume (Eq. 5)


DEFAULT_ROUTING: Final[RoutingParams] = RoutingParams()
DEFAULT_SPIN: Final[SpinParams] = SpinParams()
