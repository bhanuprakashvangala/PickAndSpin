"""The paper configuration, and the string behaviour of the package's enums."""

import csv
import dataclasses
import enum
import io
import json

import pytest

from pickspin.config import (
    DEFAULT_ROUTING,
    DEFAULT_SPIN,
    MODELS,
    TIER_ORDER,
    TIERS,
    ModelSpec,
    RoutingParams,
    SpinParams,
    Tier,
)
from pickspin.pick import Stage
from pickspin.simulation.policies import Policy
from pickspin.spin import LatencySignal, ModelState

# MODELS, ROUTING and SPIN of src/pickspin/config.py at tag v1.1.0, copied verbatim.
OLD_MODELS = {
    "llama3.2_1B": {"label": "Llama-3.2-1B", "tier": "SIMPLE", "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
                    "weight_gb": 2, "cold_start_s": 32, "gpus": 1},
    "qwen2.5_1.5B": {"label": "Qwen2.5-1.5B", "tier": "SIMPLE", "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
                     "weight_gb": 3, "cold_start_s": 35, "gpus": 1},
    "gemma2_2B": {"label": "Gemma-2-2B", "tier": "SIMPLE", "hf_id": "google/gemma-2-2b-it",
                  "weight_gb": 4, "cold_start_s": 38, "gpus": 1},
    "llama3.2_3B": {"label": "Llama-3.2-3B", "tier": "SIMPLE", "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
                    "weight_gb": 6, "cold_start_s": 32, "gpus": 1},
    "qwen2.5_7B": {"label": "Qwen2.5-7B", "tier": "MEDIUM", "hf_id": "Qwen/Qwen2.5-7B-Instruct",
                   "weight_gb": 14, "cold_start_s": 48, "gpus": 1},
    "llama3.1_8B": {"label": "Llama-3.1-8B", "tier": "MEDIUM", "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
                    "weight_gb": 16, "cold_start_s": 42, "gpus": 1},
    "gemma2_9B": {"label": "Gemma-2-9B", "tier": "MEDIUM", "hf_id": "google/gemma-2-9b-it",
                  "weight_gb": 18, "cold_start_s": 48, "gpus": 1},
    "qwen2.5_14B": {"label": "Qwen2.5-14B", "tier": "COMPLEX", "hf_id": "Qwen/Qwen2.5-14B-Instruct",
                    "weight_gb": 28, "cold_start_s": 65, "gpus": 1},
    "gemma3_27B": {"label": "Gemma-3-27B", "tier": "COMPLEX", "hf_id": "google/gemma-3-27b-it",
                   "weight_gb": 54, "cold_start_s": 95, "gpus": 1},
}  # fmt: skip
OLD_ROUTING = {
    "alpha_prior": 1.0,
    "beta_prior": 1.0,
    "tier_weight": 0.3,
    "latency_weight": 0.3,
    "exploration_bonus": 0.1,
}
OLD_SPIN = {"cooldown_s": 300, "storage_gbps": 1.2}

PAPER_ORDER = (
    "llama3.2_1B",
    "qwen2.5_1.5B",
    "gemma2_2B",
    "llama3.2_3B",
    "qwen2.5_7B",
    "llama3.1_8B",
    "gemma2_9B",
    "qwen2.5_14B",
    "gemma3_27B",
)

ENUMS = (Tier, Stage, ModelState, LatencySignal, Policy)


def test_models_follow_paper_order_and_match_the_old_catalog() -> None:
    assert tuple(MODELS) == PAPER_ORDER == tuple(OLD_MODELS)
    assert [f.name for f in dataclasses.fields(ModelSpec)] == ["key", *OLD_MODELS["llama3.2_1B"]]
    for key, old in OLD_MODELS.items():
        spec = MODELS[key]
        assert spec.key == key
        assert {name: getattr(spec, name) for name in old} == old
        assert type(spec.tier) is Tier
        assert all(type(getattr(spec, name)) is int for name in ("weight_gb", "cold_start_s", "gpus"))


def test_tiers_list_their_models_in_catalog_order() -> None:
    assert TIER_ORDER == (Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX)
    assert dict(TIERS) == {
        Tier.SIMPLE: ("llama3.2_1B", "qwen2.5_1.5B", "gemma2_2B", "llama3.2_3B"),
        Tier.MEDIUM: ("qwen2.5_7B", "llama3.1_8B", "gemma2_9B"),
        Tier.COMPLEX: ("qwen2.5_14B", "gemma3_27B"),
    }
    assert tuple(TIERS) == TIER_ORDER
    old_tiers = {t: [m for m, c in OLD_MODELS.items() if c["tier"] == t] for t in ["SIMPLE", "MEDIUM", "COMPLEX"]}
    assert {t: list(models) for t, models in TIERS.items()} == old_tiers


def test_default_parameters_are_the_papers() -> None:
    assert RoutingParams(1.0, 1.0, 0.3, 0.3, 0.1) == DEFAULT_ROUTING
    assert SpinParams(300.0, 1.2) == DEFAULT_SPIN
    assert dataclasses.asdict(DEFAULT_ROUTING) == OLD_ROUTING
    assert dataclasses.asdict(DEFAULT_SPIN) == OLD_SPIN
    assert list(dataclasses.asdict(DEFAULT_ROUTING)) == list(OLD_ROUTING)
    assert list(dataclasses.asdict(DEFAULT_SPIN)) == list(OLD_SPIN)


@pytest.mark.parametrize(
    ("instance", "field"),
    [(MODELS["gemma3_27B"], "weight_gb"), (DEFAULT_ROUTING, "tier_weight"), (DEFAULT_SPIN, "cooldown_s")],
    ids=["ModelSpec", "RoutingParams", "SpinParams"],
)
def test_config_dataclasses_are_frozen(instance: object, field: str) -> None:
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(instance, field, 0)


def test_catalog_mappings_are_read_only() -> None:
    with pytest.raises(TypeError):
        MODELS["llama3_70B"] = MODELS["gemma3_27B"]  # type: ignore[index]
    with pytest.raises(TypeError):
        del MODELS["gemma3_27B"]  # type: ignore[attr-defined]
    with pytest.raises(TypeError):
        TIERS[Tier.SIMPLE] = ()  # type: ignore[index]
    assert len(MODELS) == 9


def test_enum_values_are_the_old_strings() -> None:
    assert [t.value for t in Tier] == ["SIMPLE", "MEDIUM", "COMPLEX"]
    assert [s.value for s in Stage] == ["keyword", "distilbert", "default"]
    assert [s.value for s in ModelState] == ["COLD", "LOADING", "WARM"]
    assert [s.value for s in LatencySignal] == ["spin", "observed", "inference"]
    assert [p.value for p in Policy] == ["pick-and-spin", "pick-and-spin-observed", "unaware", "static"]
    assert all(issubclass(cls, enum.StrEnum) for cls in ENUMS)


@pytest.mark.parametrize("member", [m for cls in ENUMS for m in cls], ids=lambda m: f"{type(m).__name__}.{m.name}")
def test_enum_members_behave_as_their_plain_string_values(member: enum.StrEnum) -> None:
    value = member.value
    assert type(value) is str
    assert str(member) == value
    assert f"{member}" == value
    assert format(member, "24s") == format(value, "24s")
    assert json.dumps(member) == json.dumps(value)
    assert json.dumps({member: [member]}) == json.dumps({value: [value]})
    rows = []
    for cell in (member, value):
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerow([cell])
        rows.append(buffer.getvalue())
    assert rows[0] == rows[1] == f"{value}\r\n"
    assert member == value
    assert hash(member) == hash(value)
    assert type(member)(value) is member
