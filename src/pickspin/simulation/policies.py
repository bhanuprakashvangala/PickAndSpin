"""The four policies compared in the simulation experiments.

    pick-and-spin           scale to zero; Pick scores Spin's latency, which adds the cold-start
                            penalty of a model that is not warm (default design)
    pick-and-spin-observed  scale to zero; Pick scores mean observed latency, cold-start waits included
    unaware                 scale to zero; Pick scores inference latency only
    static                  every model warm for the whole run, never scaled down

The declaration order is the command line's default order and the order of the rows in the CSV
outputs.
"""

import enum
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from pickspin.spin.lifecycle import LatencySignal


class Policy(enum.StrEnum):
    """A simulated deployment policy, named as on the command line."""

    PICK_AND_SPIN = "pick-and-spin"
    PICK_AND_SPIN_OBSERVED = "pick-and-spin-observed"
    UNAWARE = "unaware"
    STATIC = "static"


@dataclass(frozen=True, slots=True)
class PolicySpec:
    """How a policy runs: whether Spin scales idle models to zero, and the latency Pick scores on."""

    scale_to_zero: bool
    signal: LatencySignal
    description: str


POLICIES: Final[Mapping[Policy, PolicySpec]] = MappingProxyType(
    {
        Policy.PICK_AND_SPIN: PolicySpec(
            scale_to_zero=True,
            signal=LatencySignal.SPIN,
            description="scale to zero; Pick scores Spin's latency, which adds the cold-start penalty of a model "
            "that is not warm (default design)",
        ),
        Policy.PICK_AND_SPIN_OBSERVED: PolicySpec(
            scale_to_zero=True,
            signal=LatencySignal.OBSERVED,
            description="scale to zero; Pick scores mean observed latency, cold-start waits included",
        ),
        Policy.UNAWARE: PolicySpec(
            scale_to_zero=True,
            signal=LatencySignal.INFERENCE,
            description="scale to zero; Pick scores inference latency only",
        ),
        Policy.STATIC: PolicySpec(
            scale_to_zero=False,
            signal=LatencySignal.SPIN,
            description="every model warm for the whole run, never scaled down",
        ),
    }
)
