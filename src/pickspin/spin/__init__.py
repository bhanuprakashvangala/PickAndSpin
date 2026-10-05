"""Spin: lifecycle-aware orchestration (Sec. V of the paper).

Each model is COLD (no pod, no GPU), LOADING (pod scheduled, weights being read into GPU memory) or
WARM (ready to serve):

    COLD -> LOADING   a query is routed to a cold model (a cold-start event)
    LOADING -> WARM   the weights are loaded; queries that waited are forwarded
    WARM -> COLD      the model has had no query in flight for T_cooldown seconds; the pod is
                      scaled to zero and its GPU released

A cold start takes L_cold(m) = WeightSize(m) / StorageBandwidth + L_init(m) (Eq. 5), and loads that
overlap share the storage bandwidth.

- pickspin.spin.lifecycle: the state machine with an explicit clock, the latency Pick scores on, and
  the GPU accounting (Spin), shared by the simulator and the live runner
- pickspin.spin.storage: Eq. 5 with bandwidth contention (SharedStorage, init_seconds), used by the
  simulator

This package re-exports both APIs, which use only the standard library.
"""

from .lifecycle import LatencySignal, LoadEstimator, ModelState, ModelUsage, Spin, SpinSummary
from .storage import SharedStorage, init_seconds

__all__ = [
    "LatencySignal",
    "LoadEstimator",
    "ModelState",
    "ModelUsage",
    "SharedStorage",
    "Spin",
    "SpinSummary",
    "init_seconds",
]
