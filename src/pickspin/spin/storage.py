"""Weight loads that share the storage bandwidth: Eq. 5 with contention.

A cold start takes L_cold(m) = WeightSize(m) / StorageBandwidth + L_init(m) (Eq. 5). Transfers that
overlap share the bandwidth fluidly: while k transfers are in flight, each moves at gbps / k. When a
model's weights have arrived, its load still spends L_init(m), the part of the stated cold-start time
that is not the weight transfer. Without contention a load takes the model's stated cold_start_s.

The simulator uses SharedStorage.estimate as Spin's load estimator. estimate() deliberately advances
the transfer clock to `now` before it answers, which changes the remaining gigabytes of the transfers
in flight at the bit level; seeded simulation results depend on that side effect, so it must stay.
SharedStorage is not thread-safe: only the single-threaded simulator uses it.
"""

from pickspin.config import DEFAULT_SPIN, MODELS


def init_seconds(model: str, gbps: float = DEFAULT_SPIN.storage_gbps) -> float:
    """Return L_init(m): the part of the model's stated cold-start time that is not the weight transfer."""
    spec = MODELS[model]
    return max(0.0, spec.cold_start_s - spec.weight_gb / gbps)


class SharedStorage:
    """Weight transfers in flight on the shared storage volume, for the simulator.

    remaining maps each model whose weights are being transferred to the gigabytes it still has to
    move, in the order the transfers began; t is the time up to which they have been moved.
    """

    gbps: float
    remaining: dict[str, float]
    t: float

    def __init__(self, gbps: float = DEFAULT_SPIN.storage_gbps) -> None:
        self.gbps = gbps
        self.remaining = {}
        self.t = 0.0

    def begin(self, model: str, now: float) -> None:
        """Start transferring the model's weights at time now."""
        self._advance(now)
        self.remaining[model] = float(MODELS[model].weight_gb)

    def next_transfer_done(self) -> tuple[float, str] | None:
        """Return (time, model) of the next transfer to finish at the current sharing, or None.

        Of transfers with equally few gigabytes left, the one that began first finishes first.
        """
        if not self.remaining:
            return None
        m = min(self.remaining, key=self.remaining.__getitem__)
        return self.t + max(0.0, self.remaining[m]) * len(self.remaining) / self.gbps, m

    def transfer_done(self, model: str, now: float) -> None:
        """Remove the model's finished transfer at time now; the others speed up from now on."""
        self._advance(now)
        del self.remaining[model]

    def estimate(self, model: str, now: float) -> float:
        """Return the expected cold-start time of the model if it started loading now.

        The transfer is assumed to share the bandwidth with the transfers in flight for its whole
        duration, then to spend L_init. This first advances the transfers to now (see the module
        docstring): it is not a pure query.
        """
        self._advance(now)
        k = len(self.remaining) + (0 if model in self.remaining else 1)
        return MODELS[model].weight_gb * k / self.gbps + init_seconds(model, self.gbps)

    def _advance(self, now: float) -> None:
        """Move every transfer in flight forward to time now, at an equal share of the bandwidth each.

        The clock never moves backwards: a time earlier than t changes nothing.
        """
        if self.remaining and now > self.t:
            moved = (now - self.t) * self.gbps / len(self.remaining)
            for m in self.remaining:
                self.remaining[m] -= moved
        self.t = max(self.t, now)
