"""Spin's COLD/LOADING/WARM state machine with an explicit clock (Sec. V).

Each model is COLD (no pod, no GPU), LOADING (pod scheduled, weights being read into GPU memory) or
WARM (ready to serve):

    COLD -> LOADING   a query is routed to a cold model (a cold-start event)
    LOADING -> WARM   the weights are loaded; queries that waited are forwarded
    WARM -> COLD      the model has had nothing in flight for T_cooldown seconds; it is scaled to
                      zero and its GPUs are released
    LOADING/WARM -> COLD   the model's server is lost (its load failed, or it died); the live gateway
                      uses this to start the model afresh on the next query

Spin keeps the lifecycle state of every model, reports the latency that Pick scores on, and accounts
for GPU time: a model holds its GPUs from the start of a load until it is scaled to zero. Time is
always passed in as `now`, so the simulator (on a simulated clock) and the live runner (on
time.monotonic) drive the same object.

latency_estimate(model, now, signal) is the latency L(m) of Eq. 4. Under the 'spin' signal it adds
the cold-start penalty of a model that is not warm: the expected load time of a COLD model, or the
time left on the load of a LOADING one.

Every result must stay bit-identical to the v1.1.0 code, so the arithmetic is kept expression for
expression: the accumulators are plain += in call order, and the totals of summary() use the built-in
sum() over the models in order. Spin consults its load estimator only where it always did (a COLD
request, and the 'spin' latency of a COLD model that has served a query); the simulator's estimator
advances the shared storage as a side effect, so neither call may be dropped, cached or reordered.
"""

import dataclasses
import enum
import threading
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from pickspin.config import DEFAULT_SPIN, MODELS, ModelSpec


class ModelState(enum.StrEnum):
    """Lifecycle state of a model."""

    COLD = "COLD"
    LOADING = "LOADING"
    WARM = "WARM"


class LatencySignal(enum.StrEnum):
    """The latency Pick scores on: Spin's lifecycle-aware estimate, observed latency, or inference latency."""

    SPIN = "spin"
    OBSERVED = "observed"
    INFERENCE = "inference"


# load_estimate(model, now) -> the expected cold-start time of the model in seconds.
LoadEstimator: TypeAlias = Callable[[str, float], float]


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """GPU time and cold starts of one model."""

    gpu_hours: float  # GPU-hours held, from the start of each load until the model is scaled to zero
    busy_gpu_hours: float  # the part of them with at least one query executing
    loading_gpu_hours: float  # the part of them spent loading the weights
    cold_starts: int


@dataclass(frozen=True, slots=True)
class SpinSummary:
    """GPU-hours, utilization and cold starts of a run, in total and per model.

    The fields are declared in the order of the keys of the old summary dict, which the live runner
    writes as JSON.
    """

    gpu_hours: float
    busy_gpu_hours: float
    gpu_utilization: float  # busy_gpu_hours / gpu_hours, or 0.0 without any GPU time
    cold_starts: int
    cold_start_rate: float  # cold starts per routed query, or 0.0 without any query
    per_model: dict[str, ModelUsage]  # in the order of Spin.models

    def to_dict(self) -> dict[str, Any]:
        """Return the summary as a new dict with the old summary's keys, in the same order.

        per_model becomes a dict of dicts. The result is a copy, so callers may add keys to it.
        """
        return dataclasses.asdict(self)


def _stated_cold_start(model: str, now: float) -> float:
    """The default load estimate: the cold-start time stated for the model in pickspin.config.MODELS."""
    return float(MODELS[model].cold_start_s)


class _State:
    """Lifecycle state and accounting of one model. Spin's lock guards every change."""

    __slots__ = (
        "alloc_s",
        "alloc_since",
        "busy_s",
        "busy_since",
        "cold_starts",
        "idle_since",
        "infer_sum",
        "inflight",
        "load_s",
        "load_since",
        "n",
        "pending",
        "ready_eta",
        "status",
        "total_sum",
    )

    def __init__(self, status: ModelState, now: float) -> None:
        self.status = status
        self.pending = 0  # routed to this model, not started yet (waiting for a load or a slot)
        self.inflight = 0  # executing on this model
        self.idle_since = now  # when the last query finished or the load completed (or the run began)
        self.ready_eta: float | None = None  # when the current load is expected to finish
        self.alloc_since: float | None = now if status is ModelState.WARM else None  # GPUs held since
        self.busy_since: float | None = None  # executing since
        self.alloc_s = self.busy_s = self.load_s = 0.0  # closed intervals, in seconds
        self.load_since: float | None = None  # loading since
        self.cold_starts = 0
        self.n = 0  # queries finished
        self.infer_sum = self.total_sum = 0.0  # their inference and end-to-end seconds


class Spin:
    """Per-model lifecycle state and GPU accounting.

    scale_to_zero=False gives a static deployment: every model is WARM for the whole run and never
    scaled down. load_estimate(model, now) returns the expected cold-start time of a model; it
    defaults to the stated per-model time in pickspin.config.MODELS. The simulator passes
    SharedStorage.estimate and the live runner the actuator's measured load times.

    The models keep the order of `models`, which sets the order of idle_expired() and of the sums in
    summary(). Spin is thread-safe: one reentrant lock guards every method that reads or changes more
    than one value, and status() and inflight() each read a single value.
    """

    models: list[str]
    cooldown_s: float
    scale_to_zero: bool
    load_estimate: LoadEstimator
    t0: float  # the time the run started (the `now` given to the constructor)
    queries: int  # queries routed so far

    def __init__(
        self,
        models: Iterable[str] = MODELS,
        *,
        cooldown_s: float = DEFAULT_SPIN.cooldown_s,
        scale_to_zero: bool = True,
        now: float = 0.0,
        load_estimate: LoadEstimator | None = None,
        catalog: Mapping[str, ModelSpec] = MODELS,
    ) -> None:
        self.models = list(models)
        self.cooldown_s = cooldown_s
        self.scale_to_zero = scale_to_zero
        self.catalog = catalog
        if load_estimate is None:
            load_estimate = _stated_cold_start if catalog is MODELS else self._catalog_cold_start
        self.load_estimate = load_estimate
        start = ModelState.COLD if scale_to_zero else ModelState.WARM
        self._states = {m: _State(start, now) for m in self.models}
        self._lock = threading.RLock()
        self.t0 = now
        self.queries = 0

    def _catalog_cold_start(self, model: str, now: float) -> float:
        """The default load estimate for a custom catalog: the model's stated cold-start time."""
        return float(self.catalog[model].cold_start_s)

    # --- transitions -------------------------------------------------------------------------
    def status(self, model: str) -> ModelState:
        """Return the model's lifecycle state."""
        return self._states[model].status

    def inflight(self, model: str) -> int:
        """Return the number of queries executing on the model."""
        return self._states[model].inflight

    def request(self, model: str, now: float) -> ModelState:
        """Count a query routed to the model and return the model's state before the call.

        A COLD model starts loading: this is a cold start, its GPUs are held from now on, and its load
        is expected to finish at now + load_estimate(model, now). The query counts as pending until
        start(), so the model cannot be scaled down in between.
        """
        with self._lock:
            st = self._states[model]
            self.queries += 1
            st.pending += 1
            before = st.status
            if before is ModelState.COLD:
                st.status = ModelState.LOADING
                st.cold_starts += 1
                st.alloc_since = now
                st.load_since = now
                st.ready_eta = now + self.load_estimate(model, now)
            return before

    def loaded(self, model: str, now: float) -> None:
        """Mark a LOADING model as WARM; does nothing if the model is not loading."""
        with self._lock:
            st = self._states[model]
            if st.status is not ModelState.LOADING:
                return
            st.status = ModelState.WARM
            assert st.load_since is not None  # set by request() together with LOADING
            st.load_s += now - st.load_since
            st.load_since = st.ready_eta = None
            st.idle_since = now

    def start(self, model: str, now: float) -> None:
        """A pending query starts executing on the WARM model."""
        with self._lock:
            st = self._states[model]
            st.pending -= 1
            if st.inflight == 0:
                st.busy_since = now
            st.inflight += 1

    def finish(self, model: str, now: float, infer_s: float, total_s: float) -> None:
        """A query on the model finished after infer_s seconds of inference and total_s end to end."""
        with self._lock:
            st = self._states[model]
            st.inflight -= 1
            if st.inflight == 0:
                assert st.busy_since is not None  # set by start() when the model became busy
                st.busy_s += now - st.busy_since
                st.busy_since = None
                st.idle_since = now
            st.n += 1
            st.infer_sum += infer_s
            st.total_sum += total_s

    def idle_expired(self, now: float) -> list[str]:
        """Return the WARM models with nothing in flight or pending for at least T_cooldown.

        The idle time is compared as now - idle_since >= cooldown_s. A static deployment never has
        expired models.
        """
        if not self.scale_to_zero:
            return []
        with self._lock:
            return [
                m
                for m, st in self._states.items()
                if st.status is ModelState.WARM
                and st.inflight == 0
                and st.pending == 0
                and now - st.idle_since >= self.cooldown_s
            ]

    def stop(self, model: str, now: float) -> bool:
        """Scale a WARM model to zero and release its GPUs; return False if it is not WARM or is busy again."""
        with self._lock:
            st = self._states[model]
            if st.status is not ModelState.WARM or st.inflight or st.pending:
                return False
            st.status = ModelState.COLD
            assert st.alloc_since is not None  # a WARM model holds its GPUs
            st.alloc_s += now - st.alloc_since
            st.alloc_since = None
            return True

    def cancel(self, model: str) -> None:
        """A pending query gives up before it starts (its model failed to load or took too long)."""
        with self._lock:
            self._states[model].pending -= 1

    def lost(self, model: str, now: float) -> bool:
        """The model's server is gone (its load failed, or it died while WARM): mark the model COLD.

        Its GPUs count as released now, so the next query routed to it is a new cold start. Queries
        still executing finish as usual. Returns False if the model is already COLD.
        """
        with self._lock:
            st = self._states[model]
            if st.status is ModelState.COLD:
                return False
            if st.load_since is not None:
                st.load_s += now - st.load_since
            assert st.alloc_since is not None  # a LOADING or WARM model holds its GPUs
            st.alloc_s += now - st.alloc_since
            st.status = ModelState.COLD
            st.alloc_since = st.load_since = st.ready_eta = None
            st.idle_since = now
            return True

    # --- what Pick sees ------------------------------------------------------------------------
    def latency_estimate(
        self, model: str, now: float, signal: LatencySignal | str = LatencySignal.SPIN
    ) -> float | None:
        """Return the model's latency for Pick's L_norm (Eq. 4), or None if it has not served a query yet.

        'observed' is the mean end-to-end latency and 'inference' the mean inference latency. 'spin'
        (the default) is the mean inference latency plus the cold-start penalty: the load estimate of
        a COLD model, or the time left until a LOADING model is expected to be ready.
        """
        with self._lock:
            st = self._states[model]
            if st.n == 0:
                return None
            if signal == LatencySignal.OBSERVED:
                return st.total_sum / st.n
            infer = st.infer_sum / st.n
            if signal == LatencySignal.INFERENCE:
                return infer
            if st.status is ModelState.COLD:
                return infer + self.load_estimate(model, now)
            if st.status is ModelState.LOADING:
                assert st.ready_eta is not None  # set by request() together with LOADING
                return infer + max(0.0, st.ready_eta - now)
            return infer

    # --- accounting ----------------------------------------------------------------------------
    def summary(self, now: float) -> SpinSummary:
        """Return GPU-hours, utilization and cold starts, counting the intervals still open at now."""
        with self._lock:
            per_model: dict[str, ModelUsage] = {}
            for m, st in self._states.items():
                alloc = st.alloc_s + (now - st.alloc_since if st.alloc_since is not None else 0.0)
                busy = st.busy_s + (now - st.busy_since if st.busy_since is not None else 0.0)
                load = st.load_s + (now - st.load_since if st.load_since is not None else 0.0)
                gpus = self.catalog[m].gpus
                per_model[m] = ModelUsage(
                    gpu_hours=alloc * gpus / 3600,
                    busy_gpu_hours=busy * gpus / 3600,
                    loading_gpu_hours=load * gpus / 3600,
                    cold_starts=st.cold_starts,
                )
            gpu_h = sum(v.gpu_hours for v in per_model.values())
            busy_h = sum(v.busy_gpu_hours for v in per_model.values())
            cold = sum(v.cold_starts for v in per_model.values())
            return SpinSummary(
                gpu_hours=gpu_h,
                busy_gpu_hours=busy_h,
                gpu_utilization=busy_h / gpu_h if gpu_h else 0.0,
                cold_starts=cold,
                cold_start_rate=cold / self.queries if self.queries else 0.0,
                per_model=per_model,
            )
