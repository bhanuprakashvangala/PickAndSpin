"""The discrete-event simulator: Pick and Spin on a simulated clock.

Queries come from closed-loop workers, each of which sends its next query when the previous one
returns, or arrive as a Poisson process. Pick routes each query with its cached tier, and the query
takes the latency and success recorded for its (query, model) pair. Cold starts share the storage
bandwidth (SharedStorage, Eq. 5), queries for a model that is not warm wait for its load (Eq. 6), and
a model with nothing in flight for T_cooldown is scaled to zero. A model serves any number of queries
at once unless max_concurrency caps it; the queries over the cap wait in arrival order.

A run is deterministic given its seed. It uses three separate random streams: the query order is
shuffled with random.Random(seed + SHUFFLE_SEED_OFFSET), Pick samples with
random.Random(SAMPLER_SEED_OFFSET + seed), and the Poisson arrival times come from
random.Random(ARRIVAL_SEED_OFFSET + seed).

Seeded runs must stay bit-identical to v1.1.0 (tests/data/golden). Every event handler keeps the
statements of the old event loop in their old order, events at equal times run in the order they
were pushed, every arrival time is drawn before the loop by plain accumulation, and Spin consults the
storage's load estimate (which advances the transfers in flight) exactly where it always did. The
event loop never logs.
"""

import heapq
import itertools
import math
import random
from collections import defaultdict, deque
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final, Literal, TypeAlias

from pickspin.config import DEFAULT_SPIN, Tier
from pickspin.errors import ConfigError
from pickspin.pick.classifier import Stage
from pickspin.pick.router import Pick
from pickspin.simulation.inputs import TierAssignment, TraceData
from pickspin.simulation.policies import POLICIES, Policy
from pickspin.spin.lifecycle import ModelState, Spin, SpinSummary
from pickspin.spin.storage import SharedStorage, init_seconds

# Offsets added to a run's seed, one per random stream. Never share a stream between purposes.
SHUFFLE_SEED_OFFSET: Final = 0
SAMPLER_SEED_OFFSET: Final = 10_000
ARRIVAL_SEED_OFFSET: Final = 20_000

# Event kinds and their payloads: 'next' a closed-loop worker, 'arrive' the index of a Poisson arrival,
# 'transfer' (storage version, model), 'ready' and 'idle' a model, 'done' the finished QueryRecord.
_Kind: TypeAlias = Literal["next", "arrive", "transfer", "ready", "done", "idle"]
# A heap entry: (time, push sequence number, kind, payload). The sequence numbers are unique, so
# events at equal times run in push order and payloads are never compared.
_Event: TypeAlias = tuple[float, int, _Kind, Any]


@dataclass(frozen=True, slots=True)
class SimulationSettings:
    """The load of one simulation and Spin's limits.

    By default `workers` closed-loop clients each send their next query when the previous one returns.
    With an arrival_rate (queries per second) the queries instead arrive as a Poisson process and
    workers is ignored; a rate of None or 0 means the closed loop. max_concurrency caps the queries
    executing at once on each model (None: no cap), and cooldown_s is T_cooldown, how long a model
    must have nothing in flight before it is scaled to zero.
    """

    workers: int = 250
    arrival_rate: float | None = None
    max_concurrency: int | None = None
    cooldown_s: float = DEFAULT_SPIN.cooldown_s

    @property
    def load_name(self) -> str:
        """Name of the load and of its output directory: closed-<workers> or poisson-<rate>qps."""
        return f"poisson-{self.arrival_rate:g}qps" if self.arrival_rate else f"closed-{self.workers}"


@dataclass(slots=True)
class QueryRecord:
    """What happened to one query in a simulated run; times are in seconds from the start of the run.

    order is the query's 1-based position in the shuffled dispatch order. The query was routed at
    `arrive`, started executing at `start` and finished at `end`; wait_s = start - arrive is the time
    it waited for a load or a free slot, and total_s = end - arrive its end-to-end latency. infer_s and
    success are recorded for its (query, model) pair, and correct is the judge's label, or None when
    the pair was not scored. cold_start is True when this query started its model's load. worker is
    the closed-loop client that sent it, or None for a Poisson arrival. The four times are NaN until
    the query starts and finishes.
    """

    order: int
    id: str
    benchmark: str
    tier: Tier
    stage: Stage
    model: str
    arrive: float
    infer_s: float
    success: bool
    correct: bool | None
    cold_start: bool
    worker: int | None
    start: float = math.nan
    end: float = math.nan
    wait_s: float = math.nan
    total_s: float = math.nan


@dataclass(frozen=True)
class SimulationResult:
    """The outcome of one run. records[i] belongs to the i-th query of the input, in file order.

    usage is Spin's GPU and cold-start accounting at the last completion, and makespan_s the time of
    that completion.
    """

    records: list[QueryRecord]
    usage: SpinSummary
    makespan_s: float


def simulate(
    policy: Policy,
    seed: int,
    data: TraceData,
    tiers: Mapping[str, TierAssignment],
    settings: SimulationSettings | None = None,
) -> SimulationResult:
    """Run one policy with one seed over every query of data.

    tiers gives the (tier, stage) of every query id, and settings the load (by default 250 closed-loop
    workers and T_cooldown = 300 s). The same arguments always give the same result, bit for bit.
    Raises ValueError when data holds no queries, and ConfigError for settings under which the run
    could never finish: fewer than one worker in a closed loop, a max_concurrency below 1, or an
    arrival rate that is not positive.
    """
    if not data.queries:
        raise ValueError("there are no queries to simulate")
    settings = SimulationSettings() if settings is None else settings
    if settings.max_concurrency is not None and settings.max_concurrency < 1:
        raise ConfigError(f"max_concurrency must be at least 1, not {settings.max_concurrency}")
    if settings.arrival_rate:
        if not settings.arrival_rate > 0:
            raise ConfigError(f"the arrival rate must be positive, not {settings.arrival_rate}")
    elif settings.workers < 1:
        raise ConfigError(f"a closed-loop run needs at least one worker, not {settings.workers}")
    return _Run(Policy(policy), seed, data, tiers, settings).run()


class _Run:
    """The state of one simulated run, with one handler per kind of event.

    Each handler is a branch of the v1.1.0 event loop, with its statements in the old order: the order
    of the pushes and of the calls into Pick, Spin and SharedStorage decides the result bit for bit.
    """

    __slots__ = (
        "data",
        "dispatched",
        "events",
        "finished",
        "makespan_s",
        "order",
        "pick",
        "records",
        "seed",
        "seq",
        "settings",
        "spec",
        "spin",
        "storage",
        "storage_version",
        "tiers",
        "usage",
        "waiting",
    )

    def __init__(
        self,
        policy: Policy,
        seed: int,
        data: TraceData,
        tiers: Mapping[str, TierAssignment],
        settings: SimulationSettings,
    ) -> None:
        self.data = data
        self.tiers = tiers
        self.settings = settings
        self.seed = seed
        self.spec = POLICIES[policy]
        # The query indices in dispatch order.
        self.order = list(range(len(data.queries)))
        random.Random(seed + SHUFFLE_SEED_OFFSET).shuffle(self.order)
        self.storage = SharedStorage()
        self.spin = Spin(
            cooldown_s=settings.cooldown_s,
            scale_to_zero=self.spec.scale_to_zero,
            now=0.0,
            load_estimate=self.storage.estimate,
        )
        self.pick = Pick(None, self.spin, self.spec.signal, rng=random.Random(SAMPLER_SEED_OFFSET + seed))
        self.events: list[_Event] = []
        self.seq = itertools.count(1)
        # Incremented whenever the next transfer is rescheduled; older 'transfer' events are stale.
        self.storage_version = 0
        # Per model, the routed queries that wait for its load or for a free slot, first come first served.
        self.waiting: defaultdict[str, deque[QueryRecord]] = defaultdict(deque)
        self.records: list[QueryRecord | None] = [None] * len(data.queries)
        self.dispatched = 0  # queries sent by the closed-loop workers so far
        self.finished = 0
        self.usage: SpinSummary | None = None
        self.makespan_s = math.nan

    def run(self) -> SimulationResult:
        """Schedule the arrivals or the workers' first queries, then process events until the last query finishes.

        Events still pending at the last completion are discarded.
        """
        n = len(self.order)
        rate = self.settings.arrival_rate
        if rate:
            arrivals, t = random.Random(ARRIVAL_SEED_OFFSET + self.seed), 0.0
            for k in range(n):
                t += arrivals.expovariate(rate)
                self._push(t, "arrive", k)
        else:
            for w in range(min(self.settings.workers, n)):
                self._push(0.0, "next", w)

        events = self.events
        while events:
            t, _, kind, payload = heapq.heappop(events)
            if kind == "next":
                self._on_next(t, payload)
            elif kind == "arrive":
                self._on_arrive(t, payload)
            elif kind == "transfer":
                self._on_transfer(t, payload)
            elif kind == "ready":
                self._on_ready(t, payload)
            elif kind == "done":
                if self._on_done(t, payload):
                    break
            elif kind == "idle":
                self._on_idle(t, payload)

        if self.usage is None:
            raise RuntimeError(f"the simulation stopped with {self.finished:,} of {n:,} queries finished")
        # Every query has finished, so every slot holds its record.
        records = [r for r in self.records if r is not None]
        return SimulationResult(records=records, usage=self.usage, makespan_s=self.makespan_s)

    # --- helpers shared by the handlers ------------------------------------------------------
    def _push(self, t: float, kind: _Kind, payload: Any) -> None:
        heapq.heappush(self.events, (t, next(self.seq), kind, payload))

    def _reschedule_storage(self) -> None:
        """Schedule the transfer that finishes next at the current sharing; earlier schedules go stale."""
        self.storage_version += 1
        nxt = self.storage.next_transfer_done()
        if nxt:
            self._push(nxt[0], "transfer", (self.storage_version, nxt[1]))

    def _can_start(self, model: str) -> bool:
        """Whether a query can start on the model now: it is WARM and below max_concurrency."""
        cap = self.settings.max_concurrency
        return self.spin.status(model) is ModelState.WARM and (cap is None or self.spin.inflight(model) < cap)

    def _start(self, rec: QueryRecord, t: float) -> None:
        self.spin.start(rec.model, t)
        rec.start = t
        self._push(t + rec.infer_s, "done", rec)

    def _start_waiting(self, model: str, t: float) -> None:
        """Start the queries waiting for the model, in the order they were routed, while it can take them."""
        waiting = self.waiting[model]
        while waiting and self._can_start(model):
            self._start(waiting.popleft(), t)

    def _dispatch(self, k: int, t: float, worker: int | None) -> None:
        """Route the k-th query of the dispatch order at time t, then start it or queue it for its model."""
        qi = self.order[k]
        q = self.data.queries[qi]
        tier, stage = self.tiers[q.id]
        m = self.pick.select(tier, t, stage).model
        ok, infer_s = self.data.runs[(q.id, m)]
        before = self.spin.request(m, t)
        rec = QueryRecord(
            order=k + 1,
            id=q.id,
            benchmark=q.benchmark,
            tier=tier,
            stage=stage,
            model=m,
            arrive=t,
            infer_s=infer_s,
            success=ok,
            correct=self.data.correct.get((q.id, m)),
            cold_start=before is ModelState.COLD,
            worker=worker,
        )
        self.records[qi] = rec
        if before is ModelState.COLD:
            self.storage.begin(m, t)
            self._reschedule_storage()
        if self._can_start(m):
            self._start(rec, t)
        else:
            self.waiting[m].append(rec)

    # --- one handler per kind of event ---------------------------------------------------------
    def _on_next(self, t: float, worker: int) -> None:
        """A closed-loop worker is free: it sends the next query, while any are left."""
        if self.dispatched < len(self.order):
            self.dispatched += 1
            self._dispatch(self.dispatched - 1, t, worker)

    def _on_arrive(self, t: float, k: int) -> None:
        """The k-th Poisson arrival."""
        self._dispatch(k, t, None)

    def _on_transfer(self, t: float, scheduled: tuple[int, str]) -> None:
        """A model's weights have arrived, unless this schedule is stale; it is ready L_init(m) later."""
        version, m = scheduled
        if version != self.storage_version:
            return
        self.storage.transfer_done(m, t)
        self._reschedule_storage()
        self._push(t + init_seconds(m, self.storage.gbps), "ready", m)

    def _on_ready(self, t: float, m: str) -> None:
        """A model has loaded: it turns WARM and starts the queries that waited for it."""
        self.spin.loaded(m, t)
        self._start_waiting(m, t)

    def _on_done(self, t: float, rec: QueryRecord) -> bool:
        """A query has finished. Returns True when it was the last one, which ends the run."""
        m = rec.model
        rec.end = t
        rec.wait_s = rec.start - rec.arrive
        rec.total_s = t - rec.arrive
        self.spin.finish(m, t, rec.infer_s, rec.total_s)
        self.pick.update(m, rec.tier, rec.success)
        self.finished += 1
        self._start_waiting(m, t)
        if self.spin.inflight(m) == 0 and self.spec.scale_to_zero:
            self._push(t + self.settings.cooldown_s, "idle", m)
        if self.finished == len(self.order):
            self.usage = self.spin.summary(t)
            self.makespan_s = t
            return True
        if rec.worker is not None:
            self._push(t, "next", rec.worker)
        return False

    def _on_idle(self, t: float, m: str) -> None:
        """The model's cooldown may have expired: scale it to zero if it is still idle."""
        if m in self.spin.idle_expired(t):
            self.spin.stop(m, t)
