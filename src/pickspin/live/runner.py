"""The live experiment: route the benchmark queries through Pick and Spin on a Kubernetes deployment.

Pick classifies each query (keyword lists, then DistilBERT) and selects a model with the latency Spin
reports. Spin scales a cold model's Deployment to one replica, holds the query until vLLM answers
/health, forwards it, and scales models with nothing in flight for T_cooldown back to zero. A run
writes <out>/pick_spin_<time>.jsonl, one line per query in completion order, and
<out>/pick_spin_<time>_summary.json with the GPU-hours, utilization and cold starts from Spin's
accounting. A static run keeps every model running and scales nothing.

run_live wires the parts together in a fixed order, which is part of the method:

1. read the endpoints, build the headers and the actuator (none for a static run);
2. create Spin, whose clock starts now, and only then load the classifier, so a static run's
   GPU-hours include the DistilBERT load;
3. create Pick with random.Random(seed);
4. scale every model to zero and wait until none is ready, then start the reaper thread, which every
   5 seconds scales to zero the models that have been idle for T_cooldown;
5. load the queries in file order, shuffle them with a second random.Random(seed) and keep the first
   `limit` (all of them without a limit);
6. send every query to a pool of worker threads, write each record as it completes, then write the
   summary.

LiveRunner holds the per-query steps as methods, so tests can drive them directly with a manual clock,
a scripted chat call and a recording actuator. For each query, process() routes it, counts it with
Spin.request() and, only when the model was COLD, starts a daemon thread that brings the model up; it
waits for the model unless it was already WARM. The locks are always taken in the order Pick's
sampler, then Spin, and never the other way round.
"""

import dataclasses
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from pickspin.config import DEFAULT_SPIN, MODELS, ModelSpec, Tier, tiers_of
from pickspin.data import Query, load_queries
from pickspin.live.actuator import Actuator, KubernetesActuator
from pickspin.live.vllm import ChatResult, Endpoint, bearer_headers, call_vllm, load_endpoints
from pickspin.pick.classifier import HybridClassifier, Stage
from pickspin.pick.router import Pick
from pickspin.spin.lifecycle import LatencySignal, ModelState, Spin

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveConfig:
    """Settings of a live run.

        endpoints is the endpoint map (JSON), queries the benchmark prompts (gzipped JSON lines), out_dir
        where the outputs go and model_dir the fine-tuned DistilBERT. namespace is the Kubernetes namespace
        of the model Deployments. workers is the number of worker threads, each with one query in flight.
        limit keeps the first N queries after the seeded shuffle; None or 0 keeps them all. max_tokens caps
        each response. static keeps every model running instead of scaling to zero. latency_signal is what
        Pick scores on, and seed seeds both the routing and the shuffle. cooldown_s is T_cooldown. api_key
        is sent as a bearer token to vLLM and never appears in repr(). servers, when set, is a server file
    (deploy/nautilus/servers.json): models then run as Jobs that a JobActuator creates and deletes instead
    of Deployments that are scaled.

        The last three fields are knobs for tests, and their defaults are the values a real run uses: the
        reaper checks for idle models every reaper_interval_s seconds, the scale-down at the start polls
        every scale_down_poll_s seconds, and progress is logged every progress_every completed queries.
    """

    endpoints: Path
    queries: Path
    out_dir: Path
    model_dir: Path
    namespace: str = "pick-and-spin"
    workers: int = 250
    limit: int | None = None
    max_tokens: int = 256
    static: bool = False
    latency_signal: LatencySignal = LatencySignal.SPIN
    seed: int = 0
    cooldown_s: float = DEFAULT_SPIN.cooldown_s
    api_key: str | None = field(default=None, repr=False)
    servers: Path | None = None
    reaper_interval_s: float = 5.0
    scale_down_poll_s: float = 2.0
    progress_every: int = 2000


@dataclass(frozen=True, slots=True)
class LiveRecord:
    """One line of the run's JSON-lines output; the field order is the key order.

    wait_s is the time from arrival until the query started (the wait for a cold model to load),
    latency the inference time reported by the chat call, and total_latency the time from arrival until
    the reply, all in seconds and rounded to milliseconds. On success response holds the first 500
    characters of the reply and error is ''; on failure response is '' and error describes the failure.
    """

    id: str
    benchmark: str
    tier: Tier
    stage: Stage | None
    model: str
    cold_start: bool
    waited_for_load: bool
    wait_s: float
    latency: float
    total_latency: float
    success: bool
    tokens: int
    response: str
    error: str

    def to_json(self) -> str:
        """Return the record as one line of JSON, keeping non-ASCII text as is."""
        return json.dumps(dataclasses.asdict(self), ensure_ascii=False)


class ModelUnavailable(Exception):  # noqa: N818 - names the condition, like TimeoutError
    """A query's model cannot serve it: its load failed, or it is still loading after the wait allowed.

    still_loading tells the two apart; in the second case the load goes on and a client may retry.
    """

    def __init__(self, model: str, *, still_loading: bool) -> None:
        super().__init__(f"{model} is still loading" if still_loading else f"{model} failed to load")
        self.model = model
        self.still_loading = still_loading


@dataclass(eq=False)
class Load:
    """One bring-up of a model: done is set when it ends, and ok tells whether the server came up."""

    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


class ModelLifecycle:
    """Spin's side of a live deployment: bring-up of cold models, the reaper and waiting for a model.

    clock returns the current time in seconds (time.monotonic in a real run) and is the time Spin sees.
    The actuator scales the models; only a static deployment may go without one, and it never calls it.

    loads[m] is model m's latest bring-up: a query that finds the model COLD starts a new one, and the
    queries that find it LOADING wait for that one. scale_lock[m] keeps a bring-up and a scale-down of
    the same model apart. Setting stop ends the reaper thread. Both the benchmark replay (LiveRunner)
    and the gateway (pickspin.live.gateway) build on this class.

    recover sets what a failed load does. Without it (the benchmark), the model is still marked WARM
    and the queries that waited for it are sent anyway and recorded as failures. With it (the gateway),
    the failed server is removed and the model goes back to COLD, so the next query starts it afresh,
    and the queries that waited get ModelUnavailable; check_server does the same for a WARM model
    whose server has died.
    """

    spin: Spin
    actuator: Actuator | None
    clock: Callable[[], float]
    static: bool
    recover: bool
    reaper_interval_s: float
    loads: dict[str, Load]
    scale_lock: dict[str, threading.Lock]
    stop: threading.Event

    def __init__(
        self,
        *,
        spin: Spin,
        actuator: Actuator | None,
        static: bool,
        clock: Callable[[], float] = time.monotonic,
        reaper_interval_s: float = 5.0,
        recover: bool = False,
    ) -> None:
        if actuator is None and not static:
            raise ValueError("a live run that scales models needs an actuator; only a static run can do without")
        self.spin = spin
        self.actuator = actuator
        self.static = static
        self.clock = clock
        self.recover = recover
        self.reaper_interval_s = reaper_interval_s
        self.loads = {m: Load() for m in spin.models}
        if static:
            for load in self.loads.values():
                load.ok = True
                load.done.set()
        self.scale_lock = {m: threading.Lock() for m in spin.models}
        self._route_lock = threading.Lock()  # a COLD query's new Load is in place before others look
        self.stop = threading.Event()

    def is_ready(self, model: str) -> bool:
        """Return True if the model is WARM and its latest load brought its server up."""
        load = self.loads[model]
        return self.spin.status(model) is ModelState.WARM and load.done.is_set() and load.ok

    def start_reaper(self) -> None:
        """Start the daemon thread that scales idle models to zero, until stop is set.

        It calls reap_once() every reaper_interval_s seconds. A static deployment starts no reaper.
        """
        if self.static:
            return
        threading.Thread(target=self._reap_until_stopped, daemon=True).start()

    def _reap_until_stopped(self) -> None:
        """The reaper thread's loop."""
        while not self.stop.wait(self.reaper_interval_s):
            self.reap_once()

    def reap_once(self) -> list[str]:
        """Scale to zero every model idle for T_cooldown and return them.

        A model is scaled to zero only if Spin.stop() agrees, which it does not once another query has
        been routed to the model. Both steps run with the model's scale lock held, so a bring-up that
        starts in the meantime scales the model up again only after the scale-down.
        """
        stopped: list[str] = []
        for m in self.spin.idle_expired(self.clock()):
            with self.scale_lock[m]:
                if self.spin.stop(m, self.clock()):
                    assert self.actuator is not None  # only a run that scales to zero has idle models
                    self.actuator.scale(m, 0)
                    stopped.append(m)
        return stopped

    def release_all(self) -> None:
        """Scale every model to zero without waiting, as the gateway does when it starts and stops.

        A failure is logged and the other models are still scaled. A static deployment scales nothing.
        """
        if self.static:
            return
        assert self.actuator is not None  # __init__ requires an actuator unless the deployment is static
        for m in self.spin.models:
            with self.scale_lock[m]:
                try:
                    self.actuator.scale(m, 0)
                except Exception as e:  # keep releasing the others
                    log.warning("Scaling %s to zero failed: %s", m, e)

    def bring_up(self, model: str, load: Load | None = None) -> None:
        """Scale a cold model to one replica, wait until it is ready, and release the queries waiting for it.

        load is the bring-up this is (the model's latest by default). A failed load is logged, and then
        handled as the class docstring describes for recover.
        """
        load = self.loads[model] if load is None else load
        t0 = self.clock()
        try:
            assert self.actuator is not None  # a static run never has a COLD model
            with self.scale_lock[model]:
                self.actuator.scale(model, 1)
            self.actuator.wait_ready(model, t0)
            load.ok = True
        except Exception as e:  # handled below, as recover says
            log.warning("Loading %s failed: %s", model, e)
        finally:
            if load.ok or not self.recover:
                self.spin.loaded(model, self.clock())
            else:
                with self.scale_lock[model]:
                    self._remove_server(model)
            load.done.set()

    def _remove_server(self, model: str) -> None:
        """Delete the model's failed or dead server and mark the model COLD; the caller holds its scale lock."""
        assert self.actuator is not None  # only a deployment that scales models removes servers
        try:
            self.actuator.scale(model, 0)
        except Exception as e:  # the model still goes COLD; its next bring-up replaces the server
            log.warning("Removing the server of %s failed: %s", model, e)
        self.spin.lost(model, self.clock())

    def check_server(self, model: str, load: Load) -> bool:
        """After a query could not reach a WARM model's server: if the server is gone, start over.

        In recover mode, if load is still the model's latest bring-up and the actuator reports that the
        server is not alive, the server is removed and the model marked COLD, so the next query brings it
        up again. Returns True if that happened.
        """
        if not self.recover or self.actuator is None:
            return False
        with self.scale_lock[model]:
            if self.loads[model] is not load or self.spin.status(model) is not ModelState.WARM:
                return False  # already handled, or a new server is on its way
            if self.actuator.alive(model):
                return False
            log.warning("The server of %s is gone; the next query starts it again", model)
            self._remove_server(model)
        return True

    def acquire(self, model: str, now: float, timeout: float | None = None) -> ModelState:
        """Count a query as routed to model and wait until the model can serve it.

        Returns the model's state when the query arrived. Only a query that finds its model COLD starts
        a bring-up, in a daemon thread; every query whose model is not WARM yet waits for that bring-up.
        A query still waiting after timeout seconds, or (in recover mode) whose model failed to load,
        gives up (Spin.cancel) and raises ModelUnavailable; the bring-up goes on without it.
        """
        with self._route_lock:
            before = self.spin.request(model, now)
            if before is ModelState.COLD:
                self.loads[model] = Load()
                threading.Thread(target=self.bring_up, args=(model, self.loads[model]), daemon=True).start()
            load = self.loads[model]
        if before is ModelState.WARM:
            return before
        if not load.done.wait(timeout):
            self.spin.cancel(model)
            raise ModelUnavailable(model, still_loading=True)
        if self.recover and not load.ok:
            self.spin.cancel(model)
            raise ModelUnavailable(model, still_loading=False)
        return before


class LiveRunner(ModelLifecycle):
    """The per-query mechanics of a live benchmark run on top of ModelLifecycle.

    chat sends one query to an endpoint, like call_vllm.
    """

    config: LiveConfig
    pick: Pick
    endpoints: Mapping[str, Endpoint]
    headers: Mapping[str, str]
    chat: Callable[[Endpoint, str, int, Mapping[str, str]], ChatResult]

    def __init__(
        self,
        config: LiveConfig,
        *,
        pick: Pick,
        spin: Spin,
        endpoints: Mapping[str, Endpoint],
        headers: Mapping[str, str],
        actuator: Actuator | None,
        clock: Callable[[], float] = time.monotonic,
        chat: Callable[[Endpoint, str, int, Mapping[str, str]], ChatResult] = call_vllm,
    ) -> None:
        super().__init__(
            spin=spin,
            actuator=actuator,
            static=config.static,
            clock=clock,
            reaper_interval_s=config.reaper_interval_s,
        )
        self.config = config
        self.pick = pick
        self.endpoints = endpoints
        self.headers = headers
        self.chat = chat

    def prepare(self) -> None:
        """Scale every model to zero and wait until none is ready, so the run starts COLD.

        Models are scaled in Spin's model order, then the ready replicas are polled every
        scale_down_poll_s seconds. A static run does nothing here.
        """
        if self.config.static:
            return
        actuator = self.actuator
        assert actuator is not None  # __init__ requires an actuator unless the run is static
        log.info("Scaling every model to zero so the run starts COLD ...")
        for m in self.spin.models:
            actuator.scale(m, 0)
        while any(actuator.ready_replicas(m) for m in self.spin.models):
            time.sleep(self.config.scale_down_poll_s)

    def process(self, query: Query) -> LiveRecord:
        """Route one query, wait for its model if needed, send it and record the outcome.

        The query counts as routed (Spin.request) before anything else happens to its model. Only a
        query that finds its model COLD starts the bring-up, in a daemon thread; every query whose
        model is not WARM yet waits until the model is ready.
        """
        t_arrive = self.clock()
        route = self.pick.route(query.query, t_arrive)
        m = route.model
        before = self.acquire(m, t_arrive)
        t_start = self.clock()
        self.spin.start(m, t_start)
        ok, text, tokens, infer_s = self.chat(self.endpoints[m], query.query, self.config.max_tokens, self.headers)
        t_end = self.clock()
        self.spin.finish(m, t_end, infer_s, t_end - t_arrive)
        self.pick.update(m, route.tier, ok)
        return LiveRecord(
            id=query.id,
            benchmark=query.benchmark,
            tier=route.tier,
            stage=route.stage,
            model=m,
            cold_start=before is ModelState.COLD,
            waited_for_load=before is not ModelState.WARM,
            wait_s=round(t_start - t_arrive, 3),
            latency=round(infer_s, 3),
            total_latency=round(t_end - t_arrive, 3),
            success=ok,
            tokens=tokens,
            response=text[:500] if ok else "",
            error="" if ok else text,
        )

    def run(self, queries: Sequence[Query]) -> Path:
        """Process the queries with the worker pool, write the outputs and return their path stem.

        Every query is submitted to the pool up front, and the records are written in completion order
        to <stem>.jsonl. Then the reaper is stopped and Spin's summary at the current time, plus the
        actuator's measured load times, is written to <stem>_summary.json.
        """
        out_dir = self.config.out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"pick_spin_{datetime.now():%Y%m%d_%H%M%S}"
        log.info(
            "Routing %s queries with %d workers (%s) -> %s.jsonl",
            f"{len(queries):,}",
            self.config.workers,
            "static" if self.config.static else "scale to zero",
            stem,
        )
        done = 0
        with (
            Path(f"{stem}.jsonl").open("w", encoding="utf-8") as f,
            ThreadPoolExecutor(max_workers=self.config.workers) as pool,
        ):
            for fut in as_completed([pool.submit(self.process, q) for q in queries]):
                f.write(fut.result().to_json() + "\n")
                done += 1
                if done % self.config.progress_every == 0:
                    log.info("[%s] %d cold starts so far", f"{done:,}", self.spin.summary(self.clock()).cold_starts)
        self.stop.set()
        usage = self.spin.summary(self.clock())
        summary = usage.to_dict()
        summary["measured_load_s"] = self.actuator.measured if self.actuator is not None else {}
        phases = getattr(self.actuator, "phases", None)
        if phases is not None:
            summary["load_phases"] = phases
        with Path(f"{stem}_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=1)
        log.info(
            "%.2f GPU-hours, utilization %.1f%%, %d cold starts (%.2f%% of queries)",
            usage.gpu_hours,
            100 * usage.gpu_utilization,
            usage.cold_starts,
            100 * usage.cold_start_rate,
        )
        return stem


def run_live(
    config: LiveConfig,
    *,
    actuator: Actuator | None = None,
    classifier: HybridClassifier | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> Path:
    """Run a live experiment and return the path stem of its outputs.

    Without an actuator a run that scales models creates a KubernetesActuator (the [live] extra) for
    config.namespace; a static run never uses one. Without a classifier the fine-tuned DistilBERT is
    loaded from config.model_dir (the [classifier] extra). clock is the time Spin sees.
    """
    endpoints = load_endpoints(config.endpoints)
    headers = bearer_headers(config.api_key)
    catalog: Mapping[str, ModelSpec] = MODELS
    servers = None
    if config.servers is not None:
        from pickspin.live.jobs import JobActuator, load_servers

        servers = load_servers(config.servers)
        catalog = servers["catalog"]
    if config.static:
        actuator = None
    elif actuator is None and servers is not None:
        timeout_s = float(servers["defaults"].get("load_timeout_s", 3600))
        actuator = JobActuator(servers, endpoints, config.namespace, timeout_s=timeout_s)
    elif actuator is None:
        actuator = KubernetesActuator(endpoints, config.namespace)
    spin = Spin(
        catalog,
        cooldown_s=config.cooldown_s,
        scale_to_zero=not config.static,
        now=clock(),
        load_estimate=actuator.load_estimate if actuator is not None else None,
        catalog=catalog,
    )
    if classifier is None:
        # Loaded after Spin's clock started, so a static run's GPU-hours include the load.
        classifier = HybridClassifier.from_pretrained(config.model_dir)
    pick = Pick(
        classifier, spin, config.latency_signal, rng=random.Random(config.seed), tiers=tiers_of(catalog), models=catalog
    )
    runner = LiveRunner(
        config, pick=pick, spin=spin, endpoints=endpoints, headers=headers, actuator=actuator, clock=clock
    )
    runner.prepare()
    runner.start_reaper()
    queries = load_queries(config.queries)
    random.Random(config.seed).shuffle(queries)
    queries = queries[: config.limit] if config.limit else queries
    return runner.run(queries)
