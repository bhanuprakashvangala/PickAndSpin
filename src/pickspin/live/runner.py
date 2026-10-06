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

from pickspin.config import DEFAULT_SPIN, MODELS, Tier
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


class LiveRunner:
    """The per-query mechanics of a live run: bring-up of cold models, the reaper and query processing.

    clock returns the current time in seconds (time.monotonic in a real run) and is the time Spin sees.
    chat sends one query to an endpoint, like call_vllm. The actuator scales the models; only a static
    run may go without one, and a static run never calls it.

    ready[m] is set while queries for model m may be sent: always in a static run, otherwise from the
    end of the model's bring-up until the reaper scales it to zero. scale_lock[m] keeps a bring-up and
    a scale-down of the same model apart. Setting stop ends the reaper thread.
    """

    config: LiveConfig
    pick: Pick
    spin: Spin
    endpoints: Mapping[str, Endpoint]
    headers: Mapping[str, str]
    actuator: Actuator | None
    clock: Callable[[], float]
    chat: Callable[[Endpoint, str, int, Mapping[str, str]], ChatResult]
    ready: dict[str, threading.Event]
    scale_lock: dict[str, threading.Lock]
    stop: threading.Event

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
        if actuator is None and not config.static:
            raise ValueError("a live run that scales models needs an actuator; only a static run can do without")
        self.config = config
        self.pick = pick
        self.spin = spin
        self.endpoints = endpoints
        self.headers = headers
        self.actuator = actuator
        self.clock = clock
        self.chat = chat
        self.ready = {m: threading.Event() for m in MODELS}
        self.scale_lock = {m: threading.Lock() for m in MODELS}
        if config.static:
            for e in self.ready.values():
                e.set()
        self.stop = threading.Event()

    def prepare(self) -> None:
        """Scale every model to zero and wait until none is ready, so the run starts COLD.

        Models are scaled in MODELS order, then the ready replicas are polled every scale_down_poll_s
        seconds. A static run does nothing here.
        """
        if self.config.static:
            return
        actuator = self.actuator
        assert actuator is not None  # __init__ requires an actuator unless the run is static
        log.info("Scaling every model to zero so the run starts COLD ...")
        for m in MODELS:
            actuator.scale(m, 0)
        while any(actuator.ready_replicas(m) for m in MODELS):
            time.sleep(self.config.scale_down_poll_s)

    def start_reaper(self) -> None:
        """Start the daemon thread that scales idle models to zero, until stop is set.

        It calls reap_once() every reaper_interval_s seconds. A static run starts no reaper.
        """
        if self.config.static:
            return
        threading.Thread(target=self._reap_until_stopped, daemon=True).start()

    def _reap_until_stopped(self) -> None:
        """The reaper thread's loop."""
        while not self.stop.wait(self.config.reaper_interval_s):
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
                    self.ready[m].clear()
                    assert self.actuator is not None  # only a run that scales to zero has idle models
                    self.actuator.scale(m, 0)
                    stopped.append(m)
        return stopped

    def bring_up(self, model: str) -> None:
        """Scale a cold model to one replica, wait until it is ready, and release the queries waiting for it.

        A failed load is logged and the model is still marked WARM: the queries waiting for it are sent
        anyway and recorded as failures.
        """
        t0 = self.clock()
        try:
            assert self.actuator is not None  # a static run never has a COLD model
            with self.scale_lock[model]:
                self.actuator.scale(model, 1)
            self.actuator.wait_ready(model, t0)
        except Exception as e:  # the waiting queries are sent anyway and recorded as failures
            log.warning("Loading %s failed: %s", model, e)
        finally:
            self.spin.loaded(model, self.clock())
            self.ready[model].set()

    def process(self, query: Query) -> LiveRecord:
        """Route one query, wait for its model if needed, send it and record the outcome.

        The query counts as routed (Spin.request) before anything else happens to its model. Only a
        query that finds its model COLD starts the bring-up, in a daemon thread; every query whose
        model is not WARM yet waits until the model is ready.
        """
        t_arrive = self.clock()
        route = self.pick.route(query.query, t_arrive)
        m = route.model
        before = self.spin.request(m, t_arrive)
        if before is ModelState.COLD:
            threading.Thread(target=self.bring_up, args=(m,), daemon=True).start()
        if before is not ModelState.WARM:
            self.ready[m].wait()
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
    if config.static:
        actuator = None
    elif actuator is None and config.servers is not None:
        from pickspin.live.jobs import JobActuator, load_servers

        servers = load_servers(config.servers)
        timeout_s = float(servers["defaults"].get("load_timeout_s", 3600))
        actuator = JobActuator(servers, endpoints, config.namespace, timeout_s=timeout_s)
    elif actuator is None:
        actuator = KubernetesActuator(endpoints, config.namespace)
    spin = Spin(
        cooldown_s=config.cooldown_s,
        scale_to_zero=not config.static,
        now=clock(),
        load_estimate=actuator.load_estimate if actuator is not None else None,
    )
    if classifier is None:
        # Loaded after Spin's clock started, so a static run's GPU-hours include the load.
        classifier = HybridClassifier.from_pretrained(config.model_dir)
    pick = Pick(classifier, spin, config.latency_signal, rng=random.Random(config.seed))
    runner = LiveRunner(
        config, pick=pick, spin=spin, endpoints=endpoints, headers=headers, actuator=actuator, clock=clock
    )
    runner.prepare()
    runner.start_reaper()
    queries = load_queries(config.queries)
    random.Random(config.seed).shuffle(queries)
    queries = queries[: config.limit] if config.limit else queries
    return runner.run(queries)
