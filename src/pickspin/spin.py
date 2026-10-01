"""Spin: lifecycle-aware orchestration (Sec. V).

Each model is COLD (no pod, no GPU), LOADING (pod scheduled, weights being read into GPU
memory) or WARM (ready to serve):

    COLD -> LOADING   a query is routed to a cold model (a cold-start event)
    LOADING -> WARM   the weights are loaded; queries that waited are forwarded
    WARM -> COLD      the model has had no query in flight for T_cooldown seconds; the pod is
                      scaled to zero and its GPU released

A cold start takes L_cold(m) = WeightSize(m) / StorageBandwidth + L_init(m) (Eq. 5); loads that
overlap share the storage bandwidth. A query's total latency is its inference latency plus any
wait for its model to load (Eq. 6).

Spin keeps the per-model latency that Pick scores on (latency_estimate) and accounts for GPU
time: a model holds its GPUs from the start of a load until it is scaled to zero.

`Spin` is the state machine with an explicit clock, shared by the simulator (simulate.py) and the
live runner (run_live.py). `SharedStorage` models Eq. 5 for the simulator; `KubernetesActuator`
scales real vLLM Deployments.
"""

import threading
import time

import requests

from config import MODELS, SPIN

COLD, LOADING, WARM = "COLD", "LOADING", "WARM"


def init_seconds(m, gbps=SPIN["storage_gbps"]):
    """L_init(m): the part of the stated cold-start time that is not the weight transfer."""
    c = MODELS[m]
    return max(0.0, c["cold_start_s"] - c["weight_gb"] / gbps)


class _State:
    __slots__ = ("status", "pending", "inflight", "idle_since", "ready_eta", "alloc_since", "busy_since",
                 "alloc_s", "busy_s", "load_since", "load_s", "cold_starts", "n", "infer_sum", "total_sum")

    def __init__(self, status, now):
        self.status = status
        self.pending = 0      # routed to this model, not started yet (waiting for a load or a slot)
        self.inflight = 0
        self.idle_since = now
        self.ready_eta = None
        self.alloc_since = now if status == WARM else None
        self.busy_since = None
        self.alloc_s = self.busy_s = self.load_s = 0.0
        self.load_since = None
        self.cold_starts = 0
        self.n = 0
        self.infer_sum = self.total_sum = 0.0


class Spin:
    """Per-model lifecycle state and accounting.

    scale_to_zero=False gives a static deployment: every model is WARM for the whole run and
    never scaled down. load_estimate(m, now) returns the expected cold-start time of m; it
    defaults to the stated per-model time in config.MODELS.
    """

    def __init__(self, models=MODELS, cooldown_s=SPIN["cooldown_s"], scale_to_zero=True, now=0.0,
                 load_estimate=None):
        self.models = list(models)
        self.cooldown_s = cooldown_s
        self.scale_to_zero = scale_to_zero
        self.load_estimate = load_estimate or (lambda m, now: float(MODELS[m]["cold_start_s"]))
        start = COLD if scale_to_zero else WARM
        self.s = {m: _State(start, now) for m in self.models}
        self.lock = threading.RLock()
        self.t0 = now
        self.queries = 0

    # --- transitions ---------------------------------------------------------------------
    def status(self, m):
        return self.s[m].status

    def request(self, m, now):
        """A query has been routed to m. Returns m's status before the call; COLD starts a load.

        The query counts as pending until start(), so m cannot be scaled down in between.
        """
        with self.lock:
            st = self.s[m]
            self.queries += 1
            st.pending += 1
            before = st.status
            if before == COLD:
                st.status = LOADING
                st.cold_starts += 1
                st.alloc_since = now
                st.load_since = now
                st.ready_eta = now + self.load_estimate(m, now)
            return before

    def loaded(self, m, now):
        """LOADING -> WARM."""
        with self.lock:
            st = self.s[m]
            if st.status != LOADING:
                return
            st.status = WARM
            st.load_s += now - st.load_since
            st.load_since = st.ready_eta = None
            st.idle_since = now

    def start(self, m, now):
        """A pending query starts executing on the WARM model m."""
        with self.lock:
            st = self.s[m]
            st.pending -= 1
            if st.inflight == 0:
                st.busy_since = now
            st.inflight += 1

    def finish(self, m, now, infer_s, total_s):
        """A query on m finished after infer_s seconds of inference and total_s end to end."""
        with self.lock:
            st = self.s[m]
            st.inflight -= 1
            if st.inflight == 0:
                st.busy_s += now - st.busy_since
                st.busy_since = None
                st.idle_since = now
            st.n += 1
            st.infer_sum += infer_s
            st.total_sum += total_s

    def idle_expired(self, now):
        """WARM models with nothing in flight for at least T_cooldown."""
        if not self.scale_to_zero:
            return []
        with self.lock:
            return [m for m, st in self.s.items()
                    if st.status == WARM and st.inflight == 0 and st.pending == 0
                    and now - st.idle_since >= self.cooldown_s]

    def stop(self, m, now):
        """WARM -> COLD: scale to zero and release the GPU. Returns False if m is busy again."""
        with self.lock:
            st = self.s[m]
            if st.status != WARM or st.inflight or st.pending:
                return False
            st.status = COLD
            st.alloc_s += now - st.alloc_since
            st.alloc_since = None
            return True

    # --- what Pick sees --------------------------------------------------------------------
    def latency_estimate(self, m, now, signal="spin"):
        """Latency of m for Pick's L_norm (Eq. 4), or None if m has not served a query yet."""
        with self.lock:
            st = self.s[m]
            if st.n == 0:
                return None
            if signal == "observed":
                return st.total_sum / st.n
            infer = st.infer_sum / st.n
            if signal == "inference":
                return infer
            if st.status == COLD:
                return infer + self.load_estimate(m, now)
            if st.status == LOADING:
                return infer + max(0.0, st.ready_eta - now)
            return infer

    # --- accounting --------------------------------------------------------------------------
    def summary(self, now):
        """GPU-hours, utilization and cold starts, counting open intervals up to `now`."""
        with self.lock:
            per_model = {}
            for m, st in self.s.items():
                alloc = st.alloc_s + (now - st.alloc_since if st.alloc_since is not None else 0.0)
                busy = st.busy_s + (now - st.busy_since if st.busy_since is not None else 0.0)
                load = st.load_s + (now - st.load_since if st.load_since is not None else 0.0)
                gpus = MODELS[m]["gpus"]
                per_model[m] = {"gpu_hours": alloc * gpus / 3600, "busy_gpu_hours": busy * gpus / 3600,
                                "loading_gpu_hours": load * gpus / 3600, "cold_starts": st.cold_starts}
            gpu_h = sum(v["gpu_hours"] for v in per_model.values())
            busy_h = sum(v["busy_gpu_hours"] for v in per_model.values())
            cold = sum(v["cold_starts"] for v in per_model.values())
            return {"gpu_hours": gpu_h, "busy_gpu_hours": busy_h,
                    "gpu_utilization": busy_h / gpu_h if gpu_h else 0.0,
                    "cold_starts": cold, "cold_start_rate": cold / self.queries if self.queries else 0.0,
                    "per_model": per_model}


class SharedStorage:
    """Weight loads that share the storage bandwidth (Eq. 5 with contention), for the simulator.

    Each load first transfers weight_gb at gbps / k (k = loads transferring at that moment) and
    then spends L_init(m). Without contention a load takes exactly MODELS[m]["cold_start_s"].
    """

    def __init__(self, gbps=SPIN["storage_gbps"]):
        self.gbps = gbps
        self.remaining = {}
        self.t = 0.0

    def _advance(self, now):
        if self.remaining and now > self.t:
            moved = (now - self.t) * self.gbps / len(self.remaining)
            for m in self.remaining:
                self.remaining[m] -= moved
        self.t = max(self.t, now)

    def begin(self, m, now):
        self._advance(now)
        self.remaining[m] = float(MODELS[m]["weight_gb"])

    def next_transfer_done(self):
        """(time, model) of the next transfer to finish at the current sharing, or None."""
        if not self.remaining:
            return None
        m = min(self.remaining, key=self.remaining.get)
        return self.t + max(0.0, self.remaining[m]) * len(self.remaining) / self.gbps, m

    def transfer_done(self, m, now):
        self._advance(now)
        del self.remaining[m]

    def estimate(self, m, now):
        """Expected cold-start time of m if it started loading now, given the loads in progress."""
        self._advance(now)
        k = len(self.remaining) + (0 if m in self.remaining else 1)
        return MODELS[m]["weight_gb"] * k / self.gbps + init_seconds(m, self.gbps)


class KubernetesActuator:
    """Scales one vLLM Deployment per model and waits for its /health endpoint.

    endpoints: {model: {"base_url": ..., "deployment": ...}} (deploy/endpoints.example.json).
    Uses the in-cluster ServiceAccount when running in a pod, otherwise ~/.kube/config.
    Measured load times replace the stated per-model times in load_estimate once available.
    """

    def __init__(self, endpoints, namespace="pick-and-spin", poll_s=2.0, timeout_s=1200):
        from kubernetes import client, config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()
        self.apps = client.AppsV1Api()
        self.endpoints = endpoints
        self.namespace = namespace
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.measured = {}
        self.lock = threading.Lock()

    def scale(self, m, replicas):
        self.apps.patch_namespaced_deployment_scale(self.endpoints[m]["deployment"], self.namespace,
                                                    {"spec": {"replicas": replicas}})

    def ready_replicas(self, m):
        status = self.apps.read_namespaced_deployment_status(self.endpoints[m]["deployment"], self.namespace).status
        return status.ready_replicas or 0

    def healthy(self, m):
        try:
            return requests.get(self.endpoints[m]["base_url"].rstrip("/") + "/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, m, t0):
        """Block until m's Deployment has a ready pod and vLLM answers /health. Returns the load time."""
        while not (self.ready_replicas(m) >= 1 and self.healthy(m)):
            if time.monotonic() - t0 > self.timeout_s:
                raise TimeoutError(f"{m} not ready after {self.timeout_s}s")
            time.sleep(self.poll_s)
        took = time.monotonic() - t0
        with self.lock:
            self.measured.setdefault(m, []).append(took)
        return took

    def load_estimate(self, m, now=None):
        with self.lock:
            runs = self.measured.get(m)
            return sum(runs) / len(runs) if runs else float(MODELS[m]["cold_start_s"])
