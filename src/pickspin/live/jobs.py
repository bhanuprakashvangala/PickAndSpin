"""Spin's actuator for clusters that run GPU servers as Jobs (such as NRP Nautilus).

Some clusters do not allow long-running Deployments to request GPUs. There, each model's vLLM server
runs as a Kubernetes Job behind a permanent Service: scaling a model up creates its Job, and scaling it
to zero deletes that Job (and its pod). JobActuator implements the same Actuator protocol as the
Deployment-based KubernetesActuator, so the live runner does not change.

The server of each model is described in a JSON file (deploy/nautilus/servers.json):

    {"defaults": {...}, "models": {"<model key>": {"name": ..., "gpus": ..., "gpu_products": [...], ...}}}

A model's entry replaces a default field as a whole (an "env" map in a model replaces the default one).
"extra_args" are appended to the vLLM arguments and "env" adds environment variables to the server.

"placements" lists other ways to run a model, each a partial entry (such as {"gpus": 2, "gpu_products":
[...]}) applied over the model's own. The first is the one rendered and counted in the catalog. When the
pod of a model waits longer than schedule_patience_s for a node, JobActuator moves the model to its next
placement, and a failed load also moves it on for the next try.

render_job and render_service turn one entry into Kubernetes manifests; render_all writes every
manifest for `kubectl apply`, for example to start all servers for a static run.
"""

import json
import logging
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import requests

from pickspin.config import MODELS, ModelSpec, Tier
from pickspin.errors import import_optional
from pickspin.live.vllm import Endpoint

log = logging.getLogger(__name__)

PORT = 8000
LABELS = {"app": "pickspin-server"}


_SPEC_FIELDS = ("hf_id", "tier", "weight_gb", "cold_start_s")


def load_servers(path: Path) -> dict[str, Any]:
    """Read a server file and return {"defaults", "models", "catalog"}.

    models maps each key to its server spec merged over the defaults, with its first placement applied
    if it lists placements. catalog maps each key to a
    ModelSpec: a key of the paper's pool takes its figures from pickspin.config.MODELS unless the entry
    overrides them, and any other key must give hf_id, tier, weight_gb and cold_start_s itself (label
    defaults to the key). gpus always comes from the server spec, since it is what the server holds.
    The order of the file is the catalog order.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    defaults = raw.get("defaults", {})
    models = {}
    for key, spec in raw["models"].items():
        merged = {**defaults, **spec}
        models[key] = {**merged, **merged["placements"][0]} if merged.get("placements") else merged
    return {"defaults": defaults, "models": models, "catalog": build_catalog(models, path)}


def build_catalog(models: Mapping[str, Mapping[str, Any]], source: Path | str = "server file") -> dict[str, ModelSpec]:
    """Return the ModelSpec of every model in server specs, in their order (see load_servers)."""
    catalog: dict[str, ModelSpec] = {}
    for key, spec in models.items():
        base = MODELS.get(key)
        missing = [f for f in _SPEC_FIELDS if f not in spec] if base is None else []
        if missing:
            raise ValueError(f"model {key!r} in {source} is not in the paper's pool and lacks {', '.join(missing)}")
        catalog[key] = ModelSpec(
            key=key,
            label=str(spec.get("label", base.label if base else key)),
            tier=Tier(spec.get("tier", base.tier if base else "")),
            hf_id=str(spec.get("hf_id", base.hf_id if base else "")),
            weight_gb=int(spec.get("weight_gb", base.weight_gb if base else 0)),
            cold_start_s=int(spec.get("cold_start_s", base.cold_start_s if base else 0)),
            gpus=int(spec.get("gpus", base.gpus if base else 1)),
        )
    return catalog


def render_job(key: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Job manifest that serves model `key` with vLLM according to its server spec."""
    gpus = int(spec["gpus"])
    hf_id = spec.get("hf_id") or MODELS[key].hf_id
    resources = {
        "requests": {"cpu": str(spec["cpu"]), "memory": spec["memory"], "nvidia.com/gpu": gpus},
        "limits": {"cpu": str(spec["cpu"]), "memory": spec["memory"], "nvidia.com/gpu": gpus},
    }
    args = [
        f"--model={hf_id}",
        f"--port={PORT}",
        f"--max-model-len={spec['max_model_len']}",
        f"--gpu-memory-utilization={spec['gpu_memory_utilization']}",
        f"--tensor-parallel-size={gpus}",
        *spec.get("extra_args", []),
    ]
    labels = {**LABELS, "model": spec["name"]}
    pod_spec: dict[str, Any] = {
        "restartPolicy": "Never",
        "affinity": {
            "nodeAffinity": {
                "requiredDuringSchedulingIgnoredDuringExecution": {
                    "nodeSelectorTerms": [
                        {
                            "matchExpressions": [
                                {
                                    "key": "nvidia.com/gpu.product",
                                    "operator": "In",
                                    "values": list(spec["gpu_products"]),
                                }
                            ]
                        }
                    ]
                }
            }
        },
        "containers": [
            {
                "name": "vllm",
                "image": spec["image"],
                "args": args,
                "env": [
                    {"name": "HF_HOME", "value": "/cache"},
                    {"name": "HF_HUB_OFFLINE", "value": "1"},
                    {
                        "name": "HF_TOKEN",
                        "valueFrom": {
                            "secretKeyRef": {"name": spec["hf_token_secret"], "key": "HF_TOKEN", "optional": True}
                        },
                    },
                    *({"name": name, "value": str(value)} for name, value in spec.get("env", {}).items()),
                ],
                "ports": [{"containerPort": PORT}],
                "readinessProbe": {
                    "httpGet": {"path": "/health", "port": PORT},
                    "periodSeconds": 5,
                    "failureThreshold": 600,
                },
                "resources": resources,
                "volumeMounts": [{"name": "hf-cache", "mountPath": "/cache"}, {"name": "shm", "mountPath": "/dev/shm"}],
            }
        ],
        "volumes": [
            {"name": "hf-cache", "persistentVolumeClaim": {"claimName": spec["cache_pvc"]}},
            {"name": "shm", "emptyDir": {"medium": "Memory", "sizeLimit": "8Gi"}},
        ],
    }
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": spec["name"], "labels": labels},
        "spec": {
            "backoffLimit": 0,
            "activeDeadlineSeconds": int(spec["active_deadline_s"]),
            "template": {"metadata": {"labels": labels}, "spec": pod_spec},
        },
    }


def render_service(key: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return the Service that gives model `key`'s server a stable address."""
    return {
        "apiVersion": "v1",
        "kind": "Service",
        "metadata": {"name": spec["name"], "labels": {**LABELS, "model": spec["name"]}},
        "spec": {"selector": {**LABELS, "model": spec["name"]}, "ports": [{"port": PORT, "targetPort": PORT}]},
    }


def render_endpoints(servers: Mapping[str, Any]) -> dict[str, Endpoint]:
    """Return the endpoint map for the servers: model key -> Service URL, served model and Job name."""
    return {
        key: {
            "base_url": f"http://{spec['name']}:{PORT}",
            "model": servers["catalog"][key].hf_id,
            "deployment": spec["name"],
        }
        for key, spec in servers["models"].items()
    }


def render_all(servers: Mapping[str, Any], *, jobs: bool) -> list[dict[str, Any]]:
    """Every Service, plus every Job when jobs is True, in catalog order."""
    out: list[dict[str, Any]] = []
    for key, spec in servers["models"].items():
        out.append(render_service(key, spec))
        if jobs:
            out.append(render_job(key, spec))
    return out


def _job_ended(job: Any) -> str | None:
    """Return why a Job has ended (it failed, hit its deadline, or its pod exited), or None while it runs."""
    for c in (job.status.conditions if job.status else None) or []:
        if c.type in ("Failed", "Complete") and c.status == "True":
            return f"its Job {'failed' if c.type == 'Failed' else 'completed'} ({c.reason or 'no reason given'})"
    return None


def _pod_ready(pod: Any) -> bool:
    return any(c.type == "Ready" and c.status == "True" for c in (pod.status.conditions or []))


def _pod_unschedulable(pod: Any) -> bool:
    return any(
        c.type == "PodScheduled" and c.status == "False" and c.reason == "Unschedulable"
        for c in (pod.status.conditions or [])
    )


def _pod_stopped(pod: Any) -> str | None:
    """Return why a pod has stopped (its container's reason and exit code), or None while it runs."""
    phase = pod.status.phase
    if phase not in ("Failed", "Succeeded"):
        return None
    for status in pod.status.container_statuses or []:
        terminated = status.state.terminated if status.state else None
        if terminated is not None:
            return f"its pod stopped: {terminated.reason or 'exited'} (exit code {terminated.exit_code})"
    return f"its pod stopped: {pod.status.reason or phase}"


class JobActuator:
    """Brings a model up by creating its vLLM Job and down by deleting it.

    Implements the Actuator protocol of pickspin.live.actuator. Only the Jobs named in the server file
    are ever created or deleted. wait_ready polls every poll_s seconds; it raises RuntimeError as soon
    as the server is gone for good (failure() says why) and TimeoutError after timeout_s. The load
    times it measures are kept in measured. phases keeps, for every load, how the time split between
    waiting for a node (scheduled_s), starting the container including the image pull
    (container_start_s) and starting vLLM, including reading the weights (server_start_s), taken from
    the pod's own timestamps, and the GPUs the server got. placement[m] is the index of the placement
    model m runs in next (see the module docstring).
    """

    def __init__(
        self,
        servers: Mapping[str, Any],
        endpoints: Mapping[str, Endpoint],
        namespace: str,
        poll_s: float = 2.0,
        timeout_s: float = 3600,
    ) -> None:
        k8s_client = import_optional("kubernetes.client", "live")
        k8s_config = import_optional("kubernetes.config", "live")
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            log.debug("Not running in a pod; using the local kube config")
            k8s_config.load_kube_config()
        self._api_exception = k8s_client.exceptions.ApiException
        self.batch = k8s_client.BatchV1Api()
        self.core = k8s_client.CoreV1Api()
        self.servers = servers
        self.endpoints = endpoints
        self.namespace = namespace
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.measured: dict[str, list[float]] = {}
        self.phases: dict[str, list[dict[str, float]]] = {}
        self.placement: dict[str, int] = {}
        self._created: dict[str, float] = {}  # time.monotonic() when each model's Job was created
        self._lock = threading.Lock()

    def _name(self, model: str) -> str:
        name: str = self.servers["models"][model]["name"]
        return name

    def _placements(self, model: str) -> list[Mapping[str, Any]]:
        placements: list[Mapping[str, Any]] = self.servers["models"][model].get("placements") or [{}]
        return placements

    def spec(self, model: str) -> dict[str, Any]:
        """Return the model's server spec in its current placement."""
        placements = self._placements(model)
        return {**self.servers["models"][model], **placements[self.placement.get(model, 0) % len(placements)]}

    def _next_placement(self, model: str) -> None:
        placements = self._placements(model)
        if len(placements) > 1:
            self.placement[model] = (self.placement.get(model, 0) + 1) % len(placements)
            log.info("%s moves to placement %d: %s", model, self.placement[model], placements[self.placement[model]])

    def _job_state(self, name: str) -> tuple[str, str | None]:
        """Return the Job's state, "absent", "deleting", "ended" or "running", and why it ended."""
        try:
            job = self.batch.read_namespaced_job(name, self.namespace)
        except self._api_exception as e:
            if e.status == 404:
                return "absent", None
            raise
        if job.metadata.deletion_timestamp is not None:
            return "deleting", None
        ended = _job_ended(job)
        return ("ended", ended) if ended else ("running", None)

    def _pods(self, model: str) -> list[Any]:
        """Return the pods of the model's Job that are not being deleted."""
        pods = self.core.list_namespaced_pod(self.namespace, label_selector=f"job-name={self._name(model)}").items
        return [pod for pod in pods if pod.metadata.deletion_timestamp is None]

    def scale(self, model: str, replicas: int) -> None:
        """Create the model's Job for replicas >= 1; delete it (and its pod) for replicas == 0.

        A running Job is kept, and one that has ended (its server failed or exited) is replaced.
        """
        name = self._name(model)
        if replicas <= 0:
            self._delete(name)
            return
        state, why = self._job_state(name)
        if state == "running":
            return
        if state == "ended":
            log.info("replacing Job %s: %s", name, why)
            self._delete(name)
        self._create(model)

    def _delete(self, name: str) -> None:
        try:
            self.batch.delete_namespaced_job(name, self.namespace, propagation_policy="Foreground")
            log.info("deleted Job %s", name)
        except self._api_exception as e:
            if e.status != 404:
                raise

    def _create(self, model: str) -> None:
        """Create the model's Job in its current placement, waiting while an old one is being deleted."""
        spec = self.spec(model)
        deadline = time.monotonic() + self.timeout_s
        while True:
            try:
                self.batch.create_namespaced_job(self.namespace, render_job(model, spec))
                break
            except self._api_exception as e:
                # A Job that is still being deleted blocks a new one with the same name.
                if e.status != 409 or time.monotonic() > deadline:
                    raise
                time.sleep(self.poll_s)
        self._created[model] = time.monotonic()
        log.info("created Job %s with %s GPU(s)", spec["name"], spec["gpus"])

    def ready_replicas(self, model: str) -> int:
        """Return the number of ready pods of the model's Job."""
        return sum(1 for pod in self._pods(model) if _pod_ready(pod))

    def healthy(self, model: str) -> bool:
        """Return True if the model's vLLM server answers /health with HTTP 200 within 5 seconds."""
        try:
            return requests.get(self.endpoints[model]["base_url"].rstrip("/") + "/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def failure(self, model: str, pods: list[Any] | None = None) -> str | None:
        """Return why the model's server is gone for good, or None while it may still serve.

        A server is gone once its Job has been deleted or has ended (it failed, hit its deadline or its
        pod exited), or once its pod has stopped. pods are the Job's pods if the caller has them.
        """
        state, why = self._job_state(self._name(model))
        if state == "absent":
            return "its Job was deleted"
        if state == "deleting":
            return "its Job is being deleted"
        if why is not None:
            return why
        for pod in self._pods(model) if pods is None else pods:
            stopped = _pod_stopped(pod)
            if stopped is not None:
                return stopped
        return None

    def alive(self, model: str) -> bool:
        """Return False if the model's server is gone for good (see failure)."""
        return self.failure(model) is None

    def _log_tail(self, pods: list[Any], lines: int = 30) -> None:
        """Log the last lines of a stopped server pod, so a failed load can be diagnosed once its Job is gone."""
        for pod in pods:
            if _pod_stopped(pod) is None:
                continue
            name = pod.metadata.name
            try:
                tail = self.core.read_namespaced_pod_log(name, self.namespace, tail_lines=lines)
            except self._api_exception as e:
                log.warning("Could not read the log of %s (HTTP %s)", name, e.status)
                return
            log.warning("Last lines of %s:\n%s", name, tail)
            return

    def wait_ready(self, model: str, t0: float) -> float:
        """Block until the model's pod is ready and vLLM answers /health; return and record the load time.

        Raises RuntimeError once the server is gone for good and TimeoutError after timeout_s. A model
        whose pod has waited schedule_patience_s for a node moves to its next placement, if it has
        one, and a failed load moves it on too, so that the next try uses another placement.
        """
        patience = float(self.servers["models"][model].get("schedule_patience_s", 600))
        while True:
            pods = self._pods(model)
            if any(_pod_ready(pod) for pod in pods) and self.healthy(model):
                break
            failure = self.failure(model, pods)
            if failure is not None:
                self._log_tail(pods)
                self._next_placement(model)
                raise RuntimeError(f"{model} server failed: {failure}")
            if time.monotonic() - t0 > self.timeout_s:
                raise TimeoutError(f"{model} not ready after {self.timeout_s}s")
            waited = time.monotonic() - self._created.get(model, t0)
            if len(self._placements(model)) > 1 and waited > patience and any(map(_pod_unschedulable, pods)):
                log.info("%s found no node in %.0f s", model, waited)
                self._next_placement(model)
                self._delete(self._name(model))
                self._create(model)
            time.sleep(self.poll_s)
        took = time.monotonic() - t0
        phases = self._load_phases(model)
        with self._lock:
            self.measured.setdefault(model, []).append(took)
            if phases:
                self.phases.setdefault(model, []).append(phases)
        log.info("%s ready after %.1f s %s", model, took, phases)
        return took

    def _load_phases(self, model: str) -> dict[str, float]:
        """Split the ready pod's startup into scheduling, container start and server start, in seconds."""
        for pod in self._pods(model):
            conditions = {c.type: c.last_transition_time for c in (pod.status.conditions or [])}
            statuses = pod.status.container_statuses or []
            running = statuses[0].state.running if statuses and statuses[0].state else None
            created = pod.metadata.creation_timestamp
            scheduled, ready = conditions.get("PodScheduled"), conditions.get("Ready")
            if None in (created, scheduled, ready) or running is None or running.started_at is None:
                continue
            return {
                "scheduled_s": (scheduled - created).total_seconds(),
                "container_start_s": (running.started_at - scheduled).total_seconds(),
                "server_start_s": (ready - running.started_at).total_seconds(),
                "gpus": float(self.spec(model)["gpus"]),
            }
        return {}

    def load_estimate(self, model: str, now: float | None = None) -> float:
        """Return the mean measured load time of the model, or its stated cold-start time before any."""
        with self._lock:
            runs = self.measured.get(model)
            return sum(runs) / len(runs) if runs else float(self.servers["catalog"][model].cold_start_s)
