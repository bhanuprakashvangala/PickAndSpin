"""Spin's actuator for clusters that run GPU servers as Jobs (such as NRP Nautilus).

Some clusters do not allow long-running Deployments to request GPUs. There, each model's vLLM server
runs as a Kubernetes Job behind a permanent Service: scaling a model up creates its Job, and scaling it
to zero deletes that Job (and its pod). JobActuator implements the same Actuator protocol as the
Deployment-based KubernetesActuator, so the live runner does not change.

The server of each model is described in a JSON file (deploy/nautilus/servers.json):

    {"defaults": {...}, "models": {"<model key>": {"name": ..., "gpus": ..., "gpu_products": [...], ...}}}

A model's entry replaces a default field as a whole (an "env" map in a model replaces the default one).
"extra_args" are appended to the vLLM arguments and "env" adds environment variables to the server.

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

    models maps each key to its server spec merged over the defaults. catalog maps each key to a
    ModelSpec: a key of the paper's pool takes its figures from pickspin.config.MODELS unless the entry
    overrides them, and any other key must give hf_id, tier, weight_gb and cold_start_s itself (label
    defaults to the key). gpus always comes from the server spec, since it is what the server holds.
    The order of the file is the catalog order.
    """
    with path.open(encoding="utf-8") as f:
        raw = json.load(f)
    defaults = raw.get("defaults", {})
    models = {key: {**defaults, **spec} for key, spec in raw["models"].items()}
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


class JobActuator:
    """Brings a model up by creating its vLLM Job and down by deleting it.

    Implements the Actuator protocol of pickspin.live.actuator. Only the Jobs named in the server file
    are ever created or deleted. wait_ready polls every poll_s seconds and gives up with TimeoutError
    after timeout_s; the load times it measures are kept in measured. phases keeps, for every load, how
    the time split between waiting for a node (scheduled_s), starting the container including the image
    pull (container_start_s) and starting vLLM, including reading the weights (server_start_s), taken
    from the pod's own timestamps.
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
        self._lock = threading.Lock()

    def _name(self, model: str) -> str:
        name: str = self.servers["models"][model]["name"]
        return name

    def _job_exists(self, name: str) -> bool:
        try:
            job = self.batch.read_namespaced_job(name, self.namespace)
        except self._api_exception as e:
            if e.status == 404:
                return False
            raise
        return job.metadata.deletion_timestamp is None

    def scale(self, model: str, replicas: int) -> None:
        """Create the model's Job for replicas >= 1; delete it (and its pod) for replicas == 0."""
        name = self._name(model)
        if replicas > 0:
            if self._job_exists(name):
                return
            # A Job that is still being deleted blocks a new one with the same name.
            while True:
                try:
                    self.batch.create_namespaced_job(self.namespace, render_job(model, self.servers["models"][model]))
                    log.info("created Job %s", name)
                    return
                except self._api_exception as e:
                    if e.status != 409:
                        raise
                    time.sleep(self.poll_s)
        try:
            self.batch.delete_namespaced_job(name, self.namespace, propagation_policy="Foreground")
            log.info("deleted Job %s", name)
        except self._api_exception as e:
            if e.status != 404:
                raise

    def ready_replicas(self, model: str) -> int:
        """Return the number of ready pods of the model's Job."""
        pods = self.core.list_namespaced_pod(self.namespace, label_selector=f"job-name={self._name(model)}").items
        return sum(
            1
            for pod in pods
            if pod.metadata.deletion_timestamp is None
            and any(c.type == "Ready" and c.status == "True" for c in (pod.status.conditions or []))
        )

    def healthy(self, model: str) -> bool:
        """Return True if the model's vLLM server answers /health with HTTP 200 within 5 seconds."""
        try:
            return requests.get(self.endpoints[model]["base_url"].rstrip("/") + "/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, model: str, t0: float) -> float:
        """Block until the model's pod is ready and vLLM answers /health; return and record the load time."""
        while not (self.ready_replicas(model) >= 1 and self.healthy(model)):
            if time.monotonic() - t0 > self.timeout_s:
                raise TimeoutError(f"{model} not ready after {self.timeout_s}s")
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
        pods = self.core.list_namespaced_pod(self.namespace, label_selector=f"job-name={self._name(model)}").items
        for pod in pods:
            if pod.metadata.deletion_timestamp is not None:
                continue
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
            }
        return {}

    def load_estimate(self, model: str, now: float | None = None) -> float:
        """Return the mean measured load time of the model, or its stated cold-start time before any."""
        with self._lock:
            runs = self.measured.get(model)
            return sum(runs) / len(runs) if runs else float(self.servers["catalog"][model].cold_start_s)
