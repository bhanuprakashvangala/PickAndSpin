"""Spin's actuator on a real cluster: scale one vLLM Deployment per model and wait until it is ready.

KubernetesActuator sets a Deployment's replica count and waits until the Deployment has a ready pod
and vLLM answers /health. It uses the in-cluster ServiceAccount when it runs in a pod, otherwise
~/.kube/config. Measured load times replace the stated per-model cold-start times in load_estimate
once available. The kubernetes client comes with the [live] extra and is imported only when an
actuator is created.

The Actuator protocol is what the live runner needs from an actuator, and what test fakes implement.
"""

import logging
import threading
import time
from collections.abc import Mapping
from typing import Any, Protocol

import requests

from pickspin.config import MODELS
from pickspin.errors import import_optional
from pickspin.live.vllm import Endpoint

log = logging.getLogger(__name__)


class Actuator(Protocol):
    """Scales model servers and reports when they are ready.

    measured maps a model key to the load times measured so far, in seconds; the live runner writes
    it into the run summary.
    """

    measured: dict[str, list[float]]

    def scale(self, model: str, replicas: int) -> None:
        """Set the number of replicas of the model's server."""
        ...

    def ready_replicas(self, model: str) -> int:
        """Return the number of the model's replicas that are ready."""
        ...

    def wait_ready(self, model: str, t0: float) -> float | None:
        """Block until the model can serve; t0 is when its load started. May return the load time."""
        ...

    def alive(self, model: str) -> bool:
        """Return False if the model's server is gone for good (it failed or was removed)."""
        ...

    def load_estimate(self, model: str, now: float | None = None) -> float:
        """Return the expected cold-start time of the model, in seconds (Spin's load estimator)."""
        ...


class KubernetesActuator:
    """Scales one vLLM Deployment per model and waits for its /health endpoint.

    endpoints maps each model key to its base_url and Deployment name (deploy/endpoints.example.json).
    wait_ready polls every poll_s seconds and gives up with TimeoutError after timeout_s. The load
    times it measures are kept in measured, and load_estimate returns their mean.

    Raises MissingDependencyError without the [live] extra. The cluster configuration is the
    in-cluster ServiceAccount when running in a pod, otherwise ~/.kube/config.
    """

    endpoints: Mapping[str, Endpoint]
    namespace: str
    poll_s: float
    timeout_s: float
    measured: dict[str, list[float]]
    apps: Any  # kubernetes.client.AppsV1Api

    def __init__(
        self,
        endpoints: Mapping[str, Endpoint],
        namespace: str = "pick-and-spin",
        poll_s: float = 2.0,
        timeout_s: float = 1200,
    ) -> None:
        k8s_client = import_optional("kubernetes.client", "live")
        k8s_config = import_optional("kubernetes.config", "live")
        try:
            k8s_config.load_incluster_config()
        except k8s_config.ConfigException:
            log.debug("Not running in a pod; using the local kube config")
            k8s_config.load_kube_config()
        self.apps = k8s_client.AppsV1Api()
        self.endpoints = endpoints
        self.namespace = namespace
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        self.measured = {}
        self._lock = threading.Lock()  # guards measured

    def scale(self, model: str, replicas: int) -> None:
        """Set the replica count of the model's Deployment."""
        self.apps.patch_namespaced_deployment_scale(
            self.endpoints[model]["deployment"], self.namespace, {"spec": {"replicas": replicas}}
        )

    def ready_replicas(self, model: str) -> int:
        """Return the number of ready pods of the model's Deployment."""
        status = self.apps.read_namespaced_deployment_status(self.endpoints[model]["deployment"], self.namespace).status
        ready: int = status.ready_replicas or 0  # None while no pod is ready
        return ready

    def healthy(self, model: str) -> bool:
        """Return True if the model's vLLM server answers /health with HTTP 200 within 5 seconds."""
        try:
            return requests.get(self.endpoints[model]["base_url"].rstrip("/") + "/health", timeout=5).status_code == 200
        except requests.RequestException:
            return False

    def wait_ready(self, model: str, t0: float) -> float:
        """Block until the model's Deployment has a ready pod and vLLM answers /health.

        t0 is the time.monotonic() reading when the load started. Returns the load time, time.monotonic()
        - t0, after recording it in measured. Raises TimeoutError once more than timeout_s seconds have
        passed since t0 and the model is still not ready.
        """
        while not (self.ready_replicas(model) >= 1 and self.healthy(model)):
            if time.monotonic() - t0 > self.timeout_s:
                raise TimeoutError(f"{model} not ready after {self.timeout_s}s")
            time.sleep(self.poll_s)
        took = time.monotonic() - t0
        with self._lock:
            self.measured.setdefault(model, []).append(took)
        log.debug("%s ready after %.1f s", model, took)
        return took

    def alive(self, model: str) -> bool:
        """Return True: a Deployment replaces a failed pod by itself, so its server is never gone for good."""
        return True

    def load_estimate(self, model: str, now: float | None = None) -> float:
        """Return the mean measured load time of the model, or its stated cold-start time before any.

        now is not used; it is there so that the method can serve as Spin's load estimator.
        """
        with self._lock:
            runs = self.measured.get(model)
            return sum(runs) / len(runs) if runs else float(MODELS[model].cold_start_s)
