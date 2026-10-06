"""Fixtures for the integration tests: HTTP stubs on localhost and a stub cluster for live runs.

StubServer is an HTTP server on a free 127.0.0.1 port that records every request and answers with
whatever its responder returns; the http_stub fixture starts such servers and shuts them down after
the test.

The stub_cluster fixture, ported from the v1.1.0 test of the live runner, stands in for the
Kubernetes deployment of a live run: one stub vLLM server per model, a FakeCluster that keeps the
replica count of every model, and a FakeClusterActuator that scales models on it the way
KubernetesActuator scales Deployments. A model scaled to one replica answers /health and chat
completions LOAD_S seconds later, and HTTP 503 until then.
"""

from __future__ import annotations

import functools
import json
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import pytest
from requests.structures import CaseInsensitiveDict

from pickspin.config import MODELS
from pickspin.live.vllm import Endpoint

LOAD_S = 0.3  # seconds from a scale-up until a stub model is ready
INFER_S = 0.01  # seconds a stub model takes to answer a chat request

# The reply of a stub vLLM server to every chat request it serves.
CHAT_REPLY: dict[str, Any] = {"choices": [{"message": {"content": "ok"}}], "usage": {"completion_tokens": 3}}


@dataclass(frozen=True)
class Request:
    """One request a stub server received; body is the decoded JSON body, or None without a body."""

    method: str
    path: str
    headers: CaseInsensitiveDict[str]
    body: Any


# A responder returns the HTTP status and the body to send as JSON (None sends an empty body).
Reply = tuple[int, Any]
Responder = Callable[[Request], Reply]


class StubServer:
    """An HTTP server on a free 127.0.0.1 port that answers every request with responder(request).

    It runs in a daemon thread until close(). received lists the requests in the order they arrived.
    """

    def __init__(self, responder: Responder) -> None:
        self.responder = responder
        self._received: list[Request] = []
        self._lock = threading.Lock()
        self._httpd = _StubHTTPServer(self)
        # A short poll interval keeps close() quick: shutdown() waits until the loop polls.
        self._thread = threading.Thread(target=self._httpd.serve_forever, args=(0.05,), daemon=True)
        self._thread.start()

    @property
    def port(self) -> int:
        """The port the server listens on."""
        return int(self._httpd.server_address[1])

    @property
    def base_url(self) -> str:
        """The server's URL, without a trailing slash."""
        return f"http://127.0.0.1:{self.port}"

    @property
    def received(self) -> list[Request]:
        """A copy of the requests received so far, in arrival order."""
        with self._lock:
            return list(self._received)

    def answer(self, request: Request) -> Reply:
        """Record the request and return the responder's reply; called from the server's threads."""
        with self._lock:
            self._received.append(request)
        return self.responder(request)

    def close(self) -> None:
        """Stop serving and release the port."""
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join()


class _StubHTTPServer(ThreadingHTTPServer):
    """The server behind a StubServer: one daemon thread per request."""

    def __init__(self, stub: StubServer) -> None:
        super().__init__(("127.0.0.1", 0), _StubHandler)
        self.stub = stub


class _StubHandler(BaseHTTPRequestHandler):
    """Passes every GET and POST to the StubServer and sends back its reply."""

    server: _StubHTTPServer

    def do_GET(self) -> None:
        self._handle(None)

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self._handle(json.loads(self.rfile.read(length)) if length else None)

    def _handle(self, body: Any) -> None:
        request = Request(self.command, self.path, CaseInsensitiveDict(self.headers.items()), body)
        status, reply = self.server.stub.answer(request)
        data = b"" if reply is None else json.dumps(reply).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format: str, *args: Any) -> None:
        """Keep the access log out of the test output."""


class FakeCluster:
    """The replica count of every model, as the Kubernetes API would report it.

    A model scaled to one replica becomes ready load_s seconds later, and scaling it to zero makes it
    unready at once. calls lists every scale(model, replicas) in the order it was made.
    """

    def __init__(self, load_s: float = LOAD_S) -> None:
        self.load_s = load_s
        self.calls: list[tuple[str, int]] = []
        self._up_since: dict[str, float] = {}
        self._lock = threading.Lock()

    def scale(self, model: str, replicas: int) -> None:
        """Set the model's replica count; a model that is already up keeps its start time."""
        with self._lock:
            self.calls.append((model, replicas))
            if replicas:
                self._up_since.setdefault(model, time.monotonic())
            else:
                self._up_since.pop(model, None)

    def ready(self, model: str) -> bool:
        """True once the model has been up for load_s seconds."""
        with self._lock:
            since = self._up_since.get(model)
            return since is not None and time.monotonic() - since >= self.load_s

    def start_all(self) -> None:
        """Give every model a ready replica now, as a static deployment (Helm model-servers.startReplicas=1) has.

        This stands for the deployment, not for an actuator, so it is not recorded in calls.
        """
        with self._lock:
            ready_since = time.monotonic() - self.load_s
            for model in MODELS:
                self._up_since[model] = ready_since


class FakeClusterActuator:
    """Implements the Actuator protocol on a FakeCluster, as KubernetesActuator does on a real cluster.

    wait_ready polls until the model is ready, then records the load time in measured and returns it.
    load_estimate always gives the cluster's load time.
    """

    def __init__(self, cluster: FakeCluster, poll_s: float = 0.02) -> None:
        self.cluster = cluster
        self.poll_s = poll_s
        self.measured: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def scale(self, model: str, replicas: int) -> None:
        """Set the model's replica count on the cluster."""
        self.cluster.scale(model, replicas)

    def ready_replicas(self, model: str) -> int:
        """Return 1 if the model's replica is ready, else 0."""
        return int(self.cluster.ready(model))

    def wait_ready(self, model: str, t0: float) -> float:
        """Block until the model is ready; t0 is the time.monotonic() reading when its load started."""
        while not self.ready_replicas(model):
            time.sleep(self.poll_s)
        took = time.monotonic() - t0
        with self._lock:
            self.measured.setdefault(model, []).append(took)
        return took

    def load_estimate(self, model: str, now: float | None = None) -> float:
        """Return the cluster's load time, whatever the model."""
        return self.cluster.load_s


def vllm_reply(cluster: FakeCluster, model: str, served_name: str, request: Request) -> Reply:
    """What the stub vLLM server of a model answers.

    GET /health gives 200 once the model is ready and 503 before. A chat completion request gives
    CHAT_REPLY after INFER_S seconds once the model is ready, and 503 before. As in vLLM, a request
    to another path, or for a model name the server does not serve, gives 404.
    """
    if request.method == "GET":
        return (200 if request.path == "/health" and cluster.ready(model) else 503), {}
    asked_for = (request.body or {}).get("model")
    if request.path != "/v1/chat/completions" or asked_for != served_name:
        return 404, {"error": f"{request.path} does not serve {asked_for!r}"}
    if not cluster.ready(model):
        return 503, {}
    time.sleep(INFER_S)
    return 200, CHAT_REPLY


@dataclass(frozen=True)
class StubCluster:
    """A stub cluster for a live run: the cluster, its actuator, the model servers and the endpoint map.

    servers and endpoints are keyed by model, in MODELS order, and endpoints_file holds the endpoint
    map as JSON, for LiveConfig.endpoints. Every server serves its model under the model's Hugging
    Face id.
    """

    cluster: FakeCluster
    actuator: FakeClusterActuator
    servers: Mapping[str, StubServer]
    endpoints: Mapping[str, Endpoint]
    endpoints_file: Path

    @property
    def load_s(self) -> float:
        """Seconds from a scale-up until a model is ready."""
        return self.cluster.load_s


@pytest.fixture
def no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Send requests to 127.0.0.1 directly, even when the environment configures an HTTP proxy."""
    for name in ("no_proxy", "NO_PROXY"):
        monkeypatch.setenv(name, "127.0.0.1")


@pytest.fixture
def http_stub(no_proxy: None) -> Iterator[Callable[[Responder], StubServer]]:
    """start(responder) -> a running StubServer. Every server started is shut down after the test.

    Requests to 127.0.0.1 bypass any proxy that the environment configures.
    """
    servers: list[StubServer] = []

    def start(responder: Responder) -> StubServer:
        server = StubServer(responder)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


@pytest.fixture
def stub_cluster(http_stub: Callable[[Responder], StubServer], tmp_path: Path) -> StubCluster:
    """A stub cluster with every model at zero replicas, and its endpoint map in tmp_path/endpoints.json.

    The servers are shut down after the test.
    """
    cluster = FakeCluster()
    servers = {m: http_stub(functools.partial(vllm_reply, cluster, m, spec.hf_id)) for m, spec in MODELS.items()}
    endpoints: dict[str, Endpoint] = {
        m: {"base_url": server.base_url, "model": MODELS[m].hf_id, "deployment": m} for m, server in servers.items()
    }
    endpoints_file = tmp_path / "endpoints.json"
    endpoints_file.write_text(json.dumps(endpoints, indent=1), encoding="utf-8")
    return StubCluster(cluster, FakeClusterActuator(cluster), servers, endpoints, endpoints_file)
