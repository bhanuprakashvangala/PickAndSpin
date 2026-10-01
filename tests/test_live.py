"""End-to-end test of the live runner against a stub vLLM server and a fake Kubernetes API."""

import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pickspin"))

import classifier  # noqa: E402
import config  # noqa: E402
import run_live  # noqa: E402
from config import MODELS  # noqa: E402

LOAD_S = 0.3


class FakeCluster:
    """Replica counts per model; a scaled-up model answers /health LOAD_S seconds later."""

    def __init__(self):
        self.lock = threading.Lock()
        self.up_since = {}
        self.calls = []

    def scale(self, m, replicas):
        with self.lock:
            self.calls.append((m, replicas))
            if replicas:
                self.up_since.setdefault(m, time.monotonic())
            else:
                self.up_since.pop(m, None)

    def ready(self, m):
        with self.lock:
            t = self.up_since.get(m)
            return t is not None and time.monotonic() - t >= LOAD_S


def make_handler(cluster, by_port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _reply(self, code, body=b"{}"):
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            m = by_port[self.server.server_address[1]]
            self._reply(200 if self.path == "/health" and cluster.ready(m) else 503)

        def do_POST(self):
            m = by_port[self.server.server_address[1]]
            self.rfile.read(int(self.headers["Content-Length"]))
            if not cluster.ready(m):
                return self._reply(503)
            time.sleep(0.01)
            self._reply(200, json.dumps({"choices": [{"message": {"content": "ok"}}],
                                         "usage": {"completion_tokens": 3}}).encode())
    return Handler


def test_live_runner_scales_models_up_and_down(tmp_path, monkeypatch):
    cluster = FakeCluster()
    servers, endpoints, by_port = [], {}, {}
    for m in MODELS:
        srv = ThreadingHTTPServer(("127.0.0.1", 0), None)
        by_port[srv.server_address[1]] = m
        servers.append(srv)
        endpoints[m] = {"base_url": f"http://127.0.0.1:{srv.server_address[1]}", "model": m, "deployment": m}
    handler = make_handler(cluster, by_port)
    for srv in servers:
        srv.RequestHandlerClass = handler
        threading.Thread(target=srv.serve_forever, daemon=True).start()

    class FakeActuator:
        def __init__(self, endpoints, namespace):
            self.measured = {}

        def scale(self, m, replicas):
            cluster.scale(m, replicas)

        def ready_replicas(self, m):
            return int(cluster.ready(m))

        def healthy(self, m):
            return cluster.ready(m)

        def wait_ready(self, m, t0):
            while not cluster.ready(m):
                time.sleep(0.02)
            self.measured.setdefault(m, []).append(time.monotonic() - t0)

        def load_estimate(self, m, now=None):
            return LOAD_S

    ep_file = tmp_path / "endpoints.json"
    ep_file.write_text(json.dumps(endpoints))
    monkeypatch.setattr(run_live, "KubernetesActuator", FakeActuator)
    monkeypatch.setattr(run_live, "HybridClassifier", lambda: classifier.HybridClassifier(distilbert=None))
    monkeypatch.setitem(config.SPIN, "cooldown_s", 0.5)

    try:
        stem = run_live.main(["--endpoints", str(ep_file), "--workers", "8", "--limit", "60",
                              "--out", str(tmp_path)])
    finally:
        for srv in servers:
            srv.shutdown()
    rows = [json.loads(line) for line in open(f"{stem}.jsonl", encoding="utf-8")]
    summary = json.load(open(f"{stem}_summary.json", encoding="utf-8"))
    assert len(rows) == 60 and all(r["success"] for r in rows)
    cold = [r for r in rows if r["cold_start"]]
    assert cold and all(r["wait_s"] >= LOAD_S * 0.9 for r in cold)
    assert summary["cold_starts"] == len(cold)
    assert all(replicas == 0 for _, replicas in cluster.calls[:len(MODELS)])   # the run starts cold
    assert any(replicas == 1 for _, replicas in cluster.calls)
