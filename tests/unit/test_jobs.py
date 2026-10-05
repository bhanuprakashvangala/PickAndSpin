"""The Job-based actuator for clusters without GPU Deployments (deploy/nautilus)."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pickspin.config import MODELS
from pickspin.live.jobs import JobActuator, load_servers, render_endpoints, render_job, render_service

REPO = Path(__file__).resolve().parents[2]
SERVERS = REPO / "deploy" / "nautilus" / "servers.json"


@pytest.fixture
def servers():
    return load_servers(SERVERS)


def test_server_file_covers_the_catalog_in_order(servers):
    assert list(servers["models"]) == list(MODELS)
    names = [spec["name"] for spec in servers["models"].values()]
    assert len(set(names)) == len(names)
    assert all(name.startswith("bhanu-pickspin-") for name in names)


def test_unknown_model_key_is_rejected(tmp_path):
    bad = tmp_path / "servers.json"
    bad.write_text(json.dumps({"defaults": {}, "models": {"gpt5": {"name": "x"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="gpt5"):
        load_servers(bad)


def test_job_follows_nrp_rules(servers):
    for key, spec in servers["models"].items():
        job = render_job(key, spec)
        assert job["kind"] == "Job"
        assert job["metadata"]["name"] == spec["name"]
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        res = container["resources"]
        assert res["requests"] == res["limits"]  # NRP: limits within 20% of requests
        assert res["limits"]["nvidia.com/gpu"] == spec["gpus"]
        assert f"--model={MODELS[key].hf_id}" in container["args"]
        assert f"--tensor-parallel-size={spec['gpus']}" in container["args"]
        assert job["spec"]["activeDeadlineSeconds"] > 0
        assert "sleep" not in json.dumps(job)
        products = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"
        ][0]["matchExpressions"][0]["values"]
        assert products == spec["gpu_products"]


def test_service_selects_the_job_pod(servers):
    key, spec = next(iter(servers["models"].items()))
    service = render_service(key, spec)
    pod_labels = render_job(key, spec)["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= pod_labels.items()


def test_endpoints_point_at_the_services(servers):
    endpoints = render_endpoints(servers)
    assert endpoints["gemma3_27B"]["base_url"] == "http://bhanu-pickspin-gemma3-27b:8000"
    assert endpoints["gemma3_27B"]["deployment"] == "bhanu-pickspin-gemma3-27b"
    committed = json.loads((REPO / "deploy" / "nautilus" / "endpoints.json").read_text(encoding="utf-8"))
    assert committed == endpoints


class ApiException(Exception):  # noqa: N818 - mirrors kubernetes.client.exceptions.ApiException
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class FakeBatch:
    def __init__(self):
        self.jobs = {}
        self.calls = []

    def read_namespaced_job(self, name, namespace):
        if name not in self.jobs:
            raise ApiException(404)
        return SimpleNamespace(metadata=SimpleNamespace(deletion_timestamp=None))

    def create_namespaced_job(self, namespace, body):
        self.calls.append(("create", body["metadata"]["name"]))
        self.jobs[body["metadata"]["name"]] = body

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        self.calls.append(("delete", name))
        if self.jobs.pop(name, None) is None:
            raise ApiException(404)


def make_actuator(servers, batch):
    actuator = JobActuator.__new__(JobActuator)
    actuator._api_exception = ApiException
    actuator.batch = batch
    actuator.core = None
    actuator.servers = servers
    actuator.endpoints = render_endpoints(servers)
    actuator.namespace = "ns"
    actuator.poll_s = 0.0
    actuator.timeout_s = 1.0
    actuator.measured = {}
    import threading

    actuator._lock = threading.Lock()
    return actuator


def test_scale_creates_once_and_deletes(servers):
    batch = FakeBatch()
    actuator = make_actuator(servers, batch)
    actuator.scale("qwen2.5_7B", 1)
    actuator.scale("qwen2.5_7B", 1)  # already running: no second Job
    actuator.scale("qwen2.5_7B", 0)
    actuator.scale("qwen2.5_7B", 0)  # already gone: no error
    name = servers["models"]["qwen2.5_7B"]["name"]
    assert batch.calls == [("create", name), ("delete", name), ("delete", name)]


def test_load_estimate_uses_measurements(servers):
    actuator = make_actuator(servers, FakeBatch())
    assert actuator.load_estimate("llama3.2_1B") == MODELS["llama3.2_1B"].cold_start_s
    actuator.measured["llama3.2_1B"] = [10.0, 20.0]
    assert actuator.load_estimate("llama3.2_1B") == 15.0
