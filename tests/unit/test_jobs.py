"""The Job-based actuator for clusters without GPU Deployments (deploy/nautilus)."""

import json
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from pickspin.config import MODELS, Tier
from pickspin.live.jobs import JobActuator, load_servers, render_endpoints, render_job, render_service

REPO = Path(__file__).resolve().parents[2]
SERVERS = REPO / "deploy" / "nautilus" / "servers.json"  # the catalog the gateway serves
PAPER = REPO / "deploy" / "nautilus" / "servers-paper.json"  # the paper's nine models


@pytest.fixture
def servers():
    return load_servers(SERVERS)


def test_paper_server_file_covers_the_paper_pool_in_order():
    paper = load_servers(PAPER)
    assert list(paper["models"]) == list(MODELS)
    assert {k: s.hf_id for k, s in paper["catalog"].items()} == {k: s.hf_id for k, s in MODELS.items()}


@pytest.mark.parametrize("path", [SERVERS, PAPER], ids=["servers", "paper"])
def test_server_files_name_every_server_and_cover_every_tier(path):
    servers = load_servers(path)
    names = [spec["name"] for spec in servers["models"].values()]
    assert len(set(names)) == len(names)
    assert all(name.startswith("bhanu-pickspin-") for name in names)
    assert {spec.tier for spec in servers["catalog"].values()} == set(Tier)


def test_new_model_needs_its_figures(tmp_path):
    bad = tmp_path / "servers.json"
    bad.write_text(json.dumps({"defaults": {}, "models": {"mistral_7B": {"name": "x"}}}), encoding="utf-8")
    with pytest.raises(ValueError, match="mistral_7B"):
        load_servers(bad)


def test_models_can_be_replaced(tmp_path):
    custom = tmp_path / "servers.json"
    custom.write_text(
        json.dumps(
            {
                "defaults": {"gpus": 1},
                "models": {
                    "mistral_7B": {
                        "name": "m7",
                        "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
                        "tier": "MEDIUM",
                        "weight_gb": 15,
                        "cold_start_s": 45,
                        "label": "Mistral-7B",
                    },
                    "llama3.2_1B": {"name": "l1", "gpus": 2},
                },
            }
        ),
        encoding="utf-8",
    )
    servers = load_servers(custom)
    catalog = servers["catalog"]
    assert list(catalog) == ["mistral_7B", "llama3.2_1B"]
    assert catalog["mistral_7B"].tier == "MEDIUM"
    assert catalog["mistral_7B"].hf_id == "mistralai/Mistral-7B-Instruct-v0.3"
    assert catalog["llama3.2_1B"].hf_id == MODELS["llama3.2_1B"].hf_id
    assert catalog["llama3.2_1B"].gpus == 2
    assert render_endpoints(servers)["mistral_7B"]["model"] == "mistralai/Mistral-7B-Instruct-v0.3"
    assert (
        "--model=mistralai/Mistral-7B-Instruct-v0.3"
        in render_job(
            "mistral_7B",
            servers["models"]["mistral_7B"]
            | {
                "memory": "1Gi",
                "cpu": 1,
                "max_model_len": 4096,
                "gpu_memory_utilization": 0.9,
                "image": "x",
                "hf_token_secret": "s",
                "cache_pvc": "c",
                "active_deadline_s": 60,
                "gpu_products": [],
            },
        )["spec"]["template"]["spec"]["containers"][0]["args"]
    )


@pytest.mark.parametrize("path", [SERVERS, PAPER], ids=["servers", "paper"])
def test_job_follows_nrp_rules(path):
    servers = load_servers(path)
    for key, spec in servers["models"].items():
        job = render_job(key, spec)
        assert job["kind"] == "Job"
        assert job["metadata"]["name"] == spec["name"]
        pod = job["spec"]["template"]["spec"]
        container = pod["containers"][0]
        res = container["resources"]
        assert res["requests"] == res["limits"]  # NRP: limits within 20% of requests
        assert res["limits"]["nvidia.com/gpu"] == spec["gpus"]
        assert f"--model={servers['catalog'][key].hf_id}" in container["args"]
        assert f"--tensor-parallel-size={spec['gpus']}" in container["args"]
        assert job["spec"]["activeDeadlineSeconds"] > 0
        assert "sleep" not in json.dumps(job)
        products = pod["affinity"]["nodeAffinity"]["requiredDuringSchedulingIgnoredDuringExecution"][
            "nodeSelectorTerms"
        ][0]["matchExpressions"][0]["values"]
        assert products == spec["gpu_products"]
        env = {e["name"]: e.get("value") for e in container["env"]}
        assert spec.get("env", {}).items() <= env.items()


def test_service_selects_the_job_pod(servers):
    key, spec = next(iter(servers["models"].items()))
    service = render_service(key, spec)
    pod_labels = render_job(key, spec)["spec"]["template"]["metadata"]["labels"]
    assert service["spec"]["selector"].items() <= pod_labels.items()


def test_endpoints_point_at_the_services():
    paper = render_endpoints(load_servers(PAPER))
    assert paper["gemma3_27B"]["base_url"] == "http://bhanu-pickspin-gemma3-27b:8000"
    assert paper["gemma3_27B"]["deployment"] == "bhanu-pickspin-gemma3-27b"
    for path, rendered in [
        ("endpoints.json", render_endpoints(load_servers(SERVERS))),
        ("endpoints-paper.json", paper),
    ]:
        assert json.loads((REPO / "deploy" / "nautilus" / path).read_text(encoding="utf-8")) == rendered
    services = json.loads((REPO / "deploy" / "nautilus" / "services.json").read_text(encoding="utf-8"))["items"]
    named = {service["metadata"]["name"] for service in services}
    assert named == {e["deployment"] for e in [*paper.values(), *render_endpoints(load_servers(SERVERS)).values()]}


class ApiException(Exception):  # noqa: N818 - mirrors kubernetes.client.exceptions.ApiException
    def __init__(self, status):
        super().__init__(status)
        self.status = status


class FakeBatch:
    def __init__(self):
        self.jobs = {}
        self.ended = {}  # Job name -> the condition it ended with
        self.calls = []

    def read_namespaced_job(self, name, namespace):
        if name not in self.jobs:
            raise ApiException(404)
        ended = self.ended.get(name)
        return SimpleNamespace(
            metadata=SimpleNamespace(deletion_timestamp=None),
            status=SimpleNamespace(conditions=[ended] if ended else None),
        )

    def create_namespaced_job(self, namespace, body):
        self.calls.append(("create", body["metadata"]["name"]))
        self.jobs[body["metadata"]["name"]] = body

    def delete_namespaced_job(self, name, namespace, propagation_policy):
        self.calls.append(("delete", name))
        self.ended.pop(name, None)
        if self.jobs.pop(name, None) is None:
            raise ApiException(404)


class FakeCore:
    """Lists the pods that pods_of(job body) gives for the Job named in the label selector."""

    def __init__(self, batch, pods_of):
        self.batch = batch
        self.pods_of = pods_of

    def list_namespaced_pod(self, namespace, label_selector):
        body = self.batch.jobs.get(label_selector.split("=", 1)[1])
        return SimpleNamespace(items=self.pods_of(body) if body else [])

    def read_namespaced_pod_log(self, name, namespace, tail_lines):
        return f"last {tail_lines} lines of {name}: CUDA error: no kernel image is available"


def pod(phase="Pending", *, ready=False, unschedulable=False, terminated=None):
    conditions = []
    if unschedulable:
        conditions.append(SimpleNamespace(type="PodScheduled", status="False", reason="Unschedulable"))
    if ready:
        conditions.append(SimpleNamespace(type="Ready", status="True", reason=None, last_transition_time=None))
    states = [SimpleNamespace(state=SimpleNamespace(terminated=terminated, running=None))] if terminated else []
    return SimpleNamespace(
        metadata=SimpleNamespace(name="pod-0", deletion_timestamp=None, creation_timestamp=None),
        status=SimpleNamespace(phase=phase, conditions=conditions, container_statuses=states, reason=None),
    )


def gpus_of(body):
    return body["spec"]["template"]["spec"]["containers"][0]["resources"]["limits"]["nvidia.com/gpu"]


def make_actuator(servers, batch, pods_of=lambda body: []):
    actuator = JobActuator.__new__(JobActuator)
    actuator._api_exception = ApiException
    actuator.batch = batch
    actuator.core = FakeCore(batch, pods_of)
    actuator.servers = servers
    actuator.endpoints = render_endpoints(servers)
    actuator.namespace = "ns"
    actuator.poll_s = 0.0
    actuator.timeout_s = 1.0
    actuator.measured = {}
    actuator.phases = {}
    actuator.placement = {}
    actuator._created = {}
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


def test_scale_replaces_a_job_that_has_ended(servers):
    batch = FakeBatch()
    actuator = make_actuator(servers, batch)
    name = servers["models"]["qwen2.5_7B"]["name"]
    actuator.scale("qwen2.5_7B", 1)
    assert actuator.alive("qwen2.5_7B")
    batch.ended[name] = SimpleNamespace(type="Failed", status="True", reason="DeadlineExceeded")
    assert actuator.failure("qwen2.5_7B") == "its Job failed (DeadlineExceeded)"
    assert not actuator.alive("qwen2.5_7B")
    actuator.scale("qwen2.5_7B", 1)  # the ended Job is replaced, not adopted
    assert batch.calls == [("create", name), ("delete", name), ("create", name)]
    assert actuator.alive("qwen2.5_7B")


def test_wait_ready_fails_as_soon_as_the_server_stops_and_logs_its_last_lines(servers, caplog):
    oom = SimpleNamespace(reason="OOMKilled", exit_code=137)
    actuator = make_actuator(servers, FakeBatch(), lambda body: [pod("Failed", terminated=oom)])
    actuator.scale("llama3.2_1B", 1)
    with pytest.raises(RuntimeError, match=r"its pod stopped: OOMKilled \(exit code 137\)"):
        actuator.wait_ready("llama3.2_1B", time.monotonic())
    assert actuator.measured == {}
    assert "last 30 lines of pod-0: CUDA error" in caplog.text
    assert "llama3.2_1B" not in actuator.placement  # a model with one placement stays in it


def test_a_model_that_finds_no_node_moves_to_its_next_placement(tmp_path):
    big, small = ["NVIDIA-L40S"], ["NVIDIA-GeForce-RTX-3090"]
    custom = tmp_path / "servers.json"
    defaults = json.loads(SERVERS.read_text(encoding="utf-8"))["defaults"]
    entry = {
        "name": "q14",
        "memory": "32Gi",
        "schedule_patience_s": 0,
        "placements": [{"gpus": 1, "gpu_products": big}, {"gpus": 2, "gpu_products": small, "env": {"X": "1"}}],
    }
    custom.write_text(json.dumps({"defaults": defaults, "models": {"qwen2.5_14B": entry}}), encoding="utf-8")
    servers = load_servers(custom)
    assert servers["catalog"]["qwen2.5_14B"].gpus == 1  # the first placement is the one counted
    assert servers["models"]["qwen2.5_14B"]["gpu_products"] == big

    # One GPU of the big kind never gets a node; two of the small kind are ready at once.
    batch = FakeBatch()
    actuator = make_actuator(
        servers, batch, lambda body: [pod(unschedulable=True)] if gpus_of(body) == 1 else [pod("Running", ready=True)]
    )
    actuator.healthy = lambda model: True
    actuator.scale("qwen2.5_14B", 1)
    actuator.wait_ready("qwen2.5_14B", time.monotonic())

    assert batch.calls == [("create", "q14"), ("delete", "q14"), ("create", "q14")]
    job = batch.jobs["q14"]
    assert gpus_of(job) == 2
    assert "--tensor-parallel-size=2" in job["spec"]["template"]["spec"]["containers"][0]["args"]
    assert {"name": "X", "value": "1"} in job["spec"]["template"]["spec"]["containers"][0]["env"]
    assert actuator.placement["qwen2.5_14B"] == 1
    assert len(actuator.measured["qwen2.5_14B"]) == 1


def test_load_estimate_uses_measurements(servers):
    actuator = make_actuator(servers, FakeBatch())
    assert actuator.load_estimate("llama3.2_1B") == servers["catalog"]["llama3.2_1B"].cold_start_s
    actuator.measured["llama3.2_1B"] = [10.0, 20.0]
    assert actuator.load_estimate("llama3.2_1B") == 15.0
