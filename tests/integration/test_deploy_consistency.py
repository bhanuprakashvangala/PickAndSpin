"""The deployment files agree with the model catalog in pickspin.config and with each other.

The live runner reads deploy/endpoints.example.json: its keys must be the catalog keys in catalog
order, and each 'model' the Hugging Face id that the model's vLLM server serves. The Helm umbrella
chart shares one model catalog (global.models) between its subcharts: model-servers deploys one vLLM
Deployment and Service per model, which must cover exactly the catalog, serve the same Hugging Face
ids on the same number of GPUs, and use the names and port that the endpoint map points the runner
at; gateway runs `pickspin serve` on the endpoint map it builds from the same catalog. The router Job
must run a `pickspin live` command line that parses, with an endpoint map that exists in the image,
whose working directory mirrors the repository root.
"""

import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pytest

from pickspin.cli.main import build_parser
from pickspin.config import MODELS
from pickspin.live.vllm import Endpoint, load_endpoints


@pytest.fixture(scope="module")
def deploy(repo_root: Path) -> Path:
    """The deploy/ directory; skips where it is absent, as in an sdist."""
    path = repo_root / "deploy"
    if not path.is_dir():
        pytest.skip("deploy/ is not in this tree")
    return path


@pytest.fixture(scope="module")
def endpoints(deploy: Path) -> dict[str, Endpoint]:
    """The example endpoint map, read as the live runner reads it."""
    return load_endpoints(deploy / "endpoints.example.json")


def load_yaml(path: Path) -> Any:
    """Parse a YAML file, skipping the test where PyYAML is not installed (it comes with the dev extra)."""
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_endpoint_map_lists_the_catalog_in_order(endpoints):
    assert list(endpoints) == list(MODELS)
    assert {key: endpoint["model"] for key, endpoint in endpoints.items()} == {
        key: spec.hf_id for key, spec in MODELS.items()
    }


CHART = Path("helm") / "pick-and-spin"


def test_helm_values_deploy_every_model_once(deploy):
    values = (deploy / CHART / "values.yaml").read_text(encoding="utf-8")
    keys = re.findall(r"\bkey:\s*[\"']?([^\s,\"'{}]+)", values)
    assert set(keys) == set(MODELS)
    assert len(keys) == len(MODELS)


def template_port(pattern: str, template: str) -> int:
    """The port number that the regular expression's group captures in the template."""
    match = re.search(pattern, template, re.DOTALL)
    assert match is not None, f"no {pattern!r} in the chart's models.yaml"
    return int(match.group(1))


def test_helm_chart_serves_what_the_endpoint_map_expects(deploy, endpoints):
    chart = deploy / CHART
    models = load_yaml(chart / "values.yaml")["global"]["models"]
    by_key = {spec["key"]: (name, spec) for name, spec in models.items()}
    assert set(by_key) == set(MODELS)
    # One Deployment and one Service per model, both named after the entry of models in values.yaml.
    template = (chart / "charts" / "model-servers" / "templates" / "models.yaml").read_text(encoding="utf-8")
    vllm_port = template_port(r"--port=(\d+)", template)
    service_port = template_port(r"kind: Service\b.*?- port: (\d+)", template)
    assert template_port(r"kind: Service\b.*?targetPort: (\d+)", template) == vllm_port
    for key, endpoint in endpoints.items():
        name, spec = by_key[key]
        assert spec["hfModel"] == MODELS[key].hf_id, key
        assert spec["gpus"] == MODELS[key].gpus, key
        # The actuator scales the Deployment, and the runner reaches the Service of the same name.
        assert endpoint["deployment"] == name, key
        url = urlsplit(endpoint["base_url"])
        assert (url.scheme, url.hostname, url.port) == ("http", name, service_port), key


def test_router_job_runs_pickspin_live_with_the_endpoint_map(deploy, repo_root):
    job = load_yaml(deploy / "router-job.yaml")
    pod = job["spec"]["template"]["spec"]
    (container,) = pod["containers"]
    command = container["command"]
    assert command[0] == "pickspin"
    args = build_parser().parse_args(command[1:])
    assert args.command == "live"
    # The image's working directory holds copies of data/, deploy/ and models/, as the repository root does.
    assert args.endpoints is not None
    assert (repo_root / args.endpoints).is_file()
    # The runner scales Deployments in its own namespace, as the chart's router ServiceAccount.
    namespace = {"name": "PS_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}}
    assert namespace in container["env"]
    values = load_yaml(deploy / CHART / "values.yaml")
    assert pod["serviceAccountName"] == values["global"]["serviceAccount"]


def test_umbrella_chart_has_the_model_servers_and_the_gateway(deploy):
    chart = deploy / CHART
    meta = load_yaml(chart / "Chart.yaml")
    deps = {d["name"]: d for d in meta["dependencies"]}
    assert set(deps) == {"model-servers", "gateway"}
    for name, dep in deps.items():
        sub = load_yaml(chart / "charts" / name / "Chart.yaml")
        assert (sub["name"], sub["version"]) == (name, dep["version"])
        assert dep["repository"] == f"file://charts/{name}"


def test_gateway_serves_the_endpoint_map_with_pickspin_serve(deploy):
    templates = deploy / CHART / "charts" / "gateway" / "templates"
    deployment = (templates / "deployment.yaml").read_text(encoding="utf-8")
    args = re.findall(r"^\s*- (--?[\w-]+(?:=\S+)?|serve)\s*$", deployment, re.MULTILINE)
    assert args[0] == "serve"
    assert "--endpoints=/config/endpoints.json" in args
    assert "configMap: {name: {{ .Release.Name }}-gateway-endpoints}" in deployment
    assert "serviceAccountName: {{ .Values.global.serviceAccount }}" in deployment
    configmap = (templates / "configmap.yaml").read_text(encoding="utf-8")
    assert 'printf "http://%s:8000" $name' in configmap  # the Service name and vLLM port of each model
