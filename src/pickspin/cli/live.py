"""`pickspin live`: route the benchmark queries through Pick and Spin on a Kubernetes deployment.

Builds a LiveConfig from the flags and the environment ($PS_ENDPOINTS, $PS_NAMESPACE, $PS_CLASSIFIER,
$VLLM_API_KEY) and runs pickspin.live.runner.run_live. Needs the [classifier] extra, and the [live]
extra unless --static.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Final

from pickspin.config import DEFAULT_SPIN
from pickspin.paths import Paths, require_file, resolve_path
from pickspin.spin.lifecycle import LatencySignal

log = logging.getLogger(__name__)

# Wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
Route the benchmark queries through Pick and Spin on a Kubernetes deployment
with one vLLM Deployment per model. Pick classifies each query (keyword lists,
then DistilBERT) and selects a model with the latency Spin reports. Every
model starts at zero replicas: Spin scales a cold model to one replica, holds
its queries until vLLM answers /health, and scales a model that has had
nothing in flight for --cooldown seconds back to zero. --static keeps every
model running instead. Writes one JSON line per query to
<out>/pick_spin_<time>.jsonl, and the GPU-hours, utilization and cold starts
to <out>/pick_spin_<time>_summary.json. Needs the [classifier] extra, and the
[live] extra unless --static."""

_EPILOG: Final = """\
environment:
  PS_ENDPOINTS   the endpoint map when --endpoints is not given
  PS_NAMESPACE   the namespace when --namespace is not given
  PS_CLASSIFIER  the DistilBERT directory when --model-dir is not given
  VLLM_API_KEY   sent to the vLLM servers as a bearer token when set"""


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the live command."""
    parser = subparsers.add_parser(
        "live",
        help="route the queries through Pick and Spin on Kubernetes",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--namespace",
        # Read when the parser is built, as the old runner did: an empty PS_NAMESPACE stays empty.
        default=os.environ.get("PS_NAMESPACE", "pick-and-spin"),
        metavar="NS",
        help="the Kubernetes namespace of the model Deployments (default: $PS_NAMESPACE, else pick-and-spin)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=250,
        metavar="N",
        help="worker threads, each with one query in flight (default: 250)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="route only the first N queries of the seeded shuffle, for a quick check (default: all; 0 means all)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, metavar="N", help="the most tokens a response may have (default: 256)"
    )
    parser.add_argument(
        "--static",
        action="store_true",
        help="keep every model running, with no cold starts or scaling (install the Helm chart with "
        "--set startReplicas=1)",
    )
    # Plain strings, so that argparse's messages show them as typed; run() converts them.
    parser.add_argument(
        "--latency-signal",
        default=LatencySignal.SPIN.value,
        choices=[signal.value for signal in LatencySignal],
        help="the latency Pick scores on. spin: Spin's latency, which adds the cold-start penalty of a model "
        "that is not warm; observed: the mean observed latency, cold-start waits included; inference: the "
        "inference latency only (default: spin)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        metavar="N",
        help="the seed of the query shuffle and of Pick's Thompson sampling (default: 0)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_SPIN.cooldown_s,
        metavar="S",
        help="seconds a model must have nothing in flight before it is scaled to zero (default: %(default)s)",
    )
    paths = parser.add_argument_group("paths")
    paths.add_argument(
        "--endpoints",
        type=Path,
        metavar="FILE",
        help="the endpoint map: each model's base_url, served model and Deployment "
        "(default: $PS_ENDPOINTS, else <root>/deploy/endpoints.example.json)",
    )
    paths.add_argument(
        "--servers",
        type=Path,
        metavar="FILE",
        help="run each model as a Job described in this server file (deploy/nautilus/servers.json) instead of "
        "scaling Deployments; for clusters such as NRP Nautilus that do not allow GPU Deployments",
    )
    paths.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/live)")
    paths.add_argument(
        "--queries", type=Path, metavar="FILE", help="the benchmark queries (default: <root>/data/queries.jsonl.gz)"
    )
    paths.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="the fine-tuned DistilBERT (default: $PS_CLASSIFIER, else <root>/models/distilbert-complexity-classifier)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run a live experiment and return the exit status."""
    paths = Paths(args.root)
    endpoints = resolve_path(args.endpoints, paths.endpoints_example, env_var="PS_ENDPOINTS")
    queries = resolve_path(args.queries, paths.queries)
    model_dir = resolve_path(args.model_dir, paths.classifier_model, env_var="PS_CLASSIFIER")
    out = resolve_path(args.out, paths.live)
    require_file(endpoints, "endpoint map")
    require_file(queries, "queries file")
    require_file(model_dir, "DistilBERT classifier")
    if args.servers is not None:
        require_file(args.servers, "server file")

    from pickspin.live.runner import LiveConfig, run_live

    config = LiveConfig(
        endpoints=endpoints,
        queries=queries,
        out_dir=out,
        model_dir=model_dir,
        namespace=args.namespace,
        workers=args.workers,
        limit=args.limit,
        max_tokens=args.max_tokens,
        static=args.static,
        latency_signal=LatencySignal(args.latency_signal),
        seed=args.seed,
        cooldown_s=args.cooldown,
        api_key=os.environ.get("VLLM_API_KEY"),
        servers=args.servers,
    )
    log.debug("%s", config)  # the API key is not part of the repr
    run_live(config)
    return 0
