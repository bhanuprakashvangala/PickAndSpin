"""`pickspin serve`: run the Pick and Spin gateway, an OpenAI-compatible service over the model servers.

The models come from a server file (deploy/nautilus/servers.json), so replacing or adding a model means
editing that file. Each model runs as a Kubernetes Job that Spin creates when a request needs it and
deletes after --cooldown idle seconds; --static instead expects every server to be running already.
The gateway deletes every model's Job when it starts and when it stops (also on SIGTERM), so no server
outlives it. Needs the [classifier] extra, and the [live] extra unless --static.
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import time
from pathlib import Path
from types import FrameType

from pickspin.config import DEFAULT_SPIN
from pickspin.paths import Paths, require_file, resolve_path
from pickspin.spin.lifecycle import LatencySignal

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the serve command."""
    parser = subparsers.add_parser(
        "serve",
        help="run the OpenAI-compatible Pick and Spin gateway",
        description=__doc__.split("\n\n", 1)[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--servers",
        type=Path,
        required=True,
        metavar="FILE",
        help="the server file that defines the models (deploy/nautilus/servers.json)",
    )
    parser.add_argument(
        "--endpoints",
        type=Path,
        metavar="FILE",
        help="an endpoint map overriding the Service URLs derived from the server file",
    )
    parser.add_argument(
        "--namespace",
        default=os.environ.get("PS_NAMESPACE", "pick-and-spin"),
        metavar="NS",
        help="the Kubernetes namespace of the model Jobs (default: $PS_NAMESPACE, else pick-and-spin)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="address to listen on (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8080, help="port to listen on (default: 8080)")
    parser.add_argument(
        "--static",
        action="store_true",
        help="never start or stop servers; every model's server must already be running",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_SPIN.cooldown_s,
        metavar="S",
        help="idle seconds before a model is scaled to zero (default: %(default)s)",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=600.0,
        metavar="S",
        help="seconds a request waits for a cold model before it gets 503 with Retry-After (default: %(default)s)",
    )
    parser.add_argument(
        "--latency-signal",
        default=LatencySignal.SPIN.value,
        choices=[signal.value for signal in LatencySignal],
        help="the latency Pick scores on (default: spin)",
    )
    parser.add_argument("--seed", type=int, default=0, metavar="N", help="seed of Pick's Thompson sampling")
    parser.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="the fine-tuned DistilBERT (default: $PS_CLASSIFIER, else <root>/models/distilbert-complexity-classifier)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run the gateway until interrupted and return the exit status."""
    paths = Paths(args.root)
    require_file(args.servers, "server file")
    model_dir = resolve_path(args.model_dir, paths.classifier_model, env_var="PS_CLASSIFIER")
    require_file(model_dir, "DistilBERT classifier")

    from pickspin.config import tiers_of
    from pickspin.live.gateway import Gateway, serve
    from pickspin.live.jobs import JobActuator, load_servers, render_endpoints
    from pickspin.live.vllm import bearer_headers, load_endpoints
    from pickspin.pick.classifier import HybridClassifier
    from pickspin.pick.router import Pick
    from pickspin.spin.lifecycle import Spin

    servers = load_servers(args.servers)
    catalog = servers["catalog"]
    endpoints = load_endpoints(args.endpoints) if args.endpoints else render_endpoints(servers)
    actuator = None
    if not args.static:
        timeout_s = float(servers["defaults"].get("load_timeout_s", 3600))
        actuator = JobActuator(servers, endpoints, args.namespace, timeout_s=timeout_s)
    spin = Spin(
        catalog,
        cooldown_s=args.cooldown,
        scale_to_zero=not args.static,
        now=time.monotonic(),
        load_estimate=actuator.load_estimate if actuator is not None else None,
        catalog=catalog,
    )
    classifier = HybridClassifier.from_pretrained(model_dir)
    pick = Pick(
        classifier,
        spin,
        LatencySignal(args.latency_signal),
        rng=random.Random(args.seed),
        tiers=tiers_of(catalog),
        models=catalog,
    )
    gateway = Gateway(
        pick=pick,
        spin=spin,
        catalog=catalog,
        endpoints=endpoints,
        actuator=actuator,
        headers=bearer_headers(os.environ.get("VLLM_API_KEY")),
        static=args.static,
        max_wait_s=args.max_wait,
    )
    if not args.static:
        log.info("Scaling every model to zero so the gateway starts COLD ...")
        gateway.release_all()
    signal.signal(signal.SIGTERM, _interrupt)  # Kubernetes stops a pod with SIGTERM
    server = serve(gateway, args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("Stopping the gateway")
    finally:
        gateway.stop.set()
        server.server_close()
        gateway.release_all()
    return 0


def _interrupt(signum: int, frame: FrameType | None) -> None:
    """Turn SIGTERM into KeyboardInterrupt, so the gateway stops the way Ctrl+C stops it."""
    raise KeyboardInterrupt
