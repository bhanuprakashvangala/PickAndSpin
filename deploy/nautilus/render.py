"""Render the Nautilus manifests from servers.json.

    python deploy/nautilus/render.py              # services.json and endpoints.json
    python deploy/nautilus/render.py --with-jobs  # also every server Job (for a static run)

Writes Kubernetes List files that `kubectl apply -f` accepts, next to this script.
"""

import argparse
import json
from pathlib import Path

from pickspin.live.jobs import load_servers, render_all, render_endpoints

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-jobs", action="store_true", help="also write jobs.json with every server Job")
    args = ap.parse_args()
    servers = load_servers(HERE / "servers.json")
    (HERE / "endpoints.json").write_text(json.dumps(render_endpoints(servers), indent=1) + "\n", encoding="utf-8")
    services = {"apiVersion": "v1", "kind": "List", "items": render_all(servers, jobs=False)}
    (HERE / "services.json").write_text(json.dumps(services, indent=1) + "\n", encoding="utf-8")
    if args.with_jobs:
        jobs = [m for m in render_all(servers, jobs=True) if m["kind"] == "Job"]
        (HERE / "jobs.json").write_text(
            json.dumps({"apiVersion": "v1", "kind": "List", "items": jobs}, indent=1) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
