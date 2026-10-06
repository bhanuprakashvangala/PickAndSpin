"""Render the Nautilus manifests from the server files.

    python deploy/nautilus/render.py              # services.json, endpoints.json, endpoints-paper.json
    python deploy/nautilus/render.py --with-jobs  # also every server Job (for a static run)

servers.json is the catalog the gateway serves; servers-paper.json holds the paper's nine models for the
benchmark replay (router-job.yaml). Each catalog gets its endpoint map (and Job list), and services.json
holds the Services of both. Writes Kubernetes List files that `kubectl apply -f` accepts, next to this script.
"""

import argparse
import json
from pathlib import Path

from pickspin.live.jobs import load_servers, render_all, render_endpoints

HERE = Path(__file__).resolve().parent
CATALOGS = {"servers.json": "", "servers-paper.json": "-paper"}  # server file -> suffix of its outputs


def write(name: str, content: object) -> None:
    (HERE / name).write_text(json.dumps(content, indent=1) + "\n", encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--with-jobs", action="store_true", help="also write jobs.json and jobs-paper.json")
    args = ap.parse_args()
    services: dict[str, dict[str, object]] = {}
    for file, suffix in CATALOGS.items():
        servers = load_servers(HERE / file)
        write(f"endpoints{suffix}.json", render_endpoints(servers))
        for manifest in render_all(servers, jobs=False):
            services.setdefault(manifest["metadata"]["name"], manifest)
        if args.with_jobs:
            jobs = [m for m in render_all(servers, jobs=True) if m["kind"] == "Job"]
            write(f"jobs{suffix}.json", {"apiVersion": "v1", "kind": "List", "items": jobs})
    write("services.json", {"apiVersion": "v1", "kind": "List", "items": list(services.values())})


if __name__ == "__main__":
    main()
