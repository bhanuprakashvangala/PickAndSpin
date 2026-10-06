"""`pickspin replay`: load-test a running gateway with the benchmark queries.

Sends the queries, in the same seeded shuffle as `pickspin live`, to a gateway started with
`pickspin serve` and records each answer, the model the gateway chose and the time it took. Needs no
extra: the gateway does the routing and the scaling.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path

from pickspin.paths import Paths, require_file, resolve_path

log = logging.getLogger(__name__)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the replay command."""
    parser = subparsers.add_parser(
        "replay",
        help="load-test a running gateway with the benchmark queries",
        description=__doc__.split("\n\n", 1)[1] if __doc__ else None,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", required=True, help="the gateway's base URL, such as http://localhost:8080")
    parser.add_argument("--workers", type=int, default=20, metavar="N", help="queries in flight at once (default: 20)")
    parser.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="send only the first N queries of the seeded shuffle (default: all; 0 means all)",
    )
    parser.add_argument("--seed", type=int, default=0, metavar="N", help="the seed of the query shuffle (default: 0)")
    parser.add_argument(
        "--max-tokens", type=int, default=256, metavar="N", help="the most tokens an answer may have (default: 256)"
    )
    parser.add_argument(
        "--give-up",
        type=float,
        default=3600.0,
        metavar="S",
        help="stop resending a query whose model is still loading after S seconds (default: %(default)s)",
    )
    parser.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/replay)")
    parser.add_argument(
        "--queries", type=Path, metavar="FILE", help="the benchmark queries (default: <root>/data/queries.jsonl.gz)"
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run a replay and return the exit status: 0 if every query was answered, 1 otherwise."""
    paths = Paths(args.root)
    queries_file = require_file(resolve_path(args.queries, paths.queries), "queries file")
    out = resolve_path(args.out, paths.results / "replay")

    from pickspin.data import load_queries
    from pickspin.live.replay import replay

    queries = load_queries(queries_file)
    random.Random(args.seed).shuffle(queries)
    queries = queries[: args.limit] if args.limit else queries
    stem = replay(args.url, queries, out, workers=args.workers, max_tokens=args.max_tokens, give_up_s=args.give_up)
    summary_file = Path(f"{stem}_summary.json")
    log.info("Wrote %s.jsonl and %s", stem, summary_file)
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    return 0 if summary["succeeded"] == summary["queries"] else 1
