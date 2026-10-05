"""`pickspin simulate`: run the trace-driven simulator over a grid of policies, seeds and loads.

Takes the old simulator's flags with the same defaults, plus overrides for the input and output paths.
Writes <out>/<load>/{summary,per_model,cold_starts_by_tier}.csv for each load (closed-<workers>, or
poisson-<rate>qps for each --arrival-rate), optionally per-query traces, and <out>/overview.csv. Needs
only the base install; --reclassify needs the [classifier] extra.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import textwrap
from pathlib import Path
from typing import Final

from pickspin.config import DEFAULT_SPIN
from pickspin.paths import Paths, require_file, resolve_path
from pickspin.simulation.policies import POLICIES, Policy

log = logging.getLogger(__name__)

# Wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
Replay the released traces through Pick and Spin on a simulated clock: a query
routed to a model takes the latency and success recorded for that pair, and
Spin moves each model between COLD, LOADING and WARM. Runs every policy with
every seed, under closed-loop workers or, with --arrival-rate, Poisson
arrivals. Writes <out>/<load>/{summary,per_model,cold_starts_by_tier}.csv,
where <load> is closed-<workers> or poisson-<rate>qps, and <out>/overview.csv,
the mean over seeds of every load under <out>, including earlier runs."""


def _policies_epilog() -> str:
    """List the policies with their descriptions, as the old simulator's docstring did."""
    lines = ["policies:"]
    for policy, spec in POLICIES.items():
        lines += textwrap.wrap(
            spec.description,
            width=78,
            initial_indent=f"  {policy:<24}",
            subsequent_indent=" " * 26,
            break_on_hyphens=False,
        )
    return "\n".join(lines)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the simulate command."""
    parser = subparsers.add_parser(
        "simulate",
        help="run the trace-driven simulator over policies, seeds and loads",
        description=_DESCRIPTION,
        epilog=_policies_epilog(),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Policy names are plain strings, so that argparse's messages show them as typed; run() converts them.
    names = [policy.value for policy in Policy]
    parser.add_argument(
        "--policies",
        nargs="+",
        default=names,
        choices=names,
        metavar="POLICY",
        help="the policies to simulate, listed below (default: all four, in that order)",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        metavar="N",
        help="random seeds; each policy runs once per seed (default: 0 1 2 3 4)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=250,
        metavar="N",
        help="closed-loop clients (default: 250; ignored with --arrival-rate)",
    )
    parser.add_argument(
        "--arrival-rate",
        type=float,
        nargs="+",
        metavar="R",
        help="open-loop Poisson arrivals, queries per second (one run per rate)",
    )
    parser.add_argument(
        "--max-concurrency", type=int, metavar="N", help="cap on queries in flight per model (default: no cap)"
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_SPIN.cooldown_s,
        metavar="S",
        help="seconds a model must have nothing in flight before it is scaled to zero (default: %(default)s)",
    )
    parser.add_argument(
        "--reclassify",
        action="store_true",
        help="rerun the hybrid classifier and rewrite the tier cache (needs DistilBERT and the [classifier] extra)",
    )
    parser.add_argument("--write-queries", action="store_true", help="also write a per-query trace for the first seed")
    paths = parser.add_argument_group("paths")
    paths.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/simulation)")
    paths.add_argument(
        "--queries", type=Path, metavar="FILE", help="the benchmark queries (default: <root>/data/queries.jsonl.gz)"
    )
    paths.add_argument(
        "--traces",
        type=Path,
        metavar="DIR",
        help="the released traces: static_baseline.csv.gz and judgments.csv.gz (default: <root>/results/traces)",
    )
    paths.add_argument(
        "--tier-cache",
        type=Path,
        metavar="FILE",
        help="the classifier's tier for every query (default: <root>/data/query_tiers.csv.gz)",
    )
    paths.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="the fine-tuned DistilBERT, used only with --reclassify or a missing tier cache "
        "(default: $PS_CLASSIFIER, else <root>/models/distilbert-complexity-classifier)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run the simulations and return the exit status."""
    paths = Paths(args.root)
    queries = resolve_path(args.queries, paths.queries)
    traces = resolve_path(args.traces, paths.traces)
    tier_cache = resolve_path(args.tier_cache, paths.tier_cache)
    model_dir = resolve_path(args.model_dir, paths.classifier_model, env_var="PS_CLASSIFIER")
    out = resolve_path(args.out, paths.simulation)
    require_file(queries, "queries file")
    require_file(traces / "static_baseline.csv.gz", "static-baseline trace")
    require_file(traces / "judgments.csv.gz", "judge-label trace")
    policies = [Policy(name) for name in args.policies]

    from pickspin.simulation.engine import SimulationSettings
    from pickspin.simulation.experiment import run_load, write_overview
    from pickspin.simulation.inputs import describe_stages, load_trace_data, query_tiers

    log.debug("Reading %s and %s, writing to %s", queries, traces, out)
    data = load_trace_data(queries, traces)
    tiers = query_tiers(data.queries, tier_cache, reclassify=args.reclassify, model_dir=model_dir)
    log.info("%s", describe_stages(len(data.queries), tiers))
    base = SimulationSettings(workers=args.workers, max_concurrency=args.max_concurrency, cooldown_s=args.cooldown)
    for rate in args.arrival_rate or [None]:
        settings = dataclasses.replace(base, arrival_rate=rate)
        run_load(policies, args.seeds, settings, data, tiers, out, write_queries=args.write_queries)
    write_overview(out)
    return 0
