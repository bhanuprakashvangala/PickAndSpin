"""`pickspin baseline run|judge`: the static baseline and the LLM judge.

`baseline run` sends every query to every model and can resume an interrupted run. `baseline judge`
labels the responses with an LLM judge configured by $JUDGE_API_BASE, $JUDGE_API_KEY, $JUDGE_MODEL and
$JUDGE_WORKERS. Both need only the base install. Without a subcommand the group prints its help.
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import logging
import os
import sys
from pathlib import Path
from typing import Final

from pickspin.config import MODELS
from pickspin.errors import ConfigError
from pickspin.paths import Paths, require_file, resolve_path

log = logging.getLogger(__name__)

# The judge's benchmarks in its default order, for the help text. They are written out because
# pickspin.baseline.judge, which defines JUDGE_BENCHMARKS, imports requests, and building the parser
# must not; tests/unit/test_cli_live.py checks that the two agree.
_JUDGE_BENCHMARKS_HELP: Final = "gsm8k, math, arc, mmlu_pro, hellaswag, truthfulqa, humaneval, mbpp"

# The descriptions are wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
The static baseline and its LLM judge: 'run' sends every query to every
model, and 'judge' then labels the successful responses CORRECT or INCORRECT."""

_RUN_DESCRIPTION: Final = """\
Send every query to every model, with no routing. Each model is an
OpenAI-compatible vLLM endpoint from the endpoint map. Responses are appended
to <out>/<model>_results.jsonl as they complete, and an interrupted run
resumes where it stopped: queries already in a model's file are skipped.
Needs only the base install."""

_RUN_EPILOG: Final = """\
environment:
  PS_ENDPOINTS  the endpoint map when --endpoints is not given
  VLLM_API_KEY  sent to the vLLM servers as a bearer token when set"""

_JUDGE_DESCRIPTION: Final = """\
Label the static baseline's successful responses CORRECT or INCORRECT with an
LLM judge behind an OpenAI-compatible API; the released labels came from
gpt-oss-120b. Reads <responses>/*_results.jsonl and appends one JSON line per
response to <out>/<benchmark>_accuracy.jsonl. Responses already judged are
skipped, so an interrupted run resumes. Needs only the base install."""

_JUDGE_EPILOG: Final = """\
environment (see .env.example):
  JUDGE_API_BASE  the judge's API base URL, including /v1 (required)
  JUDGE_API_KEY   sent to the judge as a bearer token
  JUDGE_MODEL     the judge's served model name (default: gpt-oss)
  JUDGE_WORKERS   concurrent requests without --workers (default: 100)"""


def _print_help(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Print the group's help to stderr and return 2: the group was given without a subcommand."""
    parser.print_help(sys.stderr)
    return 2


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the baseline command group with its run and judge subcommands."""
    parser = subparsers.add_parser(
        "baseline",
        help="run the static baseline and its LLM judge",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=functools.partial(_print_help, parser))
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    static = commands.add_parser(
        "run",
        help="send every query to every model",
        description=_RUN_DESCRIPTION,
        epilog=_RUN_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    names = list(MODELS)
    static.add_argument(
        "--models",
        nargs="*",
        default=names,
        choices=names,
        metavar="KEY",
        help=f"the models to run, in this order (default: all nine: {', '.join(names)})",
    )
    static.add_argument(
        "--workers", type=int, default=50, metavar="N", help="concurrent requests to each model (default: 50)"
    )
    static.add_argument(
        "--max-tokens", type=int, default=512, metavar="N", help="the most tokens a response may have (default: 512)"
    )
    static.add_argument(
        "--limit",
        type=int,
        metavar="N",
        help="send only the first N queries of the queries file (default: all; 0 sends none)",
    )
    paths = static.add_argument_group("paths")
    paths.add_argument(
        "--endpoints",
        type=Path,
        metavar="FILE",
        help="the endpoint map: each model's base_url and served model "
        "(default: $PS_ENDPOINTS, else <root>/deploy/endpoints.example.json)",
    )
    paths.add_argument(
        "--queries", type=Path, metavar="FILE", help="the benchmark queries (default: <root>/data/queries.jsonl.gz)"
    )
    paths.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/live/static)")
    static.set_defaults(func=run_static)

    judge = commands.add_parser(
        "judge",
        help="label the static baseline's responses with an LLM judge",
        description=_JUDGE_DESCRIPTION,
        epilog=_JUDGE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Unknown names are reported by run_judge with exit status 1, as configuration errors.
    judge.add_argument(
        "benchmarks",
        nargs="*",
        metavar="BENCHMARK",
        help=f"the benchmarks to judge, in this order (default: all eight: {_JUDGE_BENCHMARKS_HELP})",
    )
    judge.add_argument(
        "--workers",
        type=int,
        metavar="N",
        help="concurrent requests to the judge (default: $JUDGE_WORKERS, else 100)",
    )
    paths = judge.add_argument_group("paths")
    paths.add_argument(
        "--responses",
        type=Path,
        metavar="DIR",
        help="the static baseline's <model>_results.jsonl files (default: <root>/results/live/static)",
    )
    paths.add_argument(
        "--out", type=Path, metavar="DIR", help="output directory (default: <root>/results/live/judgments)"
    )
    judge.set_defaults(func=run_judge)


def run_static(args: argparse.Namespace) -> int:
    """Run the static baseline and return the exit status."""
    paths = Paths(args.root)
    endpoints_file = resolve_path(args.endpoints, paths.endpoints_example, env_var="PS_ENDPOINTS")
    queries_file = resolve_path(args.queries, paths.queries)
    out = resolve_path(args.out, paths.static_baseline)
    require_file(endpoints_file, "endpoint map")
    require_file(queries_file, "queries file")

    from pickspin.baseline.static import run_static_baseline
    from pickspin.data import load_queries
    from pickspin.live.vllm import bearer_headers, load_endpoints

    endpoints = load_endpoints(endpoints_file)
    missing = [m for m in args.models if m not in endpoints]
    if missing:
        raise ConfigError(f"{endpoints_file} has no endpoint for {', '.join(missing)}")
    headers = bearer_headers(os.environ.get("VLLM_API_KEY"))
    # A plain slice of the file order: no --limit runs every query and --limit 0 runs none.
    queries = load_queries(queries_file)[: args.limit]
    log.debug("Static baseline of %d queries on %s -> %s", len(queries), ", ".join(args.models), out)
    run_static_baseline(
        endpoints, args.models, queries, out, workers=args.workers, max_tokens=args.max_tokens, headers=headers
    )
    return 0


def run_judge(args: argparse.Namespace) -> int:
    """Judge the static baseline's responses and return the exit status."""
    paths = Paths(args.root)
    responses = resolve_path(args.responses, paths.static_baseline)
    out = resolve_path(args.out, paths.judgments)

    from pickspin.baseline.judge import JUDGE_BENCHMARKS, JudgeConfig, run_benchmark

    config = JudgeConfig.from_env()
    if args.workers is not None:
        config = dataclasses.replace(config, workers=args.workers)
    unknown = [b for b in args.benchmarks if b not in JUDGE_BENCHMARKS]
    if unknown:
        raise ConfigError(f"unknown benchmark: {', '.join(unknown)} (choose from {', '.join(JUDGE_BENCHMARKS)})")
    require_file(responses, "static-baseline responses")
    log.debug("%s", config)  # the API key is not part of the repr
    for benchmark in args.benchmarks or JUDGE_BENCHMARKS:
        run_benchmark(benchmark, config, responses, out)
    return 0
