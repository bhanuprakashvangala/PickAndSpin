"""`pickspin reproduce`: regenerate the paper's tables and figures from results/traces/.

Checks that the three trace files exist, writes the tables, figures and verification.csv to --out,
prints the paper-versus-reproduced comparison ending with 'N/84 numbers match the paper.', and exits
with status 1 unless every value matches. Needs only the base install.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Final

from pickspin.paths import Paths, require_file, resolve_path

log = logging.getLogger(__name__)

# The released traces the reproduction reads, each with the name used when it is missing.
_TRACES: Final[tuple[tuple[str, str], ...]] = (
    ("static_baseline.csv.gz", "static-baseline trace"),
    ("judgments.csv.gz", "judge-label trace"),
    ("pick_spin_routed.csv.gz", "routed-run trace"),
)

# Wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
Regenerate the paper's tables and figures (Sec. VII) from the released traces
and compare the published values with the reproduced ones. Writes the four
table CSVs, verification.csv and figures/*.png to --out, prints the comparison
and exits with status 1 unless every value matches. Needs no GPU or network."""


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the reproduce command."""
    parser = subparsers.add_parser(
        "reproduce",
        help="regenerate and check the paper's tables and figures",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--traces",
        type=Path,
        metavar="DIR",
        help="the released traces: static_baseline.csv.gz, judgments.csv.gz and pick_spin_routed.csv.gz "
        "(default: <root>/results/traces)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        metavar="DIR",
        help="where the tables, figures/ and verification.csv go (default: <root>/results)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Run the reproduction and return the exit status: 0 if every value matches the paper, else 1."""
    paths = Paths(args.root)
    traces = resolve_path(args.traces, paths.traces)
    out = resolve_path(args.out, paths.results)
    for name, what in _TRACES:
        require_file(traces / name, what)

    from pickspin.paper.reproduce import format_report, reproduce

    report = reproduce(traces, out)
    print(format_report(report.checks))
    if report.figures_written:
        log.info("Wrote tables and figures to %s/", out)
    else:
        log.info("Wrote tables to %s/ (no figures: matplotlib is not installed)", out)
    return 0 if report.all_match else 1
