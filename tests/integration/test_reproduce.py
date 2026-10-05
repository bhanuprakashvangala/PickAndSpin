"""`pickspin reproduce` on the released traces gives the committed tables and all 84 published values.

The command runs once, into a temporary directory, and the tests check what it printed and wrote. The
tables are compared with the committed results/*.csv after normalising line ends on both sides, since
a checkout may hold them with either; the files written must use CRLF, as the csv module writes them.
The figures are only checked for existence: their bytes depend on the matplotlib build, so the CI
base-install job is what compares the committed tables and figures with git diff.
"""

import contextlib
import csv
import io
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from pickspin.cli.main import main

TABLES = (
    "static_baseline_summary.csv",
    "static_baseline_per_benchmark_accuracy.csv",
    "table1_routing.csv",
    "table2_scored_routed_queries_by_size.csv",
    "verification.csv",
)
FIGURES = ("fig2a_throughput.png", "fig2b_latency.png", "fig3_thompson.png")
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


@dataclass(frozen=True)
class Reproduction:
    """One run of `pickspin reproduce`: its exit status, what it printed and logged, and its --out."""

    status: int
    stdout: str
    stderr: str
    out: Path


@contextlib.contextmanager
def pickspin_logger_restored() -> Iterator[None]:
    """Put the 'pickspin' logger back as it was, since main() installs its own stderr handler on it.

    The reproduction runs in a module-scoped fixture, before the per-test fixture in tests/conftest.py
    takes its snapshot of the logger, so it restores the logger itself.
    """
    logger = logging.getLogger("pickspin")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    try:
        yield
    finally:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
            if handler not in handlers:
                handler.close()
        for handler in handlers:
            logger.addHandler(handler)
        logger.setLevel(level)
        logger.propagate = propagate


@pytest.fixture(scope="module")
def reproduction(released_data: Path, tmp_path_factory: pytest.TempPathFactory) -> Reproduction:
    """Run `pickspin --root <repo> reproduce --out <tmp>/results` once for the whole module."""
    out = tmp_path_factory.mktemp("reproduce") / "results"
    stdout, stderr = io.StringIO(), io.StringIO()
    with pickspin_logger_restored(), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        status = main(["--root", str(released_data), "reproduce", "--out", str(out)])
    return Reproduction(status, stdout.getvalue(), stderr.getvalue(), out)


def normalised(path: Path) -> bytes:
    """The file's bytes with CRLF line ends turned into LF."""
    return path.read_bytes().replace(b"\r\n", b"\n")


def test_reproduce_exits_0_and_reports_84_of_84(reproduction):
    assert reproduction.status == 0, reproduction.stdout + reproduction.stderr
    assert reproduction.stdout.endswith("\n\n84/84 numbers match the paper.\n")
    # Logging goes to stderr, and stdout carries only the report.
    assert f"Wrote tables and figures to {reproduction.out}/" in reproduction.stderr
    assert "Wrote tables" not in reproduction.stdout


def test_reproduced_tables_equal_the_committed_ones(reproduction, released_data):
    committed = released_data / "results"
    assert sorted(path.name for path in committed.glob("*.csv")) == sorted(TABLES)
    assert sorted(path.name for path in reproduction.out.glob("*.csv")) == sorted(TABLES)
    for name in TABLES:
        produced = (reproduction.out / name).read_bytes()
        assert produced.endswith(b"\r\n"), f"{name} does not end with CRLF"
        assert produced.count(b"\n") == produced.count(b"\r\n"), f"{name} has a line end other than CRLF"
        assert normalised(reproduction.out / name) == normalised(committed / name), f"{name} differs"


def test_verification_lists_84_matching_values(reproduction):
    with (reproduction.out / "verification.csv").open(newline="", encoding="utf-8") as f:
        header, *rows = list(csv.reader(f))
    assert header == ["claim", "paper", "reproduced", "match"]
    assert len(rows) == 84
    assert [row[3] for row in rows] == ["yes"] * 84


def test_reproduce_draws_the_three_figures(reproduction):
    for name in FIGURES:
        path = reproduction.out / "figures" / name
        assert path.is_file(), f"{name} was not written"
        with path.open("rb") as f:
            assert f.read(len(PNG_SIGNATURE)) == PNG_SIGNATURE, f"{name} is not a PNG file"
