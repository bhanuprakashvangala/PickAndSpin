"""The `pickspin` command line: parsing, defaults, paths, logging, exit codes and the simulate wiring.

The tests are hermetic. `pickspin simulate` runs for real on a 12-query copy of the released inputs
written to tmp_path; elsewhere the library call behind a command is replaced with a recorder. The real
reproduction and the full-size simulations are checked by tests/integration (test_reproduce.py and
test_simulation_golden.py).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import logging
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

import pickspin
from pickspin.cli import reproduce as reproduce_command
from pickspin.cli.main import build_parser, configure_logging, main
from pickspin.config import MODELS, Tier
from pickspin.errors import ConfigError
from pickspin.paper import reproduce as paper_reproduce
from pickspin.paper.reproduce import Check, ReproductionReport, format_report
from pickspin.simulation import experiment, inputs
from pickspin.simulation.engine import SimulationSettings
from pickspin.simulation.experiment import run_load, write_overview
from pickspin.simulation.inputs import TraceData, load_trace_data, read_tier_cache
from pickspin.simulation.policies import POLICIES, Policy

TRACE_FILES = ("static_baseline.csv.gz", "judgments.csv.gz", "pick_spin_routed.csv.gz")
N_QUERIES = 12

# main() configures the 'pickspin' logger for the whole process; the autouse fixture
# restore_pickspin_logger in tests/conftest.py puts it back after each test.


def touch(*paths: Path) -> None:
    """Create empty files, with their directories, where a command only checks that its inputs exist."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def write_csv_gz(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def tree(root: Path) -> dict[str, bytes]:
    """Every file under root by relative path, with gzip files decompressed (their headers hold an mtime)."""
    return {
        p.relative_to(root).as_posix(): gzip.decompress(p.read_bytes()) if p.suffix == ".gz" else p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    """A repository root with a 12-query version of the released inputs that the simulator reads.

    Query i takes 1 + i % 4 seconds on every model and fails when i % 6 == 5. The judge marks it
    correct when i is even and did not score it when i % 4 == 3. The queries cycle through the three
    tiers; those with i % 3 == 2 were tiered by DistilBERT, the others by keyword.
    """
    root = tmp_path / "repo"
    ids = [f"q{i}" for i in range(N_QUERIES)]
    queries = root / "data" / "queries.jsonl.gz"
    queries.parent.mkdir(parents=True)
    with gzip.open(queries, "wt", encoding="utf-8") as f:
        for i, qid in enumerate(ids):
            record = {"id": qid, "benchmark": "gsm8k" if i % 2 else "mbpp", "query": f"query {i}"}
            f.write(json.dumps({**record, "ground_truth": "", "query_type": ""}) + "\n")
    write_csv_gz(
        root / "results" / "traces" / "static_baseline.csv.gz",
        ["id", "model", "success", "latency_ms"],
        [[qid, m, "0" if i % 6 == 5 else "1", 1000 * (1 + i % 4)] for i, qid in enumerate(ids) for m in MODELS],
    )
    write_csv_gz(
        root / "results" / "traces" / "judgments.csv.gz",
        ["id", "model", "is_correct"],
        [[qid, m, "1" if i % 2 == 0 else "0"] for i, qid in enumerate(ids) for m in MODELS if i % 4 != 3],
    )
    tiers = list(Tier)
    write_csv_gz(
        root / "data" / "query_tiers.csv.gz",
        ["id", "tier", "stage"],
        [[qid, tiers[i % 3], "distilbert" if i % 3 == 2 else "keyword"] for i, qid in enumerate(ids)],
    )
    return root


# --- the parser ----------------------------------------------------------------------------------------


def test_commands_are_listed_in_order() -> None:
    [subparsers] = [a for a in build_parser()._actions if isinstance(a, argparse._SubParsersAction)]
    assert list(subparsers.choices) == ["reproduce", "simulate", "live", "baseline", "classifier"]


@pytest.mark.parametrize(
    "argv",
    [
        ["reproduce"],
        ["simulate"],
        ["live"],
        ["baseline", "run"],
        ["baseline", "judge"],
        ["classifier", "labels"],
        ["classifier", "train"],
        ["classifier", "evaluate"],
    ],
    ids=" ".join,
)
def test_every_subcommand_parses(argv: list[str]) -> None:
    args = build_parser().parse_args(argv)
    assert args.command == argv[0]
    assert callable(args.func)


def test_global_option_defaults() -> None:
    args = build_parser().parse_args(["reproduce"])
    assert (args.root, args.verbose, args.quiet) == (Path(), 0, False)
    args = build_parser().parse_args(["--root", "elsewhere", "-vv", "reproduce"])
    assert (args.root, args.verbose, args.quiet) == (Path("elsewhere"), 2, False)


def test_global_options_come_before_the_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["reproduce", "--root", "elsewhere"])
    assert exited.value.code == 2
    assert "unrecognized arguments: --root" in capsys.readouterr().err


def test_verbose_and_quiet_exclude_each_other(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["-v", "-q", "reproduce"])
    assert exited.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["--version"])
    assert exited.value.code == 0
    assert capsys.readouterr().out == f"pickspin {pickspin.__version__}\n"


def test_python_dash_m_pickspin_is_the_same_command() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "pickspin", "--version"], capture_output=True, text=True, timeout=60, check=False
    )
    assert (done.returncode, done.stdout) == (0, f"pickspin {pickspin.__version__}\n"), done.stderr


def test_no_command_prints_help_and_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("usage: pickspin ")
    assert "reproduce" in err
    assert "simulate" in err


@pytest.mark.parametrize("group", ["baseline", "classifier"])
def test_a_group_without_its_subcommand_prints_help_and_returns_2(
    group: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([group]) == 2
    out, err = capsys.readouterr()
    assert f"usage: pickspin {group} " in out + err


# --- simulate flags ----------------------------------------------------------------------------------


def test_simulate_defaults_match_v1_1_0() -> None:
    args = build_parser().parse_args(["simulate"])
    assert args.policies == ["pick-and-spin", "pick-and-spin-observed", "unaware", "static"]
    assert args.policies == list(POLICIES)
    assert all(type(p) is str for p in args.policies)  # converted to Policy only after parsing
    assert args.seeds == [0, 1, 2, 3, 4]
    assert args.workers == 250
    assert args.cooldown == 300.0
    assert isinstance(args.cooldown, float)
    assert args.arrival_rate is None
    assert args.max_concurrency is None
    assert args.reclassify is False
    assert args.write_queries is False
    assert (args.out, args.queries, args.traces, args.tier_cache, args.model_dir) == (None,) * 5


def test_simulate_flags() -> None:
    argv = "simulate --policies static unaware --seeds 3 1 --workers 8 --arrival-rate 0.25 4 --max-concurrency 2"
    argv += " --cooldown 10 --reclassify --write-queries --out o --queries q --traces t --tier-cache c --model-dir m"
    args = build_parser().parse_args(argv.split())
    assert args.policies == ["static", "unaware"]
    assert args.seeds == [3, 1]
    assert args.workers == 8
    assert args.arrival_rate == [0.25, 4.0]
    assert args.max_concurrency == 2
    assert args.cooldown == 10.0
    assert isinstance(args.cooldown, float)
    assert args.reclassify is True
    assert args.write_queries is True
    assert (args.out, args.queries, args.traces, args.tier_cache, args.model_dir) == tuple(map(Path, "oqtcm"))


def test_an_unknown_policy_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["simulate", "--policies", "pick-and-spin", "bogus"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'bogus'" in err
    assert "pick-and-spin-observed" in err
    assert "Policy." not in err  # the choices are listed as the plain names


def test_simulate_help_describes_every_policy(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["simulate", "--help"])
    assert exited.value.code == 0
    out = capsys.readouterr().out
    for policy in Policy:
        assert re.search(rf"^  {re.escape(policy)} +\S", out, re.MULTILINE), policy


# --- reproduce -----------------------------------------------------------------------------------------


def fake_reproduce(
    monkeypatch: pytest.MonkeyPatch, checks: Sequence[Check], *, figures: bool = True
) -> list[tuple[Path, Path]]:
    """Make pickspin.paper.reproduce.reproduce return a report of these checks; returns its calls."""
    calls: list[tuple[Path, Path]] = []

    def reproduce(traces_dir: Path, out_dir: Path) -> ReproductionReport:
        calls.append((traces_dir, out_dir))
        return ReproductionReport(checks=tuple(checks), out_dir=out_dir, figures_written=figures)

    monkeypatch.setattr(paper_reproduce, "reproduce", reproduce)
    return calls


MATCHING = [Check("Queries", "31,019", "31019"), Check("Ratio", "10x", "10x")]


def test_reproduce_resolves_its_defaults_under_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch(*(tmp_path / "results" / "traces" / name for name in TRACE_FILES))
    calls = fake_reproduce(monkeypatch, MATCHING)
    assert main(["--root", str(tmp_path), "reproduce"]) == 0
    assert calls == [(tmp_path / "results" / "traces", tmp_path / "results")]


def test_reproduce_uses_explicit_paths_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    touch(*(Path("my traces") / name for name in TRACE_FILES))
    calls = fake_reproduce(monkeypatch, MATCHING)
    argv = ["--root", str(tmp_path / "elsewhere"), "reproduce", "--traces", "my traces", "--out", "out"]
    assert main(argv) == 0
    assert calls == [(Path("my traces"), Path("out"))]


def test_reproduce_prints_the_report_and_returns_0_when_every_value_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch(*(tmp_path / "results" / "traces" / name for name in TRACE_FILES))
    fake_reproduce(monkeypatch, MATCHING)
    assert main(["--root", str(tmp_path), "reproduce", "--out", str(tmp_path / "out")]) == 0
    out, err = capsys.readouterr()
    assert out == format_report(MATCHING) + "\n"
    assert out.endswith("\n2/2 numbers match the paper.\n")
    assert err == f"Wrote tables and figures to {tmp_path / 'out'}/\n"


def test_reproduce_returns_1_when_a_value_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch(*(tmp_path / "results" / "traces" / name for name in TRACE_FILES))
    fake_reproduce(monkeypatch, [*MATCHING, Check("Latency", "1.51", "1.52")])
    assert main(["--root", str(tmp_path), "reproduce"]) == 1
    assert capsys.readouterr().out.endswith("\n2/3 numbers match the paper.\n")


def test_reproduce_says_when_the_figures_were_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch(*(tmp_path / "results" / "traces" / name for name in TRACE_FILES))
    fake_reproduce(monkeypatch, MATCHING, figures=False)
    assert main(["--root", str(tmp_path), "reproduce"]) == 0
    assert "no figures: matplotlib is not installed" in capsys.readouterr().err


@pytest.mark.parametrize("missing", TRACE_FILES)
def test_reproduce_without_a_trace_returns_1_before_writing_anything(
    missing: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    traces = tmp_path / "results" / "traces"
    touch(*(traces / name for name in TRACE_FILES if name != missing))
    assert main(["--root", str(tmp_path), "reproduce"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("pickspin: error: ")
    assert f"not found: {traces / missing} (run from the repository root" in err
    assert sorted(p.name for p in (tmp_path / "results").iterdir()) == ["traces"]


def test_reproduce_in_an_empty_directory_returns_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["--root", str(tmp_path), "reproduce"]) == 1
    assert "not found" in capsys.readouterr().err


# --- simulate: paths -----------------------------------------------------------------------------------


@pytest.fixture
def simulate_calls(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[Any]]:
    """Replace the library calls behind `pickspin simulate` with recorders; returns their arguments by name."""
    calls: dict[str, list[Any]] = {"load_trace_data": [], "query_tiers": [], "run_load": [], "write_overview": []}
    data = TraceData(queries=[], runs={}, correct={})

    def load_trace_data(queries_path: Path, traces_dir: Path) -> TraceData:
        calls["load_trace_data"].append((queries_path, traces_dir))
        return data

    def query_tiers(
        queries: Sequence[Any], cache: Path, *, reclassify: bool = False, model_dir: Path | None = None
    ) -> dict[str, Any]:
        calls["query_tiers"].append((cache, reclassify, model_dir))
        return {}

    def run_load(
        policies: Sequence[Policy],
        seeds: Sequence[int],
        settings: SimulationSettings,
        trace_data: TraceData,
        tiers: dict[str, Any],
        out_root: Path,
        *,
        write_queries: bool = False,
    ) -> Path:
        assert trace_data is data
        calls["run_load"].append((list(policies), list(seeds), settings, out_root, write_queries))
        return out_root / settings.load_name

    def write_overview(root: Path) -> Path:
        calls["write_overview"].append(root)
        return root / "overview.csv"

    monkeypatch.setattr(inputs, "load_trace_data", load_trace_data)
    monkeypatch.setattr(inputs, "query_tiers", query_tiers)
    monkeypatch.setattr(experiment, "run_load", run_load)
    monkeypatch.setattr(experiment, "write_overview", write_overview)
    return calls


def simulate_inputs(root: Path) -> list[Path]:
    return [
        root / "data" / "queries.jsonl.gz",
        root / "results" / "traces" / "static_baseline.csv.gz",
        root / "results" / "traces" / "judgments.csv.gz",
    ]


def test_simulate_resolves_its_defaults_under_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulate_calls: dict[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.delenv("PS_CLASSIFIER", raising=False)
    touch(*simulate_inputs(tmp_path))
    assert main(["--root", str(tmp_path), "simulate"]) == 0
    assert simulate_calls["load_trace_data"] == [
        (tmp_path / "data" / "queries.jsonl.gz", tmp_path / "results" / "traces")
    ]
    model_dir = tmp_path / "models" / "distilbert-complexity-classifier"
    assert simulate_calls["query_tiers"] == [(tmp_path / "data" / "query_tiers.csv.gz", False, model_dir)]
    assert simulate_calls["run_load"] == [
        (list(Policy), [0, 1, 2, 3, 4], SimulationSettings(), tmp_path / "results" / "simulation", False)
    ]
    assert all(type(p) is Policy for p in simulate_calls["run_load"][0][0])
    assert simulate_calls["write_overview"] == [tmp_path / "results" / "simulation"]


def test_simulate_uses_explicit_paths_as_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulate_calls: dict[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    touch(Path("in/q.gz"), Path("tr/static_baseline.csv.gz"), Path("tr/judgments.csv.gz"))
    paths = ["--queries", "in/q.gz", "--traces", "tr", "--tier-cache", "c.gz", "--model-dir", "m", "--out", "o"]
    assert main(["--root", str(tmp_path / "elsewhere"), "simulate", *paths]) == 0
    assert simulate_calls["load_trace_data"] == [(Path("in/q.gz"), Path("tr"))]
    assert simulate_calls["query_tiers"] == [(Path("c.gz"), False, Path("m"))]
    assert simulate_calls["run_load"][0][3] == Path("o")
    assert simulate_calls["write_overview"] == [Path("o")]


@pytest.mark.parametrize(
    ("flag", "env", "expected"),
    [
        (None, None, "<root>"),
        (None, "", "<root>"),  # an empty variable counts as unset
        (None, "env-model", "env-model"),
        ("flag-model", "env-model", "flag-model"),
    ],
)
def test_simulate_model_dir_is_flag_then_ps_classifier_then_root(
    flag: str | None,
    env: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    simulate_calls: dict[str, list[Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if env is None:
        monkeypatch.delenv("PS_CLASSIFIER", raising=False)
    else:
        monkeypatch.setenv("PS_CLASSIFIER", env)
    touch(*simulate_inputs(tmp_path))
    argv = ["--root", str(tmp_path), "simulate", "--reclassify"] + (["--model-dir", flag] if flag else [])
    assert main(argv) == 0
    default = tmp_path / "models" / "distilbert-complexity-classifier"
    [(_, reclassify, model_dir)] = simulate_calls["query_tiers"]
    assert reclassify is True
    assert model_dir == (default if expected == "<root>" else Path(expected))


def test_simulate_runs_one_load_per_arrival_rate(
    tmp_path: Path, simulate_calls: dict[str, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    touch(*simulate_inputs(tmp_path))
    flags = "--policies static pick-and-spin --seeds 2 --workers 8 --max-concurrency 3 --cooldown 10"
    flags += " --arrival-rate 0.25 4 --write-queries"
    assert main(["--root", str(tmp_path), "simulate", *flags.split()]) == 0
    out = tmp_path / "results" / "simulation"
    policies = [Policy.STATIC, Policy.PICK_AND_SPIN]
    assert simulate_calls["run_load"] == [
        (policies, [2], SimulationSettings(8, arrival_rate=0.25, max_concurrency=3, cooldown_s=10.0), out, True),
        (policies, [2], SimulationSettings(8, arrival_rate=4.0, max_concurrency=3, cooldown_s=10.0), out, True),
    ]
    assert simulate_calls["write_overview"] == [out]  # once, after every load


def test_simulate_without_an_input_returns_1_before_loading_anything(
    tmp_path: Path, simulate_calls: dict[str, list[Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    for missing in simulate_inputs(tmp_path):
        touch(*(p for p in simulate_inputs(tmp_path) if p != missing))
        missing.unlink(missing_ok=True)
        assert main(["--root", str(tmp_path), "simulate"]) == 1
        assert f"not found: {missing} (run from the repository root" in capsys.readouterr().err
    assert simulate_calls["load_trace_data"] == []


# --- simulate: real runs on a tiny trace ------------------------------------------------------------


def test_simulate_writes_every_load_and_an_overview_of_all_of_them(
    tiny_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tiny_root / "results" / "simulation"
    argv = ["--root", str(tiny_root), "simulate", "--policies", "static", "pick-and-spin", "--seeds", "0", "1"]
    assert main([*argv, "--workers", "4", "--write-queries"]) == 0
    assert sorted(p.name for p in (out / "closed-4").iterdir()) == [
        "cold_starts_by_tier.csv",
        "per_model.csv",
        "queries_pick-and-spin.csv.gz",  # for the first seed only
        "queries_static.csv.gz",
        "summary.csv",
    ]
    summary = read_csv(out / "closed-4" / "summary.csv")
    assert [(r["policy"], r["seed"]) for r in summary] == [
        ("static", "0"),
        ("static", "1"),
        ("pick-and-spin", "0"),
        ("pick-and-spin", "1"),
    ]
    assert len(read_csv(out / "closed-4" / "per_model.csv")) == 4 * len(MODELS)

    # A later run adds its loads; overview.csv then averages every load under --out.
    assert main([*argv, "--arrival-rate", "0.5", "2"]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["closed-4", "overview.csv", "poisson-0.5qps", "poisson-2qps"]
    overview = read_csv(out / "overview.csv")
    assert [(r["load"], r["policy"], r["seeds"]) for r in overview] == [
        (load, policy, "2")
        for load in ("closed-4", "poisson-0.5qps", "poisson-2qps")
        for policy in ("pick-and-spin", "static")
    ]


def test_simulate_writes_exactly_what_the_library_writes(
    tiny_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # A rate of 0 means the closed loop, as in v1.1.0.
    flags = "--policies unaware static --seeds 3 1 --workers 3 --max-concurrency 2 --cooldown 5 --arrival-rate 0 0.5"
    flags += " --write-queries"
    assert main(["--root", str(tiny_root), "simulate", *flags.split(), "--out", str(tmp_path / "cli")]) == 0

    data = load_trace_data(tiny_root / "data" / "queries.jsonl.gz", tiny_root / "results" / "traces")
    tiers = read_tier_cache(tiny_root / "data" / "query_tiers.csv.gz")
    for rate in (0.0, 0.5):
        settings = SimulationSettings(workers=3, arrival_rate=rate, max_concurrency=2, cooldown_s=5.0)
        run_load([Policy.UNAWARE, Policy.STATIC], [3, 1], settings, data, tiers, tmp_path / "lib", write_queries=True)
    write_overview(tmp_path / "lib")

    produced = tree(tmp_path / "cli")
    assert sorted(produced) == [
        "closed-3/cold_starts_by_tier.csv",
        "closed-3/per_model.csv",
        "closed-3/queries_static.csv.gz",
        "closed-3/queries_unaware.csv.gz",
        "closed-3/summary.csv",
        "overview.csv",
        "poisson-0.5qps/cold_starts_by_tier.csv",
        "poisson-0.5qps/per_model.csv",
        "poisson-0.5qps/queries_static.csv.gz",
        "poisson-0.5qps/queries_unaware.csv.gz",
        "poisson-0.5qps/summary.csv",
    ]
    assert produced == tree(tmp_path / "lib")


def test_simulate_logs_the_old_progress_lines_to_stderr_and_prints_nothing(
    tiny_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(tiny_root), "simulate", "--policies", "static", "--seeds", "0", "--workers", "4"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert lines[0] == "12 queries; classifier stages: keyword 8 (66.7%), distilbert 4 (33.3%)"
    assert lines[1].startswith(f"closed-4 {'static':24s} seed 0: ")
    assert lines[2:] == [f"Wrote {tiny_root / 'results' / 'simulation' / 'closed-4'}"]


def test_quiet_and_verbose_simulate(tiny_root: Path, capsys: pytest.CaptureFixture[str]) -> None:
    argv = ["simulate", "--policies", "static", "--seeds", "0", "--workers", "4"]
    assert main(["--root", str(tiny_root), "-q", *argv]) == 0
    assert capsys.readouterr() == ("", "")
    assert main(["--root", str(tiny_root), "-v", *argv]) == 0
    err = capsys.readouterr().err
    assert re.search(r"^\S+ \S+ DEBUG pickspin\.cli\.simulate: Reading ", err, re.MULTILINE)
    assert re.search(r"^\S+ \S+ INFO pickspin\.simulation\.experiment: closed-4 static ", err, re.MULTILINE)


def test_reclassifying_without_the_classifier_extra_returns_1(
    tiny_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)  # as if the [classifier] extra were not installed
    cache = tiny_root / "data" / "query_tiers.csv.gz"
    cached = cache.read_bytes()
    assert main(["--root", str(tiny_root), "simulate", "--reclassify", "--seeds", "0"]) == 1
    hint = "torch is required for this command: pip install 'pick-and-spin[classifier]'"
    assert capsys.readouterr().err.endswith(f"pickspin: error: {hint}\n")
    assert cache.read_bytes() == cached
    assert not (tiny_root / "results" / "simulation").exists()


# --- errors and exit codes ---------------------------------------------------------------------------


def replace_reproduce_handler(monkeypatch: pytest.MonkeyPatch, handler: Callable[[argparse.Namespace], int]) -> None:
    """Make `pickspin reproduce` run handler instead (register() picks up the module's run when called)."""
    monkeypatch.setattr(reproduce_command, "run", handler)


def test_a_pickspin_error_is_one_line_on_stderr_and_returns_1(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(args: argparse.Namespace) -> int:
        raise ConfigError("Set JUDGE_API_BASE first")

    replace_reproduce_handler(monkeypatch, fail)
    assert main(["reproduce"]) == 1
    assert capsys.readouterr() == ("", "pickspin: error: Set JUDGE_API_BASE first\n")


def test_verbose_also_logs_the_traceback_of_an_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail(args: argparse.Namespace) -> int:
        raise ConfigError("bad setting")

    replace_reproduce_handler(monkeypatch, fail)
    assert main(["-v", "reproduce"]) == 1
    err = capsys.readouterr().err
    assert "DEBUG pickspin.cli.main: pickspin reproduce failed\nTraceback (most recent call last):" in err
    assert err.endswith("pickspin: error: bad setting\n")


def test_the_handlers_exit_status_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    replace_reproduce_handler(monkeypatch, lambda args: 7)
    assert main(["reproduce"]) == 7


def test_ctrl_c_returns_130(monkeypatch: pytest.MonkeyPatch) -> None:
    def interrupted(args: argparse.Namespace) -> int:
        raise KeyboardInterrupt

    replace_reproduce_handler(monkeypatch, interrupted)
    assert main(["reproduce"]) == 130


def test_other_exceptions_are_not_swallowed(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken(args: argparse.Namespace) -> int:
        raise RuntimeError("a bug")

    replace_reproduce_handler(monkeypatch, broken)
    with pytest.raises(RuntimeError, match="a bug"):
        main(["reproduce"])


# --- logging -------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("verbosity", "level"), [(-1, logging.WARNING), (0, logging.INFO), (1, logging.DEBUG), (2, logging.DEBUG)]
)
def test_configure_logging_sends_the_pickspin_logger_to_stderr(
    verbosity: int, level: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("pickspin")
    monkeypatch.setattr(logger, "handlers", [])
    configure_logging(verbosity)
    assert logger.level == level
    assert logger.propagate is False
    [handler] = logger.handlers
    assert isinstance(handler, logging.StreamHandler)
    assert handler.stream is sys.stderr


def test_only_debug_lines_show_the_time_level_and_logger(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = logging.getLogger("pickspin")
    monkeypatch.setattr(logger, "handlers", [])
    record = logging.LogRecord("pickspin.simulation", logging.INFO, __file__, 1, "Wrote %s", ("x",), None)
    for verbosity in (-1, 0):
        configure_logging(verbosity)
        assert logger.handlers[0].format(record) == "Wrote x"
    configure_logging(1)
    line = logger.handlers[0].format(record)
    assert re.fullmatch(r"\d{4}-\d\d-\d\d \d\d:\d\d:\d\d,\d{3} INFO pickspin\.simulation: Wrote x", line), line


def test_configure_logging_twice_leaves_exactly_one_handler(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = logging.getLogger("pickspin")
    other = logging.NullHandler()  # a handler someone else installed stays
    monkeypatch.setattr(logger, "handlers", [other])
    configure_logging(0)
    configure_logging(1)
    assert len(logger.handlers) == 2
    assert logger.handlers[0] is other
    assert isinstance(logger.handlers[1], logging.StreamHandler)
    assert logger.level == logging.DEBUG
