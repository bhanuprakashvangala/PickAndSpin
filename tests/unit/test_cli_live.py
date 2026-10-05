"""`pickspin live`, `pickspin baseline run|judge` and `pickspin classifier labels|train|evaluate`.

The tests are hermetic: no network, no cluster, no trained model and no optional extra. The library call
behind a command is replaced with a recorder, except where a command fails before it reaches a server
or a model, and in `classifier labels`, which runs for real on a few synthetic queries. The runner,
the static baseline, the judge and the training pipeline are tested on their own in tests/unit and
tests/integration.
"""

from __future__ import annotations

import dataclasses
import gzip
import importlib.abc
import json
import re
import subprocess
import sys
import textwrap
import types
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pytest

import pickspin.training
from pickspin.baseline import judge as judge_module
from pickspin.baseline import static as static_module
from pickspin.baseline.judge import JUDGE_BENCHMARKS, JudgeConfig
from pickspin.cli.main import build_parser, main
from pickspin.config import MODELS
from pickspin.data import Query
from pickspin.errors import MissingDependencyError
from pickspin.live import runner as runner_module
from pickspin.live.runner import LiveConfig
from pickspin.spin.lifecycle import LatencySignal
from pickspin.training import evaluate as evaluate_module
from pickspin.training.evaluate import format_evaluation
from pickspin.training.labels import generate_labeled_dataset

# Every environment variable these commands read.
ENV_VARS = (
    "PS_ENDPOINTS",
    "PS_NAMESPACE",
    "PS_CLASSIFIER",
    "VLLM_API_KEY",
    "JUDGE_API_BASE",
    "JUDGE_API_KEY",
    "JUDGE_MODEL",
    "JUDGE_WORKERS",
)
# The hint that ends every 'not found' error.
NOT_FOUND = "(run from the repository root, pass --root DIR, or give the path explicitly)"


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Start every test with none of ENV_VARS set, whatever the developer's shell exports."""
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


# main() configures the 'pickspin' logger for the whole process; the autouse fixture
# restore_pickspin_logger in tests/conftest.py puts it back after each test.


def touch(*paths: Path) -> None:
    """Create empty files, with their directories, where a command only checks that its inputs exist."""
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def write_endpoints(path: Path, models: Iterable[str] = MODELS, *, host: str = "vllm") -> dict[str, Any]:
    """Write an endpoint map whose base URLs start with host (unresolvable on purpose); returns it."""
    endpoints = {
        m: {"base_url": f"http://{host}-{i}.invalid:8000", "model": MODELS[m].hf_id, "deployment": f"{host}-{i}"}
        for i, m in enumerate(models)
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(endpoints), encoding="utf-8")
    return endpoints


def write_queries(path: Path, n: int) -> list[Query]:
    """Write n benchmark queries as gzipped JSON lines, like data/queries.jsonl.gz; returns them."""
    queries = [
        Query(f"q{i}", "gsm8k" if i % 2 else "mbpp", f"What is {i} + {i}?", str(2 * i), "math") for i in range(n)
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(dataclasses.asdict(q)) + "\n")
    return queries


def help_text(argv: Sequence[str], capsys: pytest.CaptureFixture[str]) -> str:
    """Return the --help output of a command with its line wrapping undone."""
    with pytest.raises(SystemExit) as exited:
        main([*argv, "--help"])
    assert exited.value.code == 0
    return " ".join(capsys.readouterr().out.split())


def block_imports(monkeypatch: pytest.MonkeyPatch, *modules: str) -> None:
    """Make importing these modules fail, as if the extra that provides them were not installed."""
    for name in modules:
        monkeypatch.setitem(sys.modules, name, None)


# --- the command groups ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("group", "commands"), [("baseline", ["run", "judge"]), ("classifier", ["labels", "train", "evaluate"])]
)
def test_a_group_without_a_subcommand_prints_its_help_to_stderr_and_returns_2(
    group: str, commands: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main([group]) == 2
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith(f"usage: pickspin {group} [-h] <command> ...\n")
    for command in commands:
        assert re.search(rf"^    {command} +\S", err, re.MULTILINE), command


def test_the_help_of_these_commands_imports_no_optional_or_network_dependency() -> None:
    code = textwrap.dedent(
        """
        import sys
        from pickspin.cli.main import main
        for argv in (["live"], ["baseline"], ["baseline", "run"], ["baseline", "judge"], ["classifier"],
                     ["classifier", "labels"], ["classifier", "train"], ["classifier", "evaluate"]):
            try:
                main([*argv, "--help"])
            except SystemExit as e:
                assert e.code == 0, (argv, e.code)
        heavy = ("torch", "transformers", "kubernetes", "matplotlib", "requests", "datasets", "sklearn", "seaborn")
        print(sorted(m for m in heavy if m in sys.modules))
        """
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.splitlines()[-1] == "[]"


# --- live ----------------------------------------------------------------------------------------------


@pytest.fixture
def live_runs(monkeypatch: pytest.MonkeyPatch) -> list[LiveConfig]:
    """Replace run_live with a recorder; returns the configs it is called with."""
    configs: list[LiveConfig] = []

    def run_live(config: LiveConfig, **kwargs: Any) -> Path:
        assert kwargs == {}  # the command leaves the actuator, classifier and clock to run_live
        configs.append(config)
        return config.out_dir / "pick_spin_20260101_000000"

    monkeypatch.setattr(runner_module, "run_live", run_live)
    return configs


def live_root(root: Path) -> Path:
    """A repository root with the live runner's inputs: endpoint map, queries and model directory."""
    write_endpoints(root / "deploy" / "endpoints.example.json")
    write_queries(root / "data" / "queries.jsonl.gz", 3)
    (root / "models" / "distilbert-complexity-classifier").mkdir(parents=True)
    return root


def default_live_config(root: Path, **changes: Any) -> LiveConfig:
    """The config `pickspin --root <root> live` builds, with changes."""
    config = LiveConfig(
        endpoints=root / "deploy" / "endpoints.example.json",
        queries=root / "data" / "queries.jsonl.gz",
        out_dir=root / "results" / "live",
        model_dir=root / "models" / "distilbert-complexity-classifier",
    )
    return dataclasses.replace(config, **changes)


def test_live_defaults_match_v1_1_0() -> None:
    args = build_parser().parse_args(["live"])
    assert args.workers == 250
    assert args.max_tokens == 256
    assert args.latency_signal == "spin"
    assert type(args.latency_signal) is str  # converted to LatencySignal only after parsing
    assert args.seed == 0
    assert args.cooldown == 300.0
    assert isinstance(args.cooldown, float)
    assert args.limit is None
    assert args.static is False
    assert args.namespace == "pick-and-spin"
    assert (args.endpoints, args.out, args.queries, args.model_dir) == (None,) * 4


def test_live_flags() -> None:
    argv = ["live", "--namespace", "ns", "--workers", "8", "--limit", "60", "--max-tokens", "64", "--static"]
    argv += ["--latency-signal", "inference", "--seed", "3", "--cooldown", "0.5", "--endpoints", "e.json"]
    argv += ["--out", "o", "--queries", "q.jsonl.gz", "--model-dir", "m"]
    args = build_parser().parse_args(argv)
    assert (args.namespace, args.workers, args.limit, args.max_tokens, args.static) == ("ns", 8, 60, 64, True)
    assert (args.latency_signal, args.seed, args.cooldown) == ("inference", 3, 0.5)
    assert (args.endpoints, args.out, args.queries, args.model_dir) == tuple(
        map(Path, ["e.json", "o", "q.jsonl.gz", "m"])
    )


def test_an_unknown_latency_signal_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["live", "--latency-signal", "bogus"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'bogus' (choose from" in err
    assert "LatencySignal." not in err  # the choices are listed as the plain names


@pytest.mark.parametrize(("env", "expected"), [(None, "pick-and-spin"), ("", ""), ("team-a", "team-a")])
def test_the_namespace_defaults_to_ps_namespace_kept_even_when_empty(
    env: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_runs: list[LiveConfig],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if env is not None:
        monkeypatch.setenv("PS_NAMESPACE", env)
    assert build_parser().parse_args(["live"]).namespace == expected
    assert build_parser().parse_args(["live", "--namespace", "flag-ns"]).namespace == "flag-ns"
    root = live_root(tmp_path)
    assert main(["--root", str(root), "live"]) == 0
    assert live_runs[-1].namespace == expected


def test_live_resolves_its_defaults_under_root(
    tmp_path: Path, live_runs: list[LiveConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    root = live_root(tmp_path / "repo")
    assert main(["--root", str(root), "live"]) == 0
    assert live_runs == [default_live_config(root)]
    assert type(live_runs[0].latency_signal) is LatencySignal
    assert capsys.readouterr() == ("", "")  # the recorder logs nothing, and the command prints nothing


def test_live_passes_every_flag_and_uses_explicit_paths_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_runs: list[LiveConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    write_endpoints(Path("in/eps.json"))
    write_queries(Path("in/q.jsonl.gz"), 3)
    Path("in/model").mkdir()
    argv = ["--root", str(tmp_path / "elsewhere"), "live", "--endpoints", "in/eps.json", "--queries", "in/q.jsonl.gz"]
    argv += ["--model-dir", "in/model", "--out", "out", "--namespace", "ns", "--workers", "8", "--limit", "60"]
    argv += ["--max-tokens", "64", "--static", "--latency-signal", "observed", "--seed", "7", "--cooldown", "0.5"]
    assert main(argv) == 0
    expected = LiveConfig(
        endpoints=Path("in/eps.json"),
        queries=Path("in/q.jsonl.gz"),
        out_dir=Path("out"),
        model_dir=Path("in/model"),
        namespace="ns",
        workers=8,
        limit=60,
        max_tokens=64,
        static=True,
        latency_signal=LatencySignal.OBSERVED,
        seed=7,
        cooldown_s=0.5,
    )
    assert live_runs == [expected]


def test_a_live_limit_of_0_is_passed_on(
    tmp_path: Path, live_runs: list[LiveConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    # run_live routes every query when the limit is 0 or None (the shuffle comes first).
    root = live_root(tmp_path)
    assert main(["--root", str(root), "live", "--limit", "0"]) == 0
    assert live_runs == [default_live_config(root, limit=0)]


@pytest.mark.parametrize(
    ("flag", "env_var", "default", "field"),
    [
        ("--endpoints", "PS_ENDPOINTS", "deploy/endpoints.example.json", "endpoints"),
        ("--model-dir", "PS_CLASSIFIER", "models/distilbert-complexity-classifier", "model_dir"),
    ],
)
@pytest.mark.parametrize(
    ("given", "env", "expected"),
    [
        (None, None, "<root>"),
        (None, "", "<root>"),  # an empty variable counts as unset
        (None, "from-env", "from-env"),
        ("from-flag", "from-env", "from-flag"),
    ],
)
def test_live_paths_are_the_flag_then_the_environment_then_root(
    flag: str,
    env_var: str,
    default: str,
    field: str,
    given: str | None,
    env: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    live_runs: list[LiveConfig],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    root = live_root(tmp_path / "repo")
    write_endpoints(Path("from-env"))  # a file and a directory would both do: the recorder reads neither
    write_endpoints(Path("from-flag"))
    if env is not None:
        monkeypatch.setenv(env_var, env)
    assert main(["--root", str(root), "live", *([flag, given] if given else [])]) == 0
    [config] = live_runs
    assert getattr(config, field) == (root / default if expected == "<root>" else Path(expected))


def test_live_takes_the_api_key_from_vllm_api_key_and_never_logs_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, live_runs: list[LiveConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("VLLM_API_KEY", "sk-live-secret")
    root = live_root(tmp_path)
    assert main(["--root", str(root), "-v", "live"]) == 0
    assert live_runs == [default_live_config(root, api_key="sk-live-secret")]
    err = capsys.readouterr().err
    assert "DEBUG pickspin.cli.live: LiveConfig(" in err
    assert "sk-live-secret" not in err


@pytest.mark.parametrize(
    "missing", ["deploy/endpoints.example.json", "data/queries.jsonl.gz", "models/distilbert-complexity-classifier"]
)
def test_live_without_an_input_returns_1_before_running(
    missing: str, tmp_path: Path, live_runs: list[LiveConfig], capsys: pytest.CaptureFixture[str]
) -> None:
    root = live_root(tmp_path)
    path = root / missing
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    assert main(["--root", str(root), "live"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err.startswith("pickspin: error: ")
    assert f"not found: {path} {NOT_FOUND}" in err
    assert live_runs == []


def test_live_without_the_live_extra_returns_1_before_touching_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    block_imports(monkeypatch, "kubernetes", "kubernetes.client", "kubernetes.config")
    root = live_root(tmp_path)
    assert main(["--root", str(root), "live"]) == 1
    hint = "kubernetes.client is required for this command: pip install 'pick-and-spin[live]'"
    assert capsys.readouterr().err.endswith(f"pickspin: error: {hint}\n")
    assert not (root / "results").exists()


def test_a_static_live_run_without_the_classifier_extra_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A static run needs no kubernetes, but Pick still needs DistilBERT.
    block_imports(monkeypatch, "kubernetes", "kubernetes.client", "kubernetes.config", "torch", "transformers")
    root = live_root(tmp_path)
    assert main(["--root", str(root), "live", "--static"]) == 1
    hint = "torch is required for this command: pip install 'pick-and-spin[classifier]'"
    assert capsys.readouterr().err.endswith(f"pickspin: error: {hint}\n")
    assert not (root / "results").exists()


# --- baseline run --------------------------------------------------------------------------------------


@pytest.fixture
def static_runs(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """Replace run_static_baseline with a recorder; returns the arguments of each call by name."""
    calls: list[dict[str, Any]] = []

    def run_static_baseline(
        endpoints: Any,
        models: Any,
        queries: Any,
        out_dir: Path,
        *,
        workers: int = 50,
        max_tokens: int = 512,
        headers: Any,
    ) -> list[Path]:
        calls.append(
            {
                "endpoints": endpoints,
                "models": models,
                "queries": queries,
                "out_dir": out_dir,
                "workers": workers,
                "max_tokens": max_tokens,
                "headers": headers,
            }
        )
        return [out_dir / f"{m}_results.jsonl" for m in models]

    monkeypatch.setattr(static_module, "run_static_baseline", run_static_baseline)
    return calls


def test_baseline_run_defaults_match_v1_1_0() -> None:
    args = build_parser().parse_args(["baseline", "run"])
    assert args.models == list(MODELS)
    assert args.workers == 50
    assert args.max_tokens == 512
    assert args.limit is None
    assert (args.endpoints, args.queries, args.out) == (None,) * 3


def test_baseline_run_flags() -> None:
    argv = ["baseline", "run", "--models", "qwen2.5_7B", "llama3.2_1B", "--workers", "4", "--max-tokens", "32"]
    argv += ["--limit", "100", "--endpoints", "e.json", "--queries", "q.jsonl.gz", "--out", "o"]
    args = build_parser().parse_args(argv)
    assert args.models == ["qwen2.5_7B", "llama3.2_1B"]  # in the order given
    assert (args.workers, args.max_tokens, args.limit) == (4, 32, 100)
    assert (args.endpoints, args.queries, args.out) == (Path("e.json"), Path("q.jsonl.gz"), Path("o"))
    assert build_parser().parse_args(["baseline", "run", "--models"]).models == []


def test_an_unknown_model_is_a_usage_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["baseline", "run", "--models", "qwen2.5_7B", "gpt-4"])
    assert exited.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice: 'gpt-4' (choose from" in err
    assert "gemma3_27B" in err  # quoted or not, depending on the Python version


def test_baseline_run_resolves_its_defaults_under_root(
    tmp_path: Path, static_runs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    endpoints = write_endpoints(tmp_path / "deploy" / "endpoints.example.json")
    queries = write_queries(tmp_path / "data" / "queries.jsonl.gz", 5)
    assert main(["--root", str(tmp_path), "baseline", "run"]) == 0
    assert static_runs == [
        {
            "endpoints": endpoints,
            "models": list(MODELS),
            "queries": queries,
            "out_dir": tmp_path / "results" / "live" / "static",
            "workers": 50,
            "max_tokens": 512,
            "headers": {},
        }
    ]
    assert capsys.readouterr().out == ""


def test_baseline_run_passes_its_flags_and_uses_explicit_paths_as_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    static_runs: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    endpoints = write_endpoints(Path("in/eps.json"), ["qwen2.5_7B", "gemma2_2B"])
    queries = write_queries(Path("in/q.jsonl.gz"), 5)
    argv = ["--root", str(tmp_path / "elsewhere"), "baseline", "run", "--endpoints", "in/eps.json"]
    argv += ["--queries", "in/q.jsonl.gz", "--out", "out", "--models", "gemma2_2B", "qwen2.5_7B"]
    assert main([*argv, "--workers", "4", "--max-tokens", "32"]) == 0
    [call] = static_runs
    assert call["endpoints"] == endpoints
    assert call["models"] == ["gemma2_2B", "qwen2.5_7B"]
    assert call["queries"] == queries
    assert (call["out_dir"], call["workers"], call["max_tokens"]) == (Path("out"), 4, 32)


@pytest.mark.parametrize(("limit", "kept"), [(None, 5), (3, 3), (0, 0), (9, 5)])
def test_baseline_run_limit_slices_the_file_order(
    limit: int | None,
    kept: int,
    tmp_path: Path,
    static_runs: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    # Unlike live --limit, there is no shuffle, and 0 runs no query at all.
    write_endpoints(tmp_path / "deploy" / "endpoints.example.json")
    queries = write_queries(tmp_path / "data" / "queries.jsonl.gz", 5)
    argv = ["--root", str(tmp_path), "baseline", "run"] + ([] if limit is None else ["--limit", str(limit)])
    assert main(argv) == 0
    assert static_runs[0]["queries"] == queries[:kept]


@pytest.mark.parametrize(
    ("key", "headers"), [(None, {}), ("", {}), ("sk-static", {"Authorization": "Bearer sk-static"})]
)
def test_baseline_run_sends_vllm_api_key_as_a_bearer_token(
    key: str | None,
    headers: dict[str, str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    static_runs: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if key is not None:
        monkeypatch.setenv("VLLM_API_KEY", key)
    write_endpoints(tmp_path / "deploy" / "endpoints.example.json")
    write_queries(tmp_path / "data" / "queries.jsonl.gz", 1)
    assert main(["--root", str(tmp_path), "baseline", "run"]) == 0
    assert static_runs[0]["headers"] == headers


@pytest.mark.parametrize(
    ("given", "env", "expected"),
    [(None, None, "root"), (None, "", "root"), (None, "env.json", "env"), ("flag.json", "env.json", "flag")],
)
def test_baseline_run_endpoints_are_the_flag_then_ps_endpoints_then_root(
    given: str | None,
    env: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    static_runs: list[dict[str, Any]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "repo"
    maps = {
        "root": write_endpoints(root / "deploy" / "endpoints.example.json", host="root"),
        "env": write_endpoints(Path("env.json"), host="env"),
        "flag": write_endpoints(Path("flag.json"), host="flag"),
    }
    write_queries(root / "data" / "queries.jsonl.gz", 1)
    if env is not None:
        monkeypatch.setenv("PS_ENDPOINTS", env)
    assert main(["--root", str(root), "baseline", "run", *(["--endpoints", given] if given else [])]) == 0
    assert static_runs[0]["endpoints"] == maps[expected]


def test_baseline_run_needs_an_endpoint_for_every_model_it_runs(
    tmp_path: Path, static_runs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    endpoints_file = tmp_path / "deploy" / "endpoints.example.json"
    write_endpoints(endpoints_file, ["qwen2.5_7B"])
    write_queries(tmp_path / "data" / "queries.jsonl.gz", 1)
    assert main(["--root", str(tmp_path), "baseline", "run", "--models", "qwen2.5_7B"]) == 0
    assert main(["--root", str(tmp_path), "baseline", "run", "--models", "qwen2.5_7B", "gemma2_9B", "gemma2_2B"]) == 1
    assert capsys.readouterr().err.endswith(
        f"pickspin: error: {endpoints_file} has no endpoint for gemma2_9B, gemma2_2B\n"
    )
    assert len(static_runs) == 1  # nothing ran for the second command


@pytest.mark.parametrize("missing", ["deploy/endpoints.example.json", "data/queries.jsonl.gz"])
def test_baseline_run_without_an_input_returns_1(
    missing: str, tmp_path: Path, static_runs: list[dict[str, Any]], capsys: pytest.CaptureFixture[str]
) -> None:
    write_endpoints(tmp_path / "deploy" / "endpoints.example.json")
    write_queries(tmp_path / "data" / "queries.jsonl.gz", 1)
    (tmp_path / missing).unlink()
    assert main(["--root", str(tmp_path), "baseline", "run"]) == 1
    assert f"not found: {tmp_path / missing} {NOT_FOUND}" in capsys.readouterr().err
    assert static_runs == []


# --- baseline judge ------------------------------------------------------------------------------------


@pytest.fixture
def judged(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, JudgeConfig, Path, Path]]:
    """Replace run_benchmark with a recorder; returns the arguments of each call."""
    calls: list[tuple[str, JudgeConfig, Path, Path]] = []

    def run_benchmark(benchmark: str, config: JudgeConfig, responses_dir: Path, out_dir: Path) -> Path:
        calls.append((benchmark, config, responses_dir, out_dir))
        return out_dir / f"{benchmark}_accuracy.jsonl"

    monkeypatch.setattr(judge_module, "run_benchmark", run_benchmark)
    return calls


@pytest.fixture
def judge_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Configure the judge in the environment and create the default responses directory; returns the root."""
    monkeypatch.setenv("JUDGE_API_BASE", "http://judge.invalid/v1/")
    monkeypatch.setenv("JUDGE_API_KEY", "sk-judge-secret")
    (tmp_path / "results" / "live" / "static").mkdir(parents=True)
    return tmp_path


def test_baseline_judge_defaults() -> None:
    args = build_parser().parse_args(["baseline", "judge"])
    assert not args.benchmarks  # every benchmark
    assert args.workers is None  # $JUDGE_WORKERS, else 100
    assert (args.responses, args.out) == (None, None)
    args = build_parser().parse_args(["baseline", "judge", "mbpp", "gsm8k", "--workers", "8"])
    assert (args.benchmarks, args.workers) == (["mbpp", "gsm8k"], 8)


def test_baseline_judge_without_judge_api_base_returns_1(
    tmp_path: Path, judged: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "results" / "live" / "static").mkdir(parents=True)
    assert main(["--root", str(tmp_path), "baseline", "judge"]) == 1
    assert capsys.readouterr() == (
        "",
        "pickspin: error: Set JUDGE_API_BASE (and JUDGE_API_KEY) first; see .env.example\n",
    )
    assert judged == []


def test_baseline_judge_runs_every_benchmark_in_the_old_order_under_root(
    judge_env: Path, judged: list[tuple[str, JudgeConfig, Path, Path]], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(judge_env), "baseline", "judge"]) == 0
    config = JudgeConfig(api_base="http://judge.invalid/v1", api_key="sk-judge-secret", model="gpt-oss", workers=100)
    responses = judge_env / "results" / "live" / "static"
    out = judge_env / "results" / "live" / "judgments"
    assert judged == [(b, config, responses, out) for b in JUDGE_BENCHMARKS]
    assert [b for b, *_ in judged] == [
        "gsm8k",
        "math",
        "arc",
        "mmlu_pro",
        "hellaswag",
        "truthfulqa",
        "humaneval",
        "mbpp",
    ]
    assert capsys.readouterr().out == ""


def test_baseline_judge_runs_the_named_benchmarks_in_the_given_order(
    judge_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    judged: list[tuple[str, JudgeConfig, Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(judge_env)
    Path("my responses").mkdir()
    argv = ["--root", str(judge_env / "elsewhere"), "baseline", "judge", "mbpp", "gsm8k", "mbpp"]
    assert main([*argv, "--responses", "my responses", "--out", "labels"]) == 0
    assert [(b, responses, out) for b, _, responses, out in judged] == [
        ("mbpp", Path("my responses"), Path("labels")),
        ("gsm8k", Path("my responses"), Path("labels")),
        ("mbpp", Path("my responses"), Path("labels")),
    ]


@pytest.mark.parametrize(
    ("env", "flag", "workers"), [(None, None, 100), ("12", None, 12), ("12", "3", 3), (None, "3", 3)]
)
def test_baseline_judge_workers_are_the_flag_then_judge_workers_then_100(
    env: str | None,
    flag: str | None,
    workers: int,
    judge_env: Path,
    monkeypatch: pytest.MonkeyPatch,
    judged: list[tuple[str, JudgeConfig, Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    if env is not None:
        monkeypatch.setenv("JUDGE_WORKERS", env)
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    argv = ["--root", str(judge_env), "baseline", "judge", "arc", *(["--workers", flag] if flag else [])]
    assert main(argv) == 0
    [(_, config, _, _)] = judged
    assert config == JudgeConfig("http://judge.invalid/v1", "sk-judge-secret", "judge-model", workers)


def test_baseline_judge_with_an_unknown_benchmark_returns_1_before_judging_anything(
    judge_env: Path, judged: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(judge_env), "baseline", "judge", "gsm8k", "gsm8kk", "squad"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert err == f"pickspin: error: unknown benchmark: gsm8kk, squad (choose from {', '.join(JUDGE_BENCHMARKS)})\n"
    assert judged == []
    assert not (judge_env / "results" / "live" / "judgments").exists()


def test_baseline_judge_without_the_responses_returns_1(
    judge_env: Path, judged: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    responses = judge_env / "results" / "live" / "static"
    responses.rmdir()
    assert main(["--root", str(judge_env), "baseline", "judge"]) == 1
    assert f"not found: {responses} {NOT_FOUND}" in capsys.readouterr().err
    assert judged == []


def test_baseline_judge_never_logs_the_api_key(
    judge_env: Path, judged: list[Any], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["--root", str(judge_env), "-v", "baseline", "judge", "math"]) == 0
    err = capsys.readouterr().err
    assert "DEBUG pickspin.cli.baseline: JudgeConfig(api_base='http://judge.invalid/v1'" in err
    assert "sk-judge-secret" not in err


def test_baseline_judge_help_lists_the_benchmarks_in_the_judges_order(capsys: pytest.CaptureFixture[str]) -> None:
    assert f"(default: all eight: {', '.join(JUDGE_BENCHMARKS)})" in help_text(["baseline", "judge"], capsys)


# --- classifier labels ---------------------------------------------------------------------------------


def write_label_inputs(judgments: Path, queries: Path) -> None:
    """Write judge labels for 10 queries: query i is correct on qwen2.5_1.5B (SIMPLE) when i % 3 == 0,
    else on llama3.1_8B (MEDIUM) when i % 3 == 1, else on no model (COMPLEX)."""
    write_queries(queries, 10)
    judgments.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(judgments, "wt", encoding="utf-8", newline="") as f:
        f.write("id,model,is_correct\r\n")
        for i in range(10):
            f.write(f"q{i},qwen2.5_1.5B,{int(i % 3 == 0)}\r\nq{i},llama3.1_8B,{int(i % 3 == 1)}\r\n")


def test_classifier_labels_defaults() -> None:
    args = build_parser().parse_args(["classifier", "labels"])
    assert (args.judgments, args.queries, args.out) == (None, None, None)


def test_classifier_labels_writes_what_the_library_writes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    root = tmp_path / "repo"
    write_label_inputs(root / "results" / "traces" / "judgments.csv.gz", root / "data" / "queries.jsonl.gz")
    assert main(["--root", str(root), "classifier", "labels"]) == 0
    out, err = capsys.readouterr()
    assert out == ""
    rule = "=" * 60
    assert err.startswith(f"{rule}\nGENERATING COMPLEXITY LABELS FOR DISTILBERT\n{rule}\nLoaded 20 accuracy entries\n")

    generate_labeled_dataset(
        root / "results" / "traces" / "judgments.csv.gz", root / "data" / "queries.jsonl.gz", tmp_path / "lib"
    )
    produced = root / "data" / "classifier"
    assert sorted(p.name for p in produced.iterdir()) == ["label_stats.json", "train.jsonl", "val.jsonl"]
    for name in ("label_stats.json", "train.jsonl", "val.jsonl"):
        assert (produced / name).read_bytes() == (tmp_path / "lib" / name).read_bytes(), name


def test_classifier_labels_uses_explicit_paths_as_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    write_label_inputs(Path("in/judgments.csv.gz"), Path("in/queries.jsonl.gz"))
    argv = ["--root", str(tmp_path / "elsewhere"), "classifier", "labels", "--judgments", "in/judgments.csv.gz"]
    assert main([*argv, "--queries", "in/queries.jsonl.gz", "--out", "split"]) == 0
    assert sorted(p.name for p in Path("split").iterdir()) == ["label_stats.json", "train.jsonl", "val.jsonl"]
    assert json.loads(Path("split/label_stats.json").read_text(encoding="utf-8"))["total_samples"] == 10


@pytest.mark.parametrize("missing", ["results/traces/judgments.csv.gz", "data/queries.jsonl.gz"])
def test_classifier_labels_without_an_input_returns_1_before_writing_anything(
    missing: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_label_inputs(tmp_path / "results" / "traces" / "judgments.csv.gz", tmp_path / "data" / "queries.jsonl.gz")
    (tmp_path / missing).unlink()
    assert main(["--root", str(tmp_path), "classifier", "labels"]) == 1
    assert f"not found: {tmp_path / missing} {NOT_FOUND}" in capsys.readouterr().err
    assert not (tmp_path / "data" / "classifier").exists()


# --- classifier train ----------------------------------------------------------------------------------


@pytest.fixture
def trainings(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    """Put a fake pickspin.training.finetune in place, so no [train] extra is needed; returns its train() calls."""
    calls: list[tuple[Path, Path]] = []

    def train(data_dir: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
        assert kwargs == {}  # the command keeps the default base model
        calls.append((data_dir, output_dir))
        return {}

    fake = types.SimpleNamespace(train=train)
    monkeypatch.setitem(sys.modules, "pickspin.training.finetune", fake)
    monkeypatch.setattr(pickspin.training, "finetune", fake, raising=False)
    return calls


class FailingImport(importlib.abc.MetaPathFinder):
    """Makes importing pickspin.training.finetune raise an error, as a missing dependency would."""

    def __init__(self, error: ImportError) -> None:
        self.error = error

    def find_spec(self, fullname: str, path: Sequence[str] | None, target: types.ModuleType | None = None) -> None:
        if fullname == "pickspin.training.finetune":
            raise self.error


def fail_finetune_import(monkeypatch: pytest.MonkeyPatch, error: ImportError) -> None:
    """Make `from pickspin.training import finetune` raise error, even when the module was imported before."""
    monkeypatch.delitem(sys.modules, "pickspin.training.finetune", raising=False)
    monkeypatch.delattr(pickspin.training, "finetune", raising=False)
    monkeypatch.setattr(sys, "meta_path", [FailingImport(error), *sys.meta_path])


def not_installed(module: str) -> ModuleNotFoundError:
    """The error Python raises when a module is not installed."""
    return ModuleNotFoundError(f"No module named {module!r}", name=module)


def touch_split(data: Path) -> None:
    touch(data / "train.jsonl", data / "val.jsonl")


def test_classifier_train_defaults() -> None:
    args = build_parser().parse_args(["classifier", "train"])
    assert (args.data, args.out) == (None, None)


def test_classifier_train_resolves_its_defaults_under_root(
    tmp_path: Path, trainings: list[tuple[Path, Path]], capsys: pytest.CaptureFixture[str]
) -> None:
    touch_split(tmp_path / "data" / "classifier")
    assert main(["--root", str(tmp_path), "classifier", "train"]) == 0
    assert trainings == [(tmp_path / "data" / "classifier", tmp_path / "models")]
    assert capsys.readouterr() == ("", "")


def test_classifier_train_uses_explicit_paths_as_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainings: list[tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    touch_split(Path("split"))
    assert main(["--root", str(tmp_path / "elsewhere"), "classifier", "train", "--data", "split", "--out", "m"]) == 0
    assert trainings == [(Path("split"), Path("m"))]


@pytest.mark.parametrize("missing", ["train.jsonl", "val.jsonl"])
def test_classifier_train_without_the_split_returns_1_before_training(
    missing: str, tmp_path: Path, trainings: list[tuple[Path, Path]], capsys: pytest.CaptureFixture[str]
) -> None:
    # The old script printed an error and exited with status 0 here.
    data = tmp_path / "data" / "classifier"
    touch_split(data)
    (data / missing).unlink()
    assert main(["--root", str(tmp_path), "classifier", "train"]) == 1
    err = capsys.readouterr().err
    assert err.startswith("pickspin: error: ")
    assert f"from `pickspin classifier labels` not found: {data / missing} {NOT_FOUND}" in err
    assert trainings == []


def test_classifier_train_without_the_train_extra_returns_1_before_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The real fine-tuning module, which imports the [train] extra when training starts.
    touch_split(tmp_path / "data" / "classifier")
    block_imports(monkeypatch, "torch", "transformers", "datasets")
    assert main(["--root", str(tmp_path), "classifier", "train"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    hint = r"pickspin: error: \S+ is required for this command: pip install 'pick-and-spin\[train\]'"
    assert re.fullmatch(hint, err.splitlines()[-1]), err
    assert not (tmp_path / "models").exists()


def test_a_failed_import_of_the_trainer_becomes_a_hint_to_install_the_train_extra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch_split(tmp_path / "data" / "classifier")
    fail_finetune_import(monkeypatch, not_installed("datasets"))
    assert main(["--root", str(tmp_path), "classifier", "train"]) == 1
    hint = "datasets is required for this command: pip install 'pick-and-spin[train]'"
    assert capsys.readouterr() == ("", f"pickspin: error: {hint}\n")


def test_classifier_train_passes_on_a_missing_dependency_error_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Such an error already names the module and its extra, which need not be [train].
    touch_split(tmp_path / "data" / "classifier")
    hint = "transformers is required for this command: pip install 'pick-and-spin[classifier]'"
    fail_finetune_import(monkeypatch, MissingDependencyError(hint))
    assert main(["--root", str(tmp_path), "classifier", "train"]) == 1
    assert capsys.readouterr() == ("", f"pickspin: error: {hint}\n")


def test_classifier_train_does_not_blame_an_extra_for_a_broken_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    touch_split(tmp_path / "data" / "classifier")
    fail_finetune_import(monkeypatch, not_installed("pickspin.training.helpers"))
    with pytest.raises(ModuleNotFoundError, match=r"pickspin\.training\.helpers"):
        main(["--root", str(tmp_path), "classifier", "train"])
    assert capsys.readouterr().err == ""  # no 'pip install' hint


# --- classifier evaluate -------------------------------------------------------------------------------

# A made-up evaluation, shaped like pickspin.training.evaluate.evaluate's result.
EVALUATION = {
    "validation_queries": 4,
    "label_counts": {"SIMPLE": 2, "COMPLEX": 1, "MEDIUM": 1},
    "distilbert_accuracy": 0.5,
    "hybrid_accuracy": 0.75,
    "keyword_coverage": 0.25,
    "keyword_accuracy_on_matched": 1.0,
    "majority_class_accuracy": 0.5,
}


@pytest.fixture
def evaluations(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path]]:
    """Replace evaluate() with one that returns EVALUATION; returns its calls."""
    calls: list[tuple[Path, Path]] = []

    def evaluate(val_path: Path, model_dir: Path) -> dict[str, Any]:
        calls.append((val_path, model_dir))
        return dict(EVALUATION)

    monkeypatch.setattr(evaluate_module, "evaluate", evaluate)
    return calls


def evaluate_root(root: Path) -> Path:
    """A repository root with a validation split and a (dummy) model directory."""
    val = root / "data" / "classifier" / "val.jsonl"
    val.parent.mkdir(parents=True)
    val.write_text('{"id": "q0", "text": "What is 1 + 1?", "label": "SIMPLE", "benchmark": "gsm8k"}\n', "utf-8")
    (root / "models" / "distilbert-complexity-classifier").mkdir(parents=True)
    return root


def test_classifier_evaluate_defaults() -> None:
    args = build_parser().parse_args(["classifier", "evaluate"])
    assert (args.val, args.model_dir, args.out) == (None, None, None)


def test_classifier_evaluate_prints_the_table_and_writes_the_json_under_root(
    tmp_path: Path, evaluations: list[tuple[Path, Path]], capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluate_root(tmp_path)
    assert main(["--root", str(root), "classifier", "evaluate"]) == 0
    assert evaluations == [
        (root / "data" / "classifier" / "val.jsonl", root / "models" / "distilbert-complexity-classifier")
    ]
    out, err = capsys.readouterr()
    assert out == format_evaluation(EVALUATION) + "\n"
    assert out.startswith("validation_queries           4\nlabel_counts ")
    assert "\ndistilbert_accuracy          0.5000\n" in out
    assert err == ""
    written = root / "results" / "classifier" / "evaluation.json"  # results/classifier/ is created
    assert written.read_text(encoding="utf-8") == json.dumps(EVALUATION, indent=1) + "\n"


def test_classifier_evaluate_uses_explicit_paths_as_typed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluations: list[tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    touch(Path("in/val.jsonl"))
    Path("in/model").mkdir()
    argv = ["--root", str(tmp_path / "elsewhere"), "classifier", "evaluate", "--val", "in/val.jsonl"]
    assert main([*argv, "--model-dir", "in/model", "--out", "report/eval.json"]) == 0
    assert evaluations == [(Path("in/val.jsonl"), Path("in/model"))]
    assert json.loads(Path("report/eval.json").read_text(encoding="utf-8")) == EVALUATION


@pytest.mark.parametrize(
    ("given", "env", "expected"),
    [
        (None, None, "<root>"),
        (None, "", "<root>"),  # an empty variable counts as unset
        (None, "env-model", "env-model"),
        ("flag-model", "env-model", "flag-model"),
    ],
)
def test_classifier_evaluate_model_dir_is_the_flag_then_ps_classifier_then_root(
    given: str | None,
    env: str | None,
    expected: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    evaluations: list[tuple[Path, Path]],
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.chdir(tmp_path)
    root = evaluate_root(tmp_path / "repo")
    Path("env-model").mkdir()
    Path("flag-model").mkdir()
    if env is not None:
        monkeypatch.setenv("PS_CLASSIFIER", env)
    assert main(["--root", str(root), "classifier", "evaluate", *(["--model-dir", given] if given else [])]) == 0
    default = root / "models" / "distilbert-complexity-classifier"
    assert evaluations[0][1] == (default if expected == "<root>" else Path(expected))


@pytest.mark.parametrize("missing", ["data/classifier/val.jsonl", "models/distilbert-complexity-classifier"])
def test_classifier_evaluate_without_an_input_returns_1(
    missing: str, tmp_path: Path, evaluations: list[tuple[Path, Path]], capsys: pytest.CaptureFixture[str]
) -> None:
    root = evaluate_root(tmp_path)
    path = root / missing
    if path.is_dir():
        path.rmdir()
    else:
        path.unlink()
    assert main(["--root", str(root), "classifier", "evaluate"]) == 1
    out, err = capsys.readouterr()
    assert out == ""
    assert f"not found: {path} {NOT_FOUND}" in err
    assert evaluations == []


def test_classifier_evaluate_without_the_classifier_extra_returns_1(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    block_imports(monkeypatch, "torch", "transformers")
    root = evaluate_root(tmp_path)
    assert main(["--root", str(root), "classifier", "evaluate"]) == 1
    hint = "torch is required for this command: pip install 'pick-and-spin[classifier]'"
    assert capsys.readouterr() == ("", f"pickspin: error: {hint}\n")
    assert not (root / "results").exists()
