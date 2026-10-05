"""The paper reproduction (pickspin.paper) on synthetic traces: constants, analysis, checks and outputs.

These tests are hermetic. The equivalence with the released results is checked by the integration
test of `pickspin reproduce`.
"""

import csv
import gzip
import hashlib
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from pickspin.config import MODELS
from pickspin.paper.analysis import (
    ModelBaseline,
    PaperTraces,
    RoutedModelStats,
    RoutingTable,
    convergence,
    load_paper_traces,
    routing_table,
    size_class_totals,
    static_summary,
)
from pickspin.paper.constants import (
    BENCHMARK_COLUMNS,
    CONVERGENCE_MODELS,
    CONVERGENCE_STEP,
    DISPLAY_NAME,
    PAPER_FIG2A_TOK_S,
    PAPER_FIG2B_LATENCY_S,
    PAPER_MODELS,
    PAPER_TABLE1,
    ROUTED_DISPLAY_NAME,
    ROUTED_MODEL_ALIAS,
    SIZE_CLASS,
    canonical_model,
)
from pickspin.paper.figures import SIZE_COLORS, make_figures
from pickspin.paper.reproduce import (
    Check,
    ReproductionReport,
    build_checks,
    format_report,
    reproduce,
    write_table,
    write_tables,
    write_verification,
)

STATIC_HEADER = ["model", "id", "benchmark", "success", "latency_ms", "prompt_tokens", "completion_tokens", "error"]
JUDGMENTS_HEADER = ["model", "id", "benchmark", "is_correct"]
ROUTED_HEADER = [
    "order", "id", "benchmark", "tier", "model", "success", "tokens", "latency", "was_cold", "cold_penalty",
    "total_latency",
]  # fmt: skip

# sha256 of the 84 '<claim>\t<paper value>' lines, joined by newlines, of results/verification.csv as
# written by scripts/reproduce.py at tag v1.1.0.
CLAIMS_SHA256 = "46aba73af1b32911e63c9ac49cb8f28b3a7cbd5a9be2bbf75c842b2afa0de8b0"

# A routed run of four queries and its judge labels.
ROUTED = [
    {"order": "0", "id": "q0", "model": "gemma2_27B", "total_latency": "10.0"},
    {"order": "1", "id": "q1", "model": "llama3.2_1B", "total_latency": "2.0"},
    {"order": "2", "id": "q2", "model": "qwen2.5_7B", "total_latency": "4.0"},
    {"order": "3", "id": "q3", "model": "llama3.1_8B", "total_latency": "8.0"},
]
JUDGED = {
    ("q0", "gemma3_27B"): True,  # the routed run's gemma2_27B is scored with the 27B slot's label
    ("q0", "llama3.2_1B"): False,
    ("q1", "llama3.2_1B"): False,
    ("q1", "qwen2.5_7B"): True,  # another model solves q1, so the oracle does
    ("q2", "llama3.2_1B"): False,  # q2 has a label, but not for the model it was routed to
    ("q3", "llama3_70B"): True,  # labels of models outside the nine slots are ignored
}


def write_csv_gz(path: Path, header: Sequence[str], rows: Sequence[Sequence[str]]) -> None:
    """Write a gzipped CSV file in the format of the released traces."""
    with gzip.open(path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def png_size(path: Path) -> tuple[int, int]:
    """Width and height in pixels, from the PNG header."""
    data = path.read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    assert data[12:16] == b"IHDR"
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


# Constants: drift alarms against the catalog, and pins where the paper must not change.


def test_paper_models_follow_the_catalog() -> None:
    assert tuple(MODELS) == PAPER_MODELS
    assert list(DISPLAY_NAME) == list(PAPER_MODELS)
    for m in PAPER_MODELS:
        assert DISPLAY_NAME[m] == MODELS[m].label


def test_routed_display_name_differs_only_in_the_27b_slot() -> None:
    assert list(ROUTED_DISPLAY_NAME) == list(PAPER_MODELS)
    changed = {m: name for m, name in ROUTED_DISPLAY_NAME.items() if name != DISPLAY_NAME[m]}
    assert changed == {"gemma3_27B": "Gemma-2-27B"}


def test_size_classes_benchmarks_and_figure_settings_are_pinned() -> None:
    assert list(SIZE_CLASS.items()) == [
        ("llama3.2_1B", "1-3B"),
        ("qwen2.5_1.5B", "1-3B"),
        ("gemma2_2B", "1-3B"),
        ("llama3.2_3B", "1-3B"),
        ("qwen2.5_7B", "7-9B"),
        ("llama3.1_8B", "7-9B"),
        ("gemma2_9B", "7-9B"),
        ("qwen2.5_14B", "14-27B"),
        ("gemma3_27B", "14-27B"),
    ]
    assert BENCHMARK_COLUMNS == ("humaneval", "mbpp", "gsm8k", "math", "truthfulqa", "mmlu_pro", "arc", "hellaswag")
    assert CONVERGENCE_MODELS == ("llama3.1_8B", "qwen2.5_7B", "llama3.2_1B", "gemma2_9B")
    assert CONVERGENCE_STEP == 500
    assert dict(SIZE_COLORS) == {"1-3B": "#4C72B0", "7-9B": "#DD8452", "14-27B": "#55A868"}


def test_published_values_print_as_in_the_paper() -> None:
    assert f"{PAPER_TABLE1['llama3.2_3B'][2]}" == "20.0"
    assert list(PAPER_TABLE1) == list(PAPER_MODELS)
    assert [[f"{v}" for v in row] for row in PAPER_TABLE1.values()] == [
        ["5319", "17.1", "23.2", "1.82"],
        ["11", "0.04", "27.3", "4.22"],
        ["7", "0.02", "28.6", "3.34"],
        ["5", "0.02", "20.0", "8.29"],
        ["8101", "26.1", "63.3", "28.53"],
        ["13462", "43.4", "54.5", "28.87"],
        ["499", "1.6", "73.5", "30.52"],
        ["347", "1.1", "72.0", "44.63"],
        ["633", "2.0", "75.2", "13.28"],
    ]
    assert len(PAPER_FIG2A_TOK_S) == len(PAPER_FIG2B_LATENCY_S) == len(PAPER_MODELS)


def test_canonical_model_maps_only_gemma2_27b() -> None:
    assert dict(ROUTED_MODEL_ALIAS) == {"gemma2_27B": "gemma3_27B"}
    assert canonical_model("gemma2_27B") == "gemma3_27B"
    for m in (*PAPER_MODELS, "llama3_70B", "kimi_1T_MoE", "unknown"):
        assert canonical_model(m) == m


def test_check_matches_ignores_commas_and_x() -> None:
    assert Check("queries", "31,019", "31019").matches
    assert Check("ratio", "10x", "10").matches
    assert Check("ratio", "4.5x", "4.5x").matches
    assert not Check("latency", "1.51", "1.52").matches
    assert not Check("queries", "2,635", "2,636").matches


# Analysis.


def test_load_paper_traces_keeps_the_reproduction_parsing_rules(tmp_path: Path) -> None:
    write_csv_gz(
        tmp_path / "static_baseline.csv.gz",
        STATIC_HEADER,
        [
            ["llama3_70B", "q1", "gsm8k", "1", "900.5", "3", "7", ""],
            ["llama3.2_1B", "q1", "gsm8k", "1", "100.0", "3", "7", ""],
            ["llama3_70B", "q2", "mbpp", "0", "0", "3", "0", "HTTP 500"],
        ],
    )
    write_csv_gz(
        tmp_path / "judgments.csv.gz",
        JUDGMENTS_HEADER,
        [
            ["llama3.2_1B", "q1", "gsm8k", "1"],
            ["kimi_1T_MoE", "q1", "gsm8k", "0"],
            ["llama3.2_1B", "q2", "mbpp", "True"],  # only '1' counts as correct here
        ],
    )
    write_csv_gz(
        tmp_path / "pick_spin_routed.csv.gz",
        ROUTED_HEADER,
        [
            [order, f"q{order}", "gsm8k", "SIMPLE", "llama3.2_1B", "1", "7", "0.5", "0", "0.0", "0.5"]
            for order in ["10", "9", "0"]
        ],
    )

    traces = load_paper_traces(tmp_path)

    assert list(traces.static) == ["llama3_70B", "llama3.2_1B"]  # every model of the file, in file order
    assert [r["id"] for r in traces.static["llama3_70B"]] == ["q1", "q2"]
    assert traces.static["llama3_70B"][0]["latency_ms"] == "900.5"  # values stay strings
    assert traces.static["gemma2_9B"] == []  # a missing model has no runs
    assert traces.judged == {("q1", "llama3.2_1B"): True, ("q1", "kimi_1T_MoE"): False, ("q2", "llama3.2_1B"): False}
    assert [r["order"] for r in traces.routed] == ["0", "9", "10"]  # sorted by the order column as a number


def test_static_summary_counts_runs_with_output() -> None:
    def run(qid: str, benchmark: str, latency_ms: str, tokens: str) -> dict[str, str]:
        return {"id": qid, "benchmark": benchmark, "latency_ms": latency_ms, "completion_tokens": tokens}

    static = {
        m: [
            run("q0", "gsm8k", "2000", "10"),
            run("q1", "mbpp", "500", "20"),
            run("q2", "gsm8k", "0", "5"),  # no latency: not counted
            run("q3", "arc", "1000", "0"),  # no tokens: not counted
        ]
        for m in (*PAPER_MODELS, "llama3_70B")
    }
    judged = {(q, m): q in ("q0", "q2") for q in ("q0", "q1", "q2") for m in PAPER_MODELS}
    del judged[("q0", "gemma3_27B")]  # a counted run without a label counts as wrong

    summary, per_bench = static_summary(PaperTraces(static=static, judged=judged, routed=[]))

    assert list(summary) == list(per_bench) == list(PAPER_MODELS)
    assert summary["llama3.2_1B"] == ModelBaseline(runs=4, counted=2, accuracy=50.0, tok_s=22.5, latency_s=1.25)
    assert summary["gemma3_27B"].accuracy == 0.0
    assert list(per_bench["llama3.2_1B"].items()) == [("gsm8k", 100.0), ("mbpp", 0.0)]
    assert per_bench["gemma3_27B"] == {"gsm8k": 0.0, "mbpp": 0.0}


def test_routing_table_scores_each_query_with_its_slot_label() -> None:
    table = routing_table(ROUTED, JUDGED)

    assert list(table.per_model) == list(PAPER_MODELS)
    assert table.per_model["gemma3_27B"] == RoutedModelStats(n=1, correct=1, lat=[10.0])
    assert table.per_model["llama3.2_1B"] == RoutedModelStats(n=1, correct=0, lat=[2.0])
    assert all(table.per_model[m] == RoutedModelStats() for m in PAPER_MODELS[1:-1])
    assert (table.with_any, table.oracle_ok, table.ps_ok) == (3, 2, 1)
    assert table.any_lat == [10.0, 2.0, 4.0]
    assert table.judged_total == 2


def test_size_class_totals_follow_the_size_classes_in_model_order() -> None:
    totals = size_class_totals(routing_table(ROUTED, JUDGED))
    assert list(totals.items()) == [("1-3B", 1), ("7-9B", 0), ("14-27B", 1)]


def test_convergence_counts_the_raw_model_keys() -> None:
    xs, ys = convergence(ROUTED, track=("llama3.2_1B", "gemma2_27B", "gemma3_27B"), step=2)
    assert xs == [0.002, 0.004]
    assert ys == {"llama3.2_1B": [50.0, 25.0], "gemma2_27B": [50.0, 25.0], "gemma3_27B": [0.0, 0.0]}

    xs, ys = convergence(ROUTED)  # fewer queries than one step of 500
    assert xs == []
    assert ys == {m: [] for m in CONVERGENCE_MODELS}


# Checks and report.


def published_inputs() -> dict[str, Any]:
    """Keyword arguments for build_checks that carry the values printed in the paper."""
    tok_s = dict(zip(PAPER_MODELS, map(float, PAPER_FIG2A_TOK_S)))
    latency_s = dict(zip(PAPER_MODELS, map(float, PAPER_FIG2B_LATENCY_S)))
    latency_s |= {"llama3.2_1B": 1.51, "qwen2.5_7B": 4.83, "gemma3_27B": 21.66}  # quoted with two decimals
    accuracy = dict.fromkeys(PAPER_MODELS, 50.0) | {"qwen2.5_14B": 62.6, "gemma3_27B": 56.6, "qwen2.5_7B": 56.3}
    summary = {m: ModelBaseline(31_019, 31_019, accuracy[m], tok_s[m], latency_s[m]) for m in PAPER_MODELS}
    per_model = {
        m: RoutedModelStats(n=n, correct=round(acc * n / 100), lat=[latency])
        for m, (n, _pct, acc, latency) in PAPER_TABLE1.items()
    }
    table = RoutingTable(per_model=per_model, with_any=10_000, oracle_ok=8_676, ps_ok=4_966, any_lat=[23.4])
    rates = {"llama3.1_8B": [46.1], "qwen2.5_7B": [27.8], "llama3.2_1B": [19.6], "gemma2_9B": [2.5]}
    return {
        "n_queries": 31_019,
        "static_runs": 279_171,
        "summary": summary,
        "table": table,
        "sizes": size_class_totals(table),
        "rates": rates,
    }


def test_build_checks_compares_each_published_value_with_its_source() -> None:
    checks = build_checks(**published_inputs())

    assert len(checks) == 84
    claims = "\n".join(f"{c.claim}\t{c.paper}" for c in checks)
    assert hashlib.sha256(claims.encode("utf-8")).hexdigest() == CLAIMS_SHA256
    assert [c for c in checks if not c.matches] == []
    assert checks[14].claim == "Table I Llama-3.2-1B: queries"
    assert checks[46].claim == "Table I Gemma-2-27B: queries"  # the routed run's name for the 27B slot
    assert checks[-1].claim == "Fig. 2b latency Gemma-3-27B (s)"
    reproduced = {c.claim: c.reproduced for c in checks}
    assert reproduced["Throughput ratio small/large"] == "10x"
    assert reproduced["Table I Qwen2.5-1.5B: % of queries"] == "0.04"  # two decimals below 1%
    assert reproduced["Table I Llama-3.2-1B: % of queries"] == "17.1"
    assert reproduced["Table II scored routed queries, total"] == "28,384"
    assert format_report(checks).endswith("\n\n84/84 numbers match the paper.")


def test_format_report_layout() -> None:
    checks = [Check("short", "1,000", "1000"), Check("a longer claim", "10x", "9x")]
    assert format_report(checks).split("\n") == [
        "claim" + " " * 26 + "paper" + " " * 12 + "reproduced  match",
        "short" + " " * 26 + "1,000" + " " * 18 + "1000  yes",
        "a longer claim" + " " * 19 + "10x" + " " * 20 + "9x  NO",
        "",
        "1/2 numbers match the paper.",
    ]


def test_report_counts_the_matching_checks(tmp_path: Path) -> None:
    good, bad = Check("a", "1", "1"), Check("b", "1", "2")
    assert ReproductionReport((good, bad), tmp_path, figures_written=False).matched == 1
    assert not ReproductionReport((good, bad), tmp_path, figures_written=False).all_match
    assert ReproductionReport((good,), tmp_path, figures_written=False).all_match


# Output files.


def test_write_table_writes_crlf_csv_and_creates_the_directory(tmp_path: Path) -> None:
    path = tmp_path / "new" / "table.csv"
    write_table(path, ["a", "b"], [["x,y", 3], ("é", "-")])
    assert path.read_bytes() == 'a,b\r\n"x,y",3\r\né,-\r\n'.encode()


def test_write_verification_marks_each_check(tmp_path: Path) -> None:
    path = tmp_path / "verification.csv"
    write_verification(path, [Check("Queries", "31,019", "31,019"), Check("Ratio", "10x", "9x")])
    assert path.read_bytes() == b'claim,paper,reproduced,match\r\nQueries,"31,019","31,019",yes\r\nRatio,10x,9x,NO\r\n'


def test_write_tables_marks_models_without_scored_queries(tmp_path: Path) -> None:
    summary = {m: ModelBaseline(runs=2, counted=1, accuracy=100.0, tok_s=12.34, latency_s=1.25) for m in PAPER_MODELS}
    per_bench = {m: {"gsm8k": 100.0} for m in PAPER_MODELS}
    per_model = {m: RoutedModelStats() for m in PAPER_MODELS}
    per_model["gemma3_27B"] = RoutedModelStats(n=3, correct=1, lat=[1.0, 2.0, 4.0])
    table = RoutingTable(per_model=per_model, with_any=4, oracle_ok=2, ps_ok=1, any_lat=[1.0, 2.0, 4.0, 5.0])

    write_tables(
        tmp_path, summary=summary, per_bench=per_bench, table=table, n_queries=8, sizes=size_class_totals(table)
    )

    def lines(name: str) -> list[str]:
        return (tmp_path / name).read_bytes().decode("utf-8").split("\r\n")

    assert lines("static_baseline_summary.csv")[1] == "Llama-3.2-1B,1-3B,2,1,100.0,12.3,1.25"
    assert lines("static_baseline_per_benchmark_accuracy.csv")[1] == "Llama-3.2-1B,0.0,0.0,100.0,0.0,0.0,0.0,0.0,0.0"
    t1 = lines("table1_routing.csv")
    assert t1[1] == "Llama-3.2-1B,0,0.00,-,-"
    assert t1[9:] == [
        "Gemma-2-27B,3,37.50,33.3,2.33",
        "Unscored (no judge label),5,62.5,-,-",
        "Total,8,100,25.0,3.0",
        "",
    ]
    assert lines("table2_scored_routed_queries_by_size.csv") == [
        "model_size,scored_routed_queries",
        "1-3B,0",
        "7-9B,0",
        "14-27B,3",
        "Total,3",
        "",
    ]


def test_make_figures_skips_without_matplotlib(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setitem(sys.modules, "matplotlib", None)
    monkeypatch.setattr(logging.getLogger("pickspin"), "propagate", True)
    with caplog.at_level(logging.WARNING, logger="pickspin.paper.figures"):
        assert make_figures({}, [], {}, tmp_path / "figures") is False
    assert not (tmp_path / "figures").exists()
    assert "matplotlib not installed; skipping figures" in caplog.text


def write_synthetic_traces(traces_dir: Path) -> None:
    """500 queries; every model runs each one in 1 s with 10 tokens, except that every tenth run has no tokens.

    The judge marks even queries correct on every model, and the last ten queries have no labels. The
    routed run cycles through the nine slots in paper order (the 27B slot logged as gemma2_27B), takes
    2 s per query and is written in reverse order.
    """
    traces_dir.mkdir()
    n = 500
    bench = ["gsm8k" if i % 2 == 0 else "mbpp" for i in range(n)]
    models = (*PAPER_MODELS, "llama3_70B")
    slots = [{"gemma3_27B": "gemma2_27B"}.get(m, m) for m in PAPER_MODELS]
    write_csv_gz(
        traces_dir / "static_baseline.csv.gz",
        STATIC_HEADER,
        [
            [m, f"q{i}", bench[i], "1", "1000.0", "5", "0" if i % 10 == 9 else "10", ""]
            for m in models
            for i in range(n)
        ],
    )
    write_csv_gz(
        traces_dir / "judgments.csv.gz",
        JUDGMENTS_HEADER,
        [[m, f"q{i}", bench[i], "1" if i % 2 == 0 else "0"] for m in models for i in range(n - 10)],
    )
    write_csv_gz(
        traces_dir / "pick_spin_routed.csv.gz",
        ROUTED_HEADER,
        [
            [str(i), f"q{i}", bench[i], "SIMPLE", slots[i % 9], "1", "10", "2.0", "0", "0.0", "2.0"]
            for i in reversed(range(n))
        ],
    )


def test_reproduce_writes_every_output(tmp_path: Path) -> None:
    write_synthetic_traces(tmp_path / "traces")
    out = tmp_path / "out"

    report = reproduce(tmp_path / "traces", out)

    assert report.out_dir == out
    assert report.figures_written
    assert len(report.checks) == 84
    assert report.matched == 0
    assert not report.all_match

    def text(name: str) -> str:
        return (out / name).read_bytes().decode("utf-8")

    sizes = [SIZE_CLASS[m] for m in PAPER_MODELS]
    assert text("static_baseline_summary.csv") == "".join(
        ["model,size_class,runs,runs_with_output,accuracy_pct,tokens_per_s,latency_s\r\n"]
        + [f"{DISPLAY_NAME[m]},{size},500,450,54.4,10.0,1.00\r\n" for m, size in zip(PAPER_MODELS, sizes)]
    )
    assert text("static_baseline_per_benchmark_accuracy.csv") == "".join(
        ["model,humaneval,mbpp,gsm8k,math,truthfulqa,mmlu_pro,arc,hellaswag\r\n"]
        + [f"{DISPLAY_NAME[m]},0.0,0.0,98.0,0.0,0.0,0.0,0.0,0.0\r\n" for m in PAPER_MODELS]
    )
    assert text("table1_routing.csv") == (
        "model,queries,pct,accuracy_pct,latency_s\r\n"
        "Llama-3.2-1B,55,11.00,50.9,2.00\r\n"
        "Qwen2.5-1.5B,55,11.00,49.1,2.00\r\n"
        "Gemma-2-2B,55,11.00,50.9,2.00\r\n"
        "Llama-3.2-3B,55,11.00,49.1,2.00\r\n"
        "Qwen2.5-7B,54,10.80,50.0,2.00\r\n"
        "Llama-3.1-8B,54,10.80,50.0,2.00\r\n"
        "Gemma-2-9B,54,10.80,50.0,2.00\r\n"
        "Qwen2.5-14B,54,10.80,50.0,2.00\r\n"
        "Gemma-2-27B,54,10.80,50.0,2.00\r\n"
        "Unscored (no judge label),10,2.0,-,-\r\n"
        "Total,500,100,50.0,2.0\r\n"
    )
    assert text("table2_scored_routed_queries_by_size.csv") == (
        "model_size,scored_routed_queries\r\n1-3B,220\r\n7-9B,162\r\n14-27B,108\r\nTotal,490\r\n"
    )
    with (out / "verification.csv").open(newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    assert rows[0] == ["claim", "paper", "reproduced", "match"]
    assert rows[1:] == [[c.claim, c.paper, c.reproduced, "yes" if c.matches else "NO"] for c in report.checks]
    reproduced = {c.claim: c.reproduced for c in report.checks}
    assert reproduced["Static baseline runs (9 models x queries)"] == "4,500"
    assert reproduced["Fig. 3 rate at 31,000 queries, Llama-3.1-8B (%)"] == "11.0"  # 55 of 500, raw keys
    assert reproduced["Fig. 3 rate at 31,000 queries, Qwen2.5-7B (%)"] == "11.2"
    assert png_size(out / "figures" / "fig2a_throughput.png") == (1400, 900)
    assert png_size(out / "figures" / "fig2b_latency.png") == (1600, 900)
    assert png_size(out / "figures" / "fig3_thompson.png") == (1600, 900)
