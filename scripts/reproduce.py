"""Regenerate the paper's tables and figures from the released traces.

Reads only files under results/traces/ and writes CSV tables, figures and a
paper-vs-reproduced comparison to results/. No GPU or network access needed.

    python scripts/reproduce.py
"""

import csv
import gzip
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TRACES = ROOT / "results" / "traces"
OUT = ROOT / "results"
FIGS = OUT / "figures"

# The nine model slots, smallest first. The static baseline ran Gemma-3-27B in the
# 27B slot ("gemma3_27B"); the routed run served google/gemma-2-27b-it there and
# logged it as "gemma2_27B". A routed query is scored with the static-baseline
# judge label of the model slot it was routed to.
NINE = ["llama3.2_1B", "qwen2.5_1.5B", "gemma2_2B", "llama3.2_3B",
        "qwen2.5_7B", "llama3.1_8B", "gemma2_9B", "qwen2.5_14B", "gemma3_27B"]
PRETTY = {"llama3.2_1B": "Llama-3.2-1B", "qwen2.5_1.5B": "Qwen2.5-1.5B",
          "gemma2_2B": "Gemma-2-2B", "llama3.2_3B": "Llama-3.2-3B",
          "qwen2.5_7B": "Qwen2.5-7B", "llama3.1_8B": "Llama-3.1-8B",
          "gemma2_9B": "Gemma-2-9B", "qwen2.5_14B": "Qwen2.5-14B",
          "gemma3_27B": "Gemma-3-27B"}
# Names for the routed run (Table I), where the 27B slot was served by Gemma-2-27B.
ROUTED_NAME = dict(PRETTY, gemma3_27B="Gemma-2-27B")
SIZE_CLASS = {"llama3.2_1B": "1-3B", "qwen2.5_1.5B": "1-3B", "gemma2_2B": "1-3B", "llama3.2_3B": "1-3B",
              "qwen2.5_7B": "7-9B", "llama3.1_8B": "7-9B", "gemma2_9B": "7-9B",
              "qwen2.5_14B": "14-27B", "gemma3_27B": "14-27B"}
BENCHMARKS = ["humaneval", "mbpp", "gsm8k", "math", "truthfulqa", "mmlu_pro", "arc", "hellaswag"]


def alias(model):
    return "gemma3_27B" if model == "gemma2_27B" else model


def read_csv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def write_csv(path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load():
    static = defaultdict(list)
    for r in read_csv_gz(TRACES / "static_baseline.csv.gz"):
        static[r["model"]].append(r)
    judged = {}
    for r in read_csv_gz(TRACES / "judgments.csv.gz"):
        judged[(r["id"], r["model"])] = r["is_correct"] == "1"
    routed = sorted(read_csv_gz(TRACES / "pick_spin_routed.csv.gz"), key=lambda r: int(r["order"]))
    return static, judged, routed


def static_summary(static, judged):
    """Per-model static baseline metrics (Fig. 2 and the numbers quoted in Sec. VII-A).

    Same rules as the original analysis: a run counts if it returned at least one
    token with a positive latency; accuracy is judged-correct / counted runs.
    """
    rows, per_bench = {}, {}
    for m in NINE:
        tps, lat, correct, total = [], [], 0, 0
        bench = defaultdict(lambda: [0, 0])
        for r in static[m]:
            ms, toks = float(r["latency_ms"]), int(r["completion_tokens"])
            if ms <= 0 or toks <= 0:
                continue
            tps.append(toks / (ms / 1000.0))
            lat.append(ms)
            total += 1
            ok = judged.get((r["id"], m)) is True
            correct += ok
            bench[r["benchmark"]][0] += ok
            bench[r["benchmark"]][1] += 1
        rows[m] = {"runs": len(static[m]), "counted": total, "accuracy": 100 * correct / total,
                   "tok_s": statistics.mean(tps), "latency_s": statistics.mean(lat) / 1000}
        per_bench[m] = {b: 100 * c / n for b, (c, n) in bench.items()}
    return rows, per_bench


def routing_table(routed, judged):
    """Table I and the oracle analysis in Sec. VII-B."""
    per_model = defaultdict(lambda: {"n": 0, "correct": 0, "lat": []})
    with_any, oracle_ok, ps_ok = 0, 0, 0
    any_lat = []  # latency of every query that has at least one label (Total row of Table I)
    for r in routed:
        qid, m = r["id"], alias(r["model"])
        labels = {x: judged[(qid, x)] for x in NINE if (qid, x) in judged}
        mine = judged.get((qid, m))
        if mine is not None:
            d = per_model[m]
            d["n"] += 1
            d["correct"] += mine
            d["lat"].append(float(r["total_latency"]))
        if labels:
            with_any += 1
            any_lat.append(float(r["total_latency"]))
            if any(labels.values()):
                oracle_ok += 1
                ps_ok += bool(mine)
    return per_model, with_any, oracle_ok, ps_ok, any_lat


def convergence(routed, track=("llama3.1_8B", "qwen2.5_7B", "llama3.2_1B", "gemma2_9B"), step=500):
    xs, ys, counts = [], {m: [] for m in track}, Counter()
    for i, r in enumerate(routed, 1):
        counts[r["model"]] += 1
        if i % step == 0:
            xs.append(i / 1000)
            for m in track:
                ys[m].append(100 * counts[m] / i)
    return xs, ys


def make_figures(summary, xs, ys):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping figures")
        return
    FIGS.mkdir(parents=True, exist_ok=True)
    colors = {"1-3B": "#4C72B0", "7-9B": "#DD8452", "14-27B": "#55A868"}
    names = [PRETTY[m] for m in NINE]
    cols = [colors[SIZE_CLASS[m]] for m in NINE]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [summary[m]["tok_s"] for m in NINE]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Throughput (tokens/second)")
    ax.set_title("Fig. 2a: throughput by model (static baseline)")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2a_throughput.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [summary[m]["latency_s"] for m in NINE]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("Average latency (s)")
    ax.set_title("Fig. 2b: inference latency by model (static baseline)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(FIGS / "fig2b_latency.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, series in ys.items():
        ax.plot(xs, series, label=f"{PRETTY[m]} ({series[-1]:.1f}%)")
    ax.set_xlabel("Queries processed (thousands)")
    ax.set_ylabel("Cumulative selection rate (%)")
    ax.set_title("Fig. 3: Thompson Sampling selection over the routed run")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig3_thompson.png", dpi=200)
    plt.close(fig)


def main():
    static, judged, routed = load()

    # Run counts (abstract, Sec. VII)
    static_runs = sum(len(static[m]) for m in NINE)
    n_queries = len(routed)

    summary, per_bench = static_summary(static, judged)
    write_csv(OUT / "static_baseline_summary.csv",
              ["model", "size_class", "runs", "runs_with_output", "accuracy_pct", "tokens_per_s", "latency_s"],
              [[PRETTY[m], SIZE_CLASS[m], s["runs"], s["counted"], f"{s['accuracy']:.1f}",
                f"{s['tok_s']:.1f}", f"{s['latency_s']:.2f}"] for m, s in summary.items()])
    write_csv(OUT / "static_baseline_per_benchmark_accuracy.csv", ["model"] + BENCHMARKS,
              [[PRETTY[m]] + [f"{per_bench[m].get(b, 0):.1f}" for b in BENCHMARKS] for m in NINE])

    per_model, with_any, oracle_ok, ps_ok, any_lat = routing_table(routed, judged)
    judged_total = sum(d["n"] for d in per_model.values())
    unjudged = n_queries - judged_total
    t1 = []
    for m in NINE:
        d = per_model[m]
        t1.append([ROUTED_NAME[m], d["n"], f"{100 * d['n'] / n_queries:.2f}",
                   f"{100 * d['correct'] / d['n']:.1f}" if d["n"] else "-",
                   f"{statistics.mean(d['lat']):.2f}" if d["lat"] else "-"])
    t1.append(["Unscored (no judge label)", unjudged, f"{100 * unjudged / n_queries:.1f}", "-", "-"])
    t1.append(["Total", n_queries, "100", f"{100 * ps_ok / with_any:.1f}", f"{statistics.mean(any_lat):.1f}"])
    write_csv(OUT / "table1_routing.csv", ["model", "queries", "pct", "accuracy_pct", "latency_s"], t1)

    tiers = defaultdict(int)
    for m in NINE:
        tiers[SIZE_CLASS[m]] += per_model[m]["n"]
    write_csv(OUT / "table2_scored_routed_queries_by_size.csv", ["model_size", "scored_routed_queries"],
              [[t, n] for t, n in tiers.items()] + [["Total", judged_total]])

    xs, ys = convergence(routed)

    make_figures(summary, xs, ys)

    s = summary
    q7, g27 = s["qwen2.5_7B"], s["gemma3_27B"]
    tps = [s[m]["tok_s"] for m in NINE]
    lat = [s[m]["latency_s"] for m in NINE]
    complex_share = 100 * (per_model["qwen2.5_14B"]["n"] + per_model["gemma3_27B"]["n"]) / n_queries

    def pm(m):
        return per_model[m]

    # (claim, paper value, reproduced value). Values are rounded the way the paper prints them.
    checks = [
        ("Queries across 8 benchmarks", "31,019", f"{n_queries:,}"),
        ("Static baseline runs (9 models x queries)", "279,171", f"{static_runs:,}"),
        ("Total inference runs", "310,190", f"{static_runs + n_queries:,}"),
        ("Throughput, Llama-3.2-1B (tok/s)", "45.8", f"{tps[0]:.1f}"),
        ("Throughput, Gemma-3-27B (tok/s)", "4.6", f"{tps[-1]:.1f}"),
        ("Throughput ratio small/large", "10x", f"{tps[0] / tps[-1]:.0f}x"),
        ("Latency, Llama-3.2-1B (s)", "1.51", f"{lat[0]:.2f}"),
        ("Latency, Gemma-3-27B (s)", "21.66", f"{lat[-1]:.2f}"),
        ("Latency ratio large/small", "14x", f"{lat[-1] / lat[0]:.0f}x"),
        ("Accuracy, Qwen2.5-14B (%)", "62.6", f"{s['qwen2.5_14B']['accuracy']:.1f}"),
        ("Accuracy, Gemma-3-27B (%)", "56.6", f"{g27['accuracy']:.1f}"),
        ("Accuracy, Qwen2.5-7B (%)", "56.3", f"{q7['accuracy']:.1f}"),
        ("Latency, Qwen2.5-7B (s)", "4.83", f"{q7['latency_s']:.2f}"),
        ("Latency ratio Gemma-3-27B / Qwen2.5-7B", "4.5x", f"{g27['latency_s'] / q7['latency_s']:.1f}x"),
    ]
    paper_t1 = {"llama3.2_1B": (5319, 17.1, 23.2, 1.82), "qwen2.5_1.5B": (11, 0.04, 27.3, 4.22),
                "gemma2_2B": (7, 0.02, 28.6, 3.34), "llama3.2_3B": (5, 0.02, 20.0, 8.29),
                "qwen2.5_7B": (8101, 26.1, 63.3, 28.53), "llama3.1_8B": (13462, 43.4, 54.5, 28.87),
                "gemma2_9B": (499, 1.6, 73.5, 30.52), "qwen2.5_14B": (347, 1.1, 72.0, 44.63),
                "gemma3_27B": (633, 2.0, 75.2, 13.28)}
    for m, (n, pct, acc, l) in paper_t1.items():
        d = pm(m)
        pct_fmt = f"{100 * d['n'] / n_queries:.2f}" if pct < 1 else f"{100 * d['n'] / n_queries:.1f}"
        checks += [
            (f"Table I {ROUTED_NAME[m]}: queries", f"{n:,}", f"{d['n']:,}"),
            (f"Table I {ROUTED_NAME[m]}: % of queries", f"{pct}", pct_fmt),
            (f"Table I {ROUTED_NAME[m]}: accuracy (%)", f"{acc}", f"{100 * d['correct'] / d['n']:.1f}"),
            (f"Table I {ROUTED_NAME[m]}: latency (s)", f"{l}", f"{statistics.mean(d['lat']):.2f}"),
        ]
    checks += [
        ("Table I unscored queries (no judge label)", "2,635", f"{unjudged:,}"),
        ("Table I unscored share (%)", "8.5", f"{100 * unjudged / n_queries:.1f}"),
        ("Table I overall accuracy (%)", "49.7", f"{100 * ps_ok / with_any:.1f}"),
        ("Table I overall latency (s)", "23.4", f"{statistics.mean(any_lat):.1f}"),
        ("Oracle accuracy (%)", "86.8", f"{100 * oracle_ok / with_any:.1f}"),
        ("Queries no model solves (%)", "13.2", f"{100 * (with_any - oracle_ok) / with_any:.1f}"),
        ("Share of oracle ceiling captured (%)", "57.2", f"{100 * ps_ok / oracle_ok:.1f}"),
        ("Share of queries routed to the 14B and 27B models (%)", "3.2", f"{complex_share:.1f}"),
        ("Table II scored routed queries, 1-3B models", "5,342", f"{tiers['1-3B']:,}"),
        ("Table II scored routed queries, 7-9B models", "22,062", f"{tiers['7-9B']:,}"),
        ("Table II scored routed queries, 14-27B models", "980", f"{tiers['14-27B']:,}"),
        ("Table II scored routed queries, total", "28,384", f"{judged_total:,}"),
        ("Fig. 3 rate at 31,000 queries, Llama-3.1-8B (%)", "46.1", f"{ys['llama3.1_8B'][-1]:.1f}"),
        ("Fig. 3 rate at 31,000 queries, Qwen2.5-7B (%)", "27.8", f"{ys['qwen2.5_7B'][-1]:.1f}"),
        ("Fig. 3 rate at 31,000 queries, Llama-3.2-1B (%)", "19.6", f"{ys['llama3.2_1B'][-1]:.1f}"),
        ("Fig. 3 rate at 31,000 queries, Gemma-2-9B (%)", "2.5", f"{ys['gemma2_9B'][-1]:.1f}"),
    ]
    for m, paper in zip(NINE, ["45.8", "26.4", "24.9", "16.3", "7.5", "10.6", "11.5", "5.6", "4.6"]):
        checks.append((f"Fig. 2a throughput {PRETTY[m]} (tok/s)", paper, f"{s[m]['tok_s']:.1f}"))
    for m, paper in zip(NINE, ["1.5", "1.5", "2.1", "2.2", "4.8", "4.3", "6.5", "7.6", "21.7"]):
        checks.append((f"Fig. 2b latency {PRETTY[m]} (s)", paper, f"{s[m]['latency_s']:.1f}"))

    def norm(v):
        return v.replace(",", "").replace("x", "")

    rows = [(c, p, r, "yes" if norm(p) == norm(r) else "NO") for c, p, r in checks]
    write_csv(OUT / "verification.csv", ["claim", "paper", "reproduced", "match"], rows)

    width = max(len(r[0]) for r in rows)
    print(f"{'claim':<{width}}  {'paper':>20}  {'reproduced':>20}  match")
    for c, p, r, ok in rows:
        print(f"{c:<{width}}  {p:>20}  {r:>20}  {ok}")
    n_ok = sum(r[3] == "yes" for r in rows)
    print(f"\n{n_ok}/{len(rows)} numbers match the paper.")
    print(f"Wrote tables and figures to {OUT.relative_to(ROOT)}/")
    return 0 if n_ok == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
