"""Trace-driven simulation of Pick and Spin.

Pick and Spin run unchanged on a simulated clock. A query routed to model m takes the latency
and success recorded for that (query, model) pair in results/traces/static_baseline.csv.gz,
and its correctness is the judge label in results/traces/judgments.csv.gz. Spin tracks
COLD/LOADING/WARM per model, cold starts share the storage bandwidth (Eq. 5), queries routed to
a model that is not warm wait for its load (Eq. 6), and a model with nothing in flight for
T_cooldown is scaled to zero.

Load: by default --workers closed-loop clients (250, as in the live runner) send the 31,019
queries in a seeded random order; each sends its next query when the previous one returns.
With --arrival-rate R the queries instead arrive as a Poisson process of R queries per second.
A model serves any number of queries at once with their recorded latencies unless
--max-concurrency caps it.

Policies:
  pick-and-spin           scale to zero; Pick scores Spin's latency, which adds the cold-start
                          penalty of a model that is not warm (default design)
  pick-and-spin-observed  scale to zero; Pick scores mean observed latency, cold-start waits included
  unaware                 scale to zero; Pick scores inference latency only
  static                  every model warm for the whole run, never scaled down

    python src/pickspin/simulate.py                  # all policies, seeds 0-4
    python src/pickspin/simulate.py --policies pick-and-spin static --seeds 0
    python src/pickspin/simulate.py --arrival-rate 0.25 1 4    # open-loop Poisson arrivals

Writes summary.csv, per_model.csv and cold_starts_by_tier.csv to results/simulation/<load>/,
where <load> is closed-250 or poisson-<R>qps, and updates results/simulation/overview.csv (mean
over seeds for every load simulated so far). --write-queries adds a per-query trace for the
first seed.
"""

import argparse
import csv
import gzip
import heapq
import json
import random
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import MODELS, QUERIES_FILE, ROOT, SPIN, TIER_ORDER, TRACES_DIR  # noqa: E402
from pick import Pick  # noqa: E402
from spin import COLD, WARM, SharedStorage, Spin, init_seconds  # noqa: E402

OUT = ROOT / "results" / "simulation"
TIERS_CACHE = ROOT / "data" / "query_tiers.csv.gz"
POLICIES = {
    "pick-and-spin":          {"scale_to_zero": True,  "signal": "spin"},
    "pick-and-spin-observed": {"scale_to_zero": True,  "signal": "observed"},
    "unaware":                {"scale_to_zero": True,  "signal": "inference"},
    "static":                 {"scale_to_zero": False, "signal": "spin"},
}


def read_csv_gz(path):
    with gzip.open(path, "rt", encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f)


def load_data():
    queries = [json.loads(line) for line in gzip.open(QUERIES_FILE, "rt", encoding="utf-8")]
    runs = {}
    for r in read_csv_gz(TRACES_DIR / "static_baseline.csv.gz"):
        if r["model"] in MODELS:
            ms = r["latency_ms"]
            runs[(r["id"], r["model"])] = (r["success"] in ("True", "true", "1"), float(ms) / 1000 if ms else 0.0)
    correct = {(r["id"], r["model"]): r["is_correct"] in ("True", "true", "1")
               for r in read_csv_gz(TRACES_DIR / "judgments.csv.gz") if r["model"] in MODELS}
    return queries, runs, correct


def query_tiers(queries, reclassify=False):
    """Hybrid-classifier tier for every query, cached in data/query_tiers.csv.gz."""
    if TIERS_CACHE.exists() and not reclassify:
        return {r["id"]: (r["tier"], r["stage"]) for r in read_csv_gz(TIERS_CACHE)}
    from classifier import HybridClassifier
    print(f"Classifying {len(queries):,} queries (keyword lists, then DistilBERT) ...", flush=True)
    labels = HybridClassifier().classify_many([q["query"] for q in queries])
    with gzip.open(TIERS_CACHE, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "tier", "stage"])
        for q, (t, s) in zip(queries, labels):
            w.writerow([q["id"], t, s])
    return {q["id"]: lab for q, lab in zip(queries, labels)}


def simulate(policy, seed, queries, runs, correct, tiers, workers=250, max_concurrency=None,
             cooldown_s=SPIN["cooldown_s"], arrival_rate=None):
    cfg = POLICIES[policy]
    order = list(range(len(queries)))
    random.Random(seed).shuffle(order)
    storage = SharedStorage()
    spin = Spin(cooldown_s=cooldown_s, scale_to_zero=cfg["scale_to_zero"], now=0.0,
                load_estimate=storage.estimate)
    pick = Pick(classifier=None, spin=spin, latency_signal=cfg["signal"], rng=random.Random(10_000 + seed))

    events, seq = [], [0]

    def push(t, kind, payload):
        seq[0] += 1
        heapq.heappush(events, (t, seq[0], kind, payload))

    storage_version = [0]

    def reschedule_storage():
        storage_version[0] += 1
        nxt = storage.next_transfer_done()
        if nxt:
            push(nxt[0], "transfer", (storage_version[0], nxt[1]))

    waiting = defaultdict(list)
    records = [None] * len(queries)
    nxt_query = [0]
    done_count = [0]

    def can_start(m):
        return spin.status(m) == WARM and (max_concurrency is None or spin.s[m].inflight < max_concurrency)

    def start(rec, t):
        m = rec["model"]
        spin.start(m, t)
        rec["start"] = t
        push(t + rec["infer_s"], "done", rec)

    def dispatch(k, t, worker):
        qi = order[k]
        q = queries[qi]
        tier, stage = tiers[q["id"]]
        m = pick.route(None, t, tier=tier, stage=stage)["model"]
        ok, infer_s = runs[(q["id"], m)]
        before = spin.request(m, t)
        rec = {"order": k + 1, "id": q["id"], "benchmark": q["benchmark"], "tier": tier, "stage": stage,
               "model": m, "arrive": t, "infer_s": infer_s, "success": ok,
               "correct": correct.get((q["id"], m)), "cold_start": before == COLD, "worker": worker}
        records[qi] = rec
        if before == COLD:
            storage.begin(m, t)
            reschedule_storage()
        if can_start(m):
            start(rec, t)
        else:
            waiting[m].append(rec)

    if arrival_rate:
        arrivals, t = random.Random(20_000 + seed), 0.0
        for k in range(len(order)):
            t += arrivals.expovariate(arrival_rate)
            push(t, "arrive", k)
    else:
        for w in range(min(workers, len(order))):
            push(0.0, "next", w)

    summary = None
    while events:
        t, _, kind, payload = heapq.heappop(events)
        if kind == "next":
            if nxt_query[0] < len(order):
                nxt_query[0] += 1
                dispatch(nxt_query[0] - 1, t, payload)
        elif kind == "arrive":
            dispatch(payload, t, None)
        elif kind == "transfer":
            version, m = payload
            if version != storage_version[0]:
                continue
            storage.transfer_done(m, t)
            reschedule_storage()
            push(t + init_seconds(m, storage.gbps), "ready", m)
        elif kind == "ready":
            m = payload
            spin.loaded(m, t)
            while waiting[m] and can_start(m):
                start(waiting[m].pop(0), t)
        elif kind == "done":
            rec = payload
            m = rec["model"]
            rec["end"] = t
            rec["wait_s"] = rec["start"] - rec["arrive"]
            rec["total_s"] = t - rec["arrive"]
            spin.finish(m, t, rec["infer_s"], rec["total_s"])
            pick.update(m, rec["tier"], rec["success"])
            done_count[0] += 1
            while waiting[m] and can_start(m):
                start(waiting[m].pop(0), t)
            if spin.s[m].inflight == 0 and cfg["scale_to_zero"]:
                push(t + cooldown_s, "idle", m)
            if done_count[0] == len(queries):
                summary = spin.summary(t)
                summary["makespan_s"] = t
                break
            if rec["worker"] is not None:
                push(t, "next", rec["worker"])
        elif kind == "idle":
            m = payload
            if m in spin.idle_expired(t):
                spin.stop(m, t)
    return records, summary


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))]


def describe(load, policy, seed, records, s):
    n = len(records)
    scored = [r for r in records if r["correct"] is not None]
    total = [r["total_s"] for r in records]
    row = {
        "load": load, "policy": policy, "seed": seed, "queries": n,
        "makespan_h": s["makespan_s"] / 3600,
        "gpu_hours": s["gpu_hours"], "busy_gpu_hours": s["busy_gpu_hours"],
        "gpu_utilization_pct": 100 * s["gpu_utilization"],
        "cold_starts": s["cold_starts"], "cold_start_rate_pct": 100 * s["cold_start_rate"],
        "queries_waiting_for_load_pct": 100 * sum(r["wait_s"] > 0 for r in records) / n,
        "latency_mean_s": statistics.fmean(total), "latency_median_s": statistics.median(total),
        "latency_p95_s": pct(total, 0.95),
        "execution_success_pct": 100 * sum(r["success"] for r in records) / n,
        "accuracy_pct": 100 * sum(r["correct"] for r in scored) / len(scored),
        "scored_queries": len(scored),
    }
    for t in TIER_ORDER:
        row[f"share_{t.lower()}_pct"] = 100 * sum(r["tier"] == t for r in records) / n
    return row


def per_model_rows(load, policy, seed, records, s):
    rows = []
    for m in MODELS:
        rs = [r for r in records if r["model"] == m]
        scored = [r for r in rs if r["correct"] is not None]
        pm = s["per_model"][m]
        rows.append({
            "load": load, "policy": policy, "seed": seed, "model": MODELS[m]["label"], "tier": MODELS[m]["tier"],
            "queries": len(rs), "share_pct": 100 * len(rs) / len(records),
            "accuracy_pct": 100 * sum(r["correct"] for r in scored) / len(scored) if scored else "",
            "latency_mean_s": statistics.fmean(r["total_s"] for r in rs) if rs else "",
            "cold_starts": pm["cold_starts"], "gpu_hours": pm["gpu_hours"],
            "busy_gpu_hours": pm["busy_gpu_hours"],
        })
    return rows


def cold_by_tier_rows(load, policy, seed, records):
    rows = []
    for t in TIER_ORDER:
        rs = [r for r in records if MODELS[r["model"]]["tier"] == t]
        cs = sum(r["cold_start"] for r in rs)
        rows.append({"load": load, "policy": policy, "seed": seed, "tier": t, "routed_queries": len(rs),
                     "cold_starts": cs,
                     "conditional_rate_pct": 100 * cs / len(rs) if rs else 0.0})
    return rows


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--policies", nargs="+", default=list(POLICIES), choices=list(POLICIES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--workers", type=int, default=250, help="closed-loop clients (ignored with --arrival-rate)")
    ap.add_argument("--arrival-rate", type=float, nargs="+",
                    help="open-loop Poisson arrivals, queries per second (one run per rate)")
    ap.add_argument("--max-concurrency", type=int, help="cap on queries in flight per model")
    ap.add_argument("--cooldown", type=float, default=SPIN["cooldown_s"])
    ap.add_argument("--reclassify", action="store_true", help="rerun the hybrid classifier (needs DistilBERT)")
    ap.add_argument("--write-queries", action="store_true", help="also write a per-query trace for the first seed")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    queries, runs, correct = load_data()
    tiers = query_tiers(queries, args.reclassify)
    stages = Counter(s for _, s in tiers.values())
    print(f"{len(queries):,} queries; classifier stages: " +
          ", ".join(f"{k} {v:,} ({100 * v / len(queries):.1f}%)" for k, v in stages.most_common()))

    for rate in args.arrival_rate or [None]:
        run_load(args, rate, queries, runs, correct, tiers)
    write_overview(args.out)


def run_load(args, rate, queries, runs, correct, tiers):
    load = f"poisson-{rate:g}qps" if rate else f"closed-{args.workers}"
    out = args.out / load
    summary, per_model, cold_tier = [], [], []
    for policy in args.policies:
        for seed in args.seeds:
            records, s = simulate(policy, seed, queries, runs, correct, tiers, args.workers,
                                  args.max_concurrency, args.cooldown, rate)
            row = describe(load, policy, seed, records, s)
            summary.append(row)
            per_model += per_model_rows(load, policy, seed, records, s)
            cold_tier += cold_by_tier_rows(load, policy, seed, records)
            print(f"{load} {policy:24s} seed {seed}: {row['gpu_hours']:.2f} GPU-h, util {row['gpu_utilization_pct']:.1f}%, "
                  f"cold starts {row['cold_starts']} ({row['cold_start_rate_pct']:.2f}%), "
                  f"latency {row['latency_mean_s']:.2f}s, accuracy {row['accuracy_pct']:.1f}%", flush=True)
            if args.write_queries and seed == args.seeds[0]:
                keep = ["order", "id", "benchmark", "tier", "stage", "model", "arrive", "start", "end",
                        "infer_s", "wait_s", "total_s", "cold_start", "success", "correct"]
                out.mkdir(parents=True, exist_ok=True)
                with gzip.open(out / f"queries_{policy}.csv.gz", "wt", encoding="utf-8", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(keep)
                    for r in sorted(records, key=lambda r: r["order"]):
                        w.writerow([round(r[k], 3) if isinstance(r[k], float) else r[k] for k in keep])

    write_csv(out / "summary.csv", summary)
    write_csv(out / "per_model.csv", per_model)
    write_csv(out / "cold_starts_by_tier.csv", cold_tier)
    print(f"Wrote {out}")


def write_overview(root):
    """Mean (and s.d.) over seeds of every summary.csv under root, with GPU-hours relative to static."""
    rows = []
    for path in sorted(root.glob("*/summary.csv")):
        with open(path, encoding="utf-8", newline="") as f:
            rows += list(csv.DictReader(f))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["load"], r["policy"])].append(r)
    static = {load: statistics.fmean(float(r["gpu_hours"]) for r in rs)
              for (load, policy), rs in groups.items() if policy == "static"}
    cols = ["makespan_h", "gpu_hours", "gpu_utilization_pct", "cold_starts", "cold_start_rate_pct",
            "queries_waiting_for_load_pct", "latency_mean_s", "latency_p95_s", "accuracy_pct"]
    out = []
    for (load, policy), rs in sorted(groups.items()):
        row = {"load": load, "policy": policy, "seeds": len(rs)}
        for c in cols:
            vals = [float(r[c]) for r in rs]
            row[c] = statistics.fmean(vals)
            if c == "gpu_hours":
                row["gpu_hours_sd"] = statistics.stdev(vals) if len(vals) > 1 else 0.0
                row["gpu_hours_vs_static_pct"] = (100 * (row[c] / static[load] - 1)) if load in static else ""
        out.append(row)
    write_csv(root / "overview.csv", out)


if __name__ == "__main__":
    main()
