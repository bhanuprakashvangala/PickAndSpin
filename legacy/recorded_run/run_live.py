"""Route every benchmark query through Pick and Spin against live vLLM endpoints.

    python src/pickspin/run_live.py --endpoints deploy/endpoints.json --workers 250

Writes results/live/pick_spin_<timestamp>.jsonl with the same fields as
results/traces/pick_spin_routed.csv.gz (plus query, ground truth and response).
"""

import argparse
import gzip
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from config import MODELS, QUERIES_FILE, ROOT, load_endpoints
from pick import PickRouter
from spin import ModelExecutor


def load_queries(limit=None):
    with gzip.open(QUERIES_FILE, "rt", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    return queries[:limit] if limit else queries


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", help="JSON file mapping model keys to vLLM base URLs")
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--limit", type=int, help="only route the first N queries (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    router = PickRouter()
    executor = ModelExecutor(load_endpoints(args.endpoints))
    lock = threading.Lock()
    done = {"n": 0, "ok": 0}

    def process(item):
        routing = router.route(item["query"])
        result = executor.execute(routing["model"], item["query"])
        router.update(routing["model"], routing["tier"], result["success"], result["latency"])
        with lock:
            done["n"] += 1
            done["ok"] += result["success"]
            if done["n"] % 2000 == 0:
                print(f"[{done['n']:,}] ok={done['ok']:,}")
        return {"id": item["id"], "benchmark": item["benchmark"], "query": item["query"][:300],
                "ground_truth": item["ground_truth"], "tier": routing["tier"], "model": routing["model"],
                "response": result["response"][:500] if result["success"] else "",
                "success": result["success"], "tokens": result["tokens"],
                "latency": round(result["latency"], 3), "was_cold": result["was_cold"],
                "cold_penalty": result["cold_penalty"], "total_latency": round(result["total_latency"], 3)}

    queries = load_queries(args.limit)
    random.shuffle(queries)
    out_dir = ROOT / "results" / "live"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"pick_spin_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
    print(f"Routing {len(queries):,} queries over {len(MODELS)} models with {args.workers} workers -> {out}")
    start = time.time()
    with open(out, "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process, q) for q in queries]):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
    print(f"Done in {time.time() - start:.0f}s; cold starts: {executor.get_cold_stats()}")
    for m, s in sorted(router.get_stats().items(), key=lambda x: -x[1]["count"]):
        print(f"  {m:14s} {s['count']:6,d} queries  {s['avg_latency']:.2f}s avg")


if __name__ == "__main__":
    main()
