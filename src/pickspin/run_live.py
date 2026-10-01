"""Route the benchmark queries through Pick and Spin on a live Kubernetes deployment.

    python src/pickspin/run_live.py --endpoints deploy/endpoints.example.json --workers 250
    python src/pickspin/run_live.py --static ...    # every model kept running, no scaling

Pick classifies each query (keyword lists, then DistilBERT) and selects a model with the
latency Spin reports. Spin scales a cold model's Deployment to one replica, holds the query
until vLLM answers /health, forwards it, and scales models with nothing in flight for
T_cooldown back to zero. Writes results/live/pick_spin_<time>.jsonl (one line per query) and
pick_spin_<time>_summary.json (GPU-hours, utilization and cold starts from Spin's accounting).
"""

import argparse
import gzip
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

from classifier import HybridClassifier  # noqa: E402
from config import MODELS, QUERIES_FILE, ROOT, SPIN, load_endpoints  # noqa: E402
from pick import Pick  # noqa: E402
from spin import COLD, WARM, KubernetesActuator, Spin  # noqa: E402


def call_vllm(ep, query, max_tokens, headers, timeout=120):
    """Return (success, text, completion_tokens, seconds)."""
    payload = {"model": ep["model"], "messages": [{"role": "user", "content": query}],
               "max_tokens": max_tokens, "temperature": 0.1}
    t0 = time.monotonic()
    try:
        r = requests.post(ep["base_url"].rstrip("/") + "/v1/chat/completions", json=payload,
                          headers=headers, timeout=timeout)
        took = time.monotonic() - t0
        if r.status_code == 200:
            data = r.json()
            return True, data["choices"][0]["message"]["content"], data.get("usage", {}).get("completion_tokens", 0), took
        return False, f"HTTP {r.status_code}", 0, took
    except requests.RequestException as e:
        return False, type(e).__name__, 0, time.monotonic() - t0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints", help="JSON file mapping model keys to base_url, served model and Deployment")
    ap.add_argument("--namespace", default=os.environ.get("PS_NAMESPACE", "pick-and-spin"))
    ap.add_argument("--workers", type=int, default=250)
    ap.add_argument("--limit", type=int, help="only route the first N queries (smoke test)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--static", action="store_true", help="keep every model running; no cold starts or scaling")
    ap.add_argument("--latency-signal", default="spin", choices=["spin", "observed", "inference"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "live")
    args = ap.parse_args(argv)

    endpoints = load_endpoints(args.endpoints)
    key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    actuator = None if args.static else KubernetesActuator(endpoints, args.namespace)
    clock = time.monotonic
    spin = Spin(cooldown_s=SPIN["cooldown_s"], scale_to_zero=not args.static, now=clock(),
                load_estimate=actuator.load_estimate if actuator else None)
    pick = Pick(HybridClassifier(), spin, args.latency_signal, rng=random.Random(args.seed))
    ready = {m: threading.Event() for m in MODELS}
    scale_lock = {m: threading.Lock() for m in MODELS}
    if args.static:
        for e in ready.values():
            e.set()
    else:
        print("Scaling every model to zero so the run starts COLD ...")
        for m in MODELS:
            actuator.scale(m, 0)
        while any(actuator.ready_replicas(m) for m in MODELS):
            time.sleep(2)

    def bring_up(m):
        t0 = clock()
        try:
            with scale_lock[m]:
                actuator.scale(m, 1)
            actuator.wait_ready(m, t0)
        except Exception as e:  # the waiting queries are sent anyway and recorded as failures
            print(f"Loading {m} failed: {e}", flush=True)
        finally:
            spin.loaded(m, clock())
            ready[m].set()

    stop = threading.Event()

    def reaper():
        while not stop.wait(5):
            for m in spin.idle_expired(clock()):
                with scale_lock[m]:
                    if spin.stop(m, clock()):
                        ready[m].clear()
                        actuator.scale(m, 0)

    if not args.static:
        threading.Thread(target=reaper, daemon=True).start()

    def process(item):
        t_arrive = clock()
        route = pick.route(item["query"], t_arrive)
        m = route["model"]
        before = spin.request(m, t_arrive)
        if before == COLD:
            threading.Thread(target=bring_up, args=(m,), daemon=True).start()
        if before != WARM:
            ready[m].wait()
        t_start = clock()
        spin.start(m, t_start)
        ok, text, tokens, infer_s = call_vllm(endpoints[m], item["query"], args.max_tokens, headers)
        t_end = clock()
        spin.finish(m, t_end, infer_s, t_end - t_arrive)
        pick.update(m, route["tier"], ok)
        return {"id": item["id"], "benchmark": item["benchmark"], "tier": route["tier"], "stage": route["stage"],
                "model": m, "cold_start": before == COLD, "waited_for_load": before != WARM,
                "wait_s": round(t_start - t_arrive, 3), "latency": round(infer_s, 3),
                "total_latency": round(t_end - t_arrive, 3), "success": ok, "tokens": tokens,
                "response": text[:500] if ok else "", "error": "" if ok else text}

    with gzip.open(QUERIES_FILE, "rt", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f]
    random.Random(args.seed).shuffle(queries)
    queries = queries[:args.limit] if args.limit else queries

    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.out / f"pick_spin_{datetime.now():%Y%m%d_%H%M%S}"
    print(f"Routing {len(queries):,} queries with {args.workers} workers "
          f"({'static' if args.static else 'scale to zero'}) -> {stem}.jsonl")
    done = 0
    with open(f"{stem}.jsonl", "w", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(process, q) for q in queries]):
            f.write(json.dumps(fut.result(), ensure_ascii=False) + "\n")
            done += 1
            if done % 2000 == 0:
                print(f"[{done:,}] {spin.summary(clock())['cold_starts']} cold starts so far", flush=True)
    stop.set()
    summary = spin.summary(clock())
    summary["measured_load_s"] = actuator.measured if actuator else {}
    with open(f"{stem}_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=1)
    print(f"{summary['gpu_hours']:.2f} GPU-hours, utilization {100 * summary['gpu_utilization']:.1f}%, "
          f"{summary['cold_starts']} cold starts ({100 * summary['cold_start_rate']:.2f}% of queries)")
    return stem


if __name__ == "__main__":
    main()
