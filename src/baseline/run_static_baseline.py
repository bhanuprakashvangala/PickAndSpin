"""Static baseline: send every query to every model (no routing).

    python src/baseline/run_static_baseline.py --endpoints deploy/endpoints.json
    python src/baseline/run_static_baseline.py --models qwen2.5_7B llama3.1_8B --limit 100

Each model is an OpenAI-compatible endpoint (vLLM). Output goes to
results/live/static/<model>_results.jsonl and a run can be resumed.
"""

import argparse
import gzip
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pickspin"))
from config import MODELS, QUERIES_FILE, load_endpoints  # noqa: E402

lock = threading.Lock()


def call(ep, model, query, max_tokens, headers):
    payload = {"model": ep.get("model", MODELS[model]["hf_id"]),
               "messages": [{"role": "user", "content": query}],
               "max_tokens": max_tokens, "temperature": 0.1, "stream": False}
    start = time.time()
    try:
        r = requests.post(ep["base_url"].rstrip("/") + "/v1/chat/completions",
                          json=payload, headers=headers, timeout=120)
        ms = (time.time() - start) * 1000
        if r.status_code == 200:
            d = r.json()
            u = d.get("usage", {})
            return {"success": True, "response": d["choices"][0]["message"].get("content", ""),
                    "latency_ms": ms, "prompt_tokens": u.get("prompt_tokens", 0),
                    "completion_tokens": u.get("completion_tokens", 0), "error": None}
        err = f"HTTP {r.status_code}"
    except requests.exceptions.Timeout:
        ms, err = (time.time() - start) * 1000, "Timeout"
    except Exception as e:
        ms, err = (time.time() - start) * 1000, str(e)
    return {"success": False, "response": "", "latency_ms": ms, "prompt_tokens": 0,
            "completion_tokens": 0, "error": err}


def run_model(model, ep, queries, out_dir, workers, max_tokens, headers):
    out = out_dir / f"{model}_results.jsonl"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {json.loads(line)["id"] for line in f}
    todo = [q for q in queries if q["id"] not in done]
    print(f"{model}: {len(todo)} queries to run ({len(done)} already done)")
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(call, ep, model, q["query"], max_tokens, headers): q for q in todo}
        for fut in as_completed(futs):
            q = futs[fut]
            rec = {"id": q["id"], "benchmark": q["benchmark"], "model": model,
                   "query": q["query"][:500], "ground_truth": q["ground_truth"],
                   "query_type": q["query_type"], **fut.result(),
                   "timestamp": datetime.now().isoformat()}
            with lock:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoints")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--workers", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    endpoints = load_endpoints(args.endpoints)
    key = os.environ.get("VLLM_API_KEY")
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    with gzip.open(QUERIES_FILE, "rt", encoding="utf-8") as f:
        queries = [json.loads(line) for line in f][: args.limit]
    out_dir = ROOT / "results" / "live" / "static"
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in args.models:
        run_model(m, endpoints[m], queries, out_dir, args.workers, args.max_tokens, headers)


if __name__ == "__main__":
    main()
