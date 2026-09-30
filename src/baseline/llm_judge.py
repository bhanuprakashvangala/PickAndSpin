"""Label static-baseline responses CORRECT / INCORRECT with an LLM judge.

The released labels (results/traces/judgments.csv.gz) were produced with
gpt-oss-120b behind an OpenAI-compatible API. Configure the judge with:

    export JUDGE_API_BASE=https://your-openai-compatible-endpoint/v1
    export JUDGE_API_KEY=...          # see .env.example
    export JUDGE_MODEL=gpt-oss        # served model name
    python src/baseline/llm_judge.py [benchmark ...]

Reads results/live/static/*_results.jsonl, writes results/live/judgments/<benchmark>_accuracy.jsonl.
"""

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "results" / "live" / "static"
OUTPUT_DIR = ROOT / "results" / "live" / "judgments"
API_BASE = os.environ.get("JUDGE_API_BASE", "").rstrip("/")
API_KEY = os.environ.get("JUDGE_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "gpt-oss")
MAX_WORKERS = int(os.environ.get("JUDGE_WORKERS", "100"))

BENCHMARKS = ["gsm8k", "math", "arc", "mmlu_pro", "hellaswag", "truthfulqa", "humaneval", "mbpp"]
PROMPTS = {
    "gsm8k": "Question: {question}\nCorrect Answer: {ground_truth}\nModel Response: {response}\n\nDoes the model's final numerical answer equal {ground_truth}? Reply ONLY: CORRECT or INCORRECT",
    "math": "Question: {question}\nCorrect Answer: {ground_truth}\nModel Response: {response}\n\nDoes the model get the correct final answer? Reply ONLY: CORRECT or INCORRECT",
    "arc": "Question: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model select {ground_truth}? Reply ONLY: CORRECT or INCORRECT",
    "mmlu_pro": "Question: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model select {ground_truth}? Reply ONLY: CORRECT or INCORRECT",
    "hellaswag": "Context: {question}\nCorrect: {ground_truth}\nResponse: {response}\n\nDid model choose {ground_truth}? Reply ONLY: CORRECT or INCORRECT",
    "truthfulqa": "Question: {question}\nAcceptable: {ground_truth}\nResponse: {response}\n\nIs response factually correct? Reply ONLY: CORRECT or INCORRECT",
    "humaneval": "Task: {question}\nReference: {ground_truth}\nCode: {response}\n\nIs code correct? Reply ONLY: CORRECT or INCORRECT",
    "mbpp": "Task: {question}\nReference: {ground_truth}\nCode: {response}\n\nIs code correct? Reply ONLY: CORRECT or INCORRECT",
}
lock = threading.Lock()


def judge(item, benchmark):
    prompt = PROMPTS[benchmark].format(question=str(item.get("query", ""))[:2000],
                                       ground_truth=str(item.get("ground_truth", ""))[:1500],
                                       response=str(item.get("response", ""))[:2000])
    try:
        r = requests.post(f"{API_BASE}/chat/completions",
                          headers={"Authorization": f"Bearer {API_KEY}"},
                          json={"model": JUDGE_MODEL, "messages": [{"role": "user", "content": prompt}],
                                "max_tokens": 500, "temperature": 0.0}, timeout=180)
        if r.status_code != 200:
            return {"id": item["id"], "model": item["model"], "benchmark": benchmark,
                    "is_correct": None, "judge": "", "error": f"HTTP {r.status_code}"}
        msg = r.json().get("choices", [{}])[0].get("message", {})
        text = (msg.get("content") or msg.get("reasoning_content") or "").strip().upper()
        # "INCORRECT" contains "CORRECT", so test it first
        ok = False if "INCORRECT" in text else "CORRECT" in text
        return {"id": item["id"], "model": item["model"], "benchmark": benchmark,
                "is_correct": ok, "judge": text[:100], "error": None}
    except Exception as e:
        return {"id": item["id"], "model": item["model"], "benchmark": benchmark,
                "is_correct": None, "judge": "", "error": str(e)}


def run_benchmark(benchmark):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUTPUT_DIR / f"{benchmark}_accuracy.jsonl"
    done = set()
    if out.exists():
        with open(out, encoding="utf-8") as f:
            done = {(d["id"], d["model"]) for d in map(json.loads, f)}
    items = []
    for rf in sorted(RESULTS_DIR.glob("*_results.jsonl")):
        with open(rf, encoding="utf-8") as f:
            for d in map(json.loads, f):
                if d.get("success") and d["benchmark"] == benchmark and (d["id"], d["model"]) not in done:
                    items.append(d)
    print(f"{benchmark}: {len(items)} responses to judge")
    with open(out, "a", encoding="utf-8") as f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        for fut in as_completed([pool.submit(judge, i, benchmark) for i in items]):
            with lock:
                f.write(json.dumps(fut.result()) + "\n")


if __name__ == "__main__":
    if not API_BASE:
        sys.exit("Set JUDGE_API_BASE (and JUDGE_API_KEY) first; see .env.example")
    for b in sys.argv[1:] or BENCHMARKS:
        run_benchmark(b)
