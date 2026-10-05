"""Record golden fingerprints of the simulator from the frozen v1.1.0 code.

    git worktree add ../pick-and-spin-v1.1.0 v1.1.0
    python scripts/make_goldens.py --baseline ../pick-and-spin-v1.1.0 [--out tests/data/golden] [--force]

The v1.1.0 simulator is run in its own checkout through subprocesses. Only SHA-256 digests are kept, never the
outputs themselves:

- outputs.sha256: one line per output file (summary, per-model and cold-start CSVs, overview.csv, and each
  decompressed per-query trace) of every configuration in CONFIGS;
- manifest.json: the baseline commit, platform and Python version, CONFIGS, and full-precision fingerprints of the
  runs in FINGERPRINT_RUNS (every field of every query record, the run summary and the per-model accounting).

tests/integration/test_simulation_golden.py runs the same configurations with the current code and compares digests.
Generate goldens only from tag v1.1.0, never from the code under test.
"""

import argparse
import gzip
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

BASELINE_COMMIT = "094f53734b591d3663c185e43eb0764e5fa7f416"

CONFIGS: dict[str, list[str]] = {
    "closed-default": ["--write-queries"],
    "poisson": ["--arrival-rate", "0.25", "4", "--seeds", "0", "--write-queries"],
    "stress": [
        "--policies",
        "pick-and-spin",
        "pick-and-spin-observed",
        "unaware",
        "--seeds",
        "0",
        "--cooldown",
        "10",
        "--max-concurrency",
        "8",
        "--write-queries",
    ],
}

FINGERPRINT_RUNS: dict[str, dict[str, object]] = {
    "closed-250/pick-and-spin/0": {"policy": "pick-and-spin", "seed": 0},
    "closed-250/static/0": {"policy": "static", "seed": 0},
    "stress/pick-and-spin/0": {"policy": "pick-and-spin", "seed": 0, "cooldown_s": 10, "max_concurrency": 8},
    "poisson-0.25qps-cooldown10/pick-and-spin/0": {
        "policy": "pick-and-spin",
        "seed": 0,
        "arrival_rate": 0.25,
        "cooldown_s": 10,
    },
}

SNIPPET = r"""
import hashlib, json, sys
import simulate
from config import MODELS

runs_spec = json.loads(sys.argv[1])
queries, runs, correct = simulate.load_data()
tiers = simulate.query_tiers(queries)
out = {}
for name, kw in runs_spec.items():
    kw = dict(kw)
    policy, seed = kw.pop("policy"), kw.pop("seed")
    records, s = simulate.simulate(policy, seed, queries, runs, correct, tiers, **kw)
    lines = []
    for r in records:
        lines.append(repr((r["order"], r["id"], r["model"], str(r["tier"]), str(r["stage"]), r["arrive"],
                           r["start"], r["end"], r["infer_s"], r["wait_s"], r["total_s"], r["cold_start"],
                           r["success"], r["correct"], r["worker"])))
    lines.append(repr((s["gpu_hours"], s["busy_gpu_hours"], s["gpu_utilization"], s["cold_starts"],
                       s["cold_start_rate"], s["makespan_s"])))
    for m in MODELS:
        pm = s["per_model"][m]
        lines.append(repr((m, pm["gpu_hours"], pm["busy_gpu_hours"], pm["loading_gpu_hours"], pm["cold_starts"])))
    out[name] = hashlib.sha256("".join(line + "\n" for line in lines).encode("utf-8")).hexdigest()
print(json.dumps(out))
"""


def sha256_file(path: Path) -> str:
    data = gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_bytes()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--baseline", type=Path, required=True, help="a checkout of tag v1.1.0")
    ap.add_argument("--out", type=Path, default=Path("tests/data/golden"))
    ap.add_argument("--force", action="store_true", help="overwrite an existing golden directory")
    args = ap.parse_args()

    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=args.baseline, check=True, capture_output=True, text=True
    ).stdout.strip()
    if head != BASELINE_COMMIT:
        sys.exit(f"{args.baseline} is at {head}, not v1.1.0 ({BASELINE_COMMIT})")
    if args.out.exists() and any(args.out.iterdir()) and not args.force:
        sys.exit(f"{args.out} is not empty; pass --force to overwrite")
    args.out.mkdir(parents=True, exist_ok=True)

    digests = []
    with tempfile.TemporaryDirectory() as tmp:
        for name, extra in CONFIGS.items():
            run_dir = Path(tmp) / name
            subprocess.run(
                [sys.executable, "src/pickspin/simulate.py", *extra, "--out", str(run_dir)],
                cwd=args.baseline,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            for path in sorted(p for p in run_dir.rglob("*") if p.is_file()):
                rel = path.relative_to(Path(tmp)).as_posix().removesuffix(".gz")
                digests.append(f"{sha256_file(path)}  {rel}")

    result = subprocess.run(
        [sys.executable, "-c", SNIPPET, json.dumps(FINGERPRINT_RUNS)],
        cwd=args.baseline / "src" / "pickspin",
        check=True,
        capture_output=True,
        text=True,
    )
    fingerprints = json.loads(result.stdout.strip().splitlines()[-1])

    (args.out / "outputs.sha256").write_text("".join(d + "\n" for d in digests), encoding="utf-8")
    manifest = {
        "baseline_commit": BASELINE_COMMIT,
        "python": f"{sys.version_info.major}.{sys.version_info.minor}",
        "platform": sys.platform,
        "configs": CONFIGS,
        "fingerprints": {
            name: {"run": FINGERPRINT_RUNS[name], "sha256": fingerprints[name]} for name in FINGERPRINT_RUNS
        },
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(f"{len(digests)} output digests and {len(fingerprints)} fingerprints written to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
