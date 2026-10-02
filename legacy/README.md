# Recorded runner

`recorded_run/` is the runner that recorded `results/traces/pick_spin_routed.csv.gz`, kept exactly as it was run.
Its four files are byte-identical to `src/pickspin/` at tag `v1.0.0` (git blob ids `c5d30ba9` config.py,
`8a450a0b` pick.py, `7c0d93b1` run_live.py, `b1ee62e1` spin.py), and `tests/integration/test_legacy_provenance.py`
checks that they stay that way. It is not part of the `pickspin` package and is excluded from linting, type checking,
packaging and test collection.

It differs from the `pickspin` package in what it ran: keyword rules only (no DistilBERT stage), Qwen2.5-14B in the
medium tier, Pick scoring inference latency only, cold starts recorded as a fixed per-model penalty while all nine
servers kept running, and google/gemma-2-27b-it as the 27B server.

Run it from the repository root; it needs only `requests` and an endpoint map with a `gemma2_27B` entry, as in
`deploy/endpoints.example.json` at tag `v1.0.0`:

```bash
git show v1.0.0:deploy/endpoints.example.json > endpoints.v1.json
python legacy/recorded_run/run_live.py --endpoints endpoints.v1.json --workers 250
```

The docstring of `run_live.py` still names its original path, `src/pickspin/run_live.py`.
