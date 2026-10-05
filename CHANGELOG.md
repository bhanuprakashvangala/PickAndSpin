# Changelog

## 2.0.0

Pick and Spin is now an installable Python package, `pick-and-spin` (imported as `pickspin`), with one command,
`pickspin`, in place of the scripts. The code is organized by the paper: `pickspin.pick` (Sec. IV), `pickspin.spin`
(Sec. V), `pickspin.simulation`, `pickspin.live`, `pickspin.baseline`, `pickspin.training` and `pickspin.paper`
(Sec. VII). The README maps each equation to its module.

No numeric change (enforced by `tests/data/golden` and the CI golden job). In the golden configurations (the default
closed-loop grid, two Poisson rates and a stress run) every simulator output is byte-identical to that of version
1.1.0, with the gzipped per-query traces compared after decompression; four runs match it at full precision in every
field of every query record; and `pickspin reproduce` writes the same tables and matches all 84 values of the paper.
Seeds, random streams, output file names and formats, and environment variables are unchanged.

### Commands

| 1.1.0 | 2.0.0 |
|---|---|
| `pip install -r requirements.txt` | `pip install -e ".[dev]"` (add `live` and `classifier` for live runs) |
| `pip install -r requirements-classifier.txt` | `pip install -e ".[train]"` |
| `python -m pytest tests` | `pytest` |
| `python scripts/reproduce.py` | `pickspin reproduce` |
| `python src/pickspin/simulate.py ...` | `pickspin simulate ...` |
| `python src/pickspin/run_live.py ...` | `pickspin live ...` |
| `python src/baseline/run_static_baseline.py ...` | `pickspin baseline run ...` |
| `python src/baseline/llm_judge.py [benchmark ...]` | `pickspin baseline judge [benchmark ...]` |
| `python src/classifier/generate_labels.py` | `pickspin classifier labels` |
| `python src/classifier/train_distilbert.py` | `pickspin classifier train` |
| `python src/classifier/evaluate.py` | `pickspin classifier evaluate` |
| `python src/recorded_run/run_live.py ...` | `python legacy/recorded_run/run_live.py ...` |

Every flag of the old scripts keeps its name and default. Tag `v1.1.0` keeps the old layout and commands.

### Changed

- Installation: the package declares its dependencies, with the extras `live` (kubernetes), `classifier` (torch,
  transformers), `train` (fine-tuning) and `dev` (tests and checks), in place of `requirements.txt` and
  `requirements-classifier.txt`. The base install (matplotlib, requests) runs `reproduce`, `simulate`,
  `baseline run`, `baseline judge` and `classifier labels`. Python 3.11 or later.
- Progress messages go to stderr through logging; `-q` keeps only warnings and `-v` adds debug output. stdout
  carries only command results: the reproduce comparison and the evaluate table.
- A missing input file, a missing optional dependency, an unset `JUDGE_API_BASE` or a `JUDGE_WORKERS` that is not
  an integer ends the command with status 1 and one `pickspin: error:` line that says what to do.
- `pickspin simulate` exits with status 1 and a message for settings under which a run cannot finish or means
  nothing: `--workers 0` without `--arrival-rate`, a `--max-concurrency` below 1, or a negative or NaN
  `--arrival-rate`. Version 1.1.0 crashed with a `TypeError` on the first two and wrote meaningless output for the
  last. `--arrival-rate 0` still means the closed loop.
- Default paths are relative to the current directory, or to the new global `--root DIR`, instead of to each
  script's location, so commands run from the repository root behave as before. Every command has flags for its
  input and output paths; `PS_ENDPOINTS` and `PS_CLASSIFIER` still apply when a flag is not given.
- `pickspin live --cooldown S` sets T_cooldown (default 300 s).
- `pickspin baseline judge --workers N` sets the number of parallel judge requests (default `$JUDGE_WORKERS`, else
  100), and unknown benchmark names end the command with status 1 instead of writing an empty file. It also exits
  with status 1 when the responses directory is missing; the old script wrote an empty file for each benchmark.
- `pickspin baseline run` checks that the endpoint map has every requested model before it sends any query; the old
  script raised a `KeyError` on reaching the first model without an endpoint, after finishing the models before it.
- `pickspin classifier train` exits with status 1 when the training data is missing; the old script printed an error
  and exited with status 0.
- `pickspin classifier evaluate` creates `results/classifier/` when it does not exist.
- `pickspin reproduce` reports the `--out` directory it wrote to; the old script always named `results/`.
- Environment variables are read when a command runs, not when a module is imported.
- `deploy/Dockerfile` installs the package with the `live` and `classifier` extras and has `pickspin` as its
  entrypoint; `deploy/router-job.yaml` runs `pickspin live`.
- The runner that recorded the routed trace moved from `src/recorded_run/` to `legacy/recorded_run/`, byte for byte.
- Tests are split into `tests/unit` and `tests/integration`. `pytest -m slow` compares the simulator with goldens
  recorded from tag `v1.1.0` by `scripts/make_goldens.py`. Ruff, strict mypy, pre-commit and GitHub Actions CI are
  set up.

## 1.1.0

Pick and Spin as described in the paper: the hybrid classifier (keyword lists, then DistilBERT), Pick scoring the
latency Spin reports, Spin's COLD/LOADING/WARM lifecycle with scale to zero and cold starts sharing the storage
bandwidth, the trace-driven simulator with the classifier's tiers cached in `data/query_tiers.csv.gz`, the classifier
evaluation, the router image and RBAC, and tests. The runner that recorded the routed trace was kept in
`src/recorded_run/`.

## 1.0.0

Code and experiment traces for the paper: the benchmark queries, the static-baseline, judge and routed-run traces,
`scripts/reproduce.py` for the paper's tables and figures, the static baseline runner and the LLM judge, the
classifier labels and training, the Helm chart, and the runner that recorded the routed trace.
