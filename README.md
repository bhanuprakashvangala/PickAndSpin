# Pick and Spin

Code and experiment traces for **Pick and Spin: Cold-Start-Aware Routing for Self-Hosted LLM Serving**
Bhanu Prakash Vangala, Tanu Malik.
IEEE CLOUD 2026. [Paper (PDF)](https://bhanuprakashvangala.github.io/files/papers/pick-and-spin.pdf) |
[Code](https://github.com/bhanuprakashvangala/PickAndSpin)

Pick and Spin serves nine self-hosted LLMs (1B to 27B parameters) with vLLM on Kubernetes.

- **Pick** (`src/pickspin/pick/`) puts each query into a tier: three keyword lists first, and a fine-tuned
  DistilBERT for queries that match no list (Eq. 1). Within the tier it samples each model's success rate from a
  Beta posterior blended with the tier's posterior (Eqs. 2-3) and picks the model with the highest
  S(m) = 0.7 * mu_HTS + 0.3 * L_norm + 0.1 / sqrt(n + 1) (Eq. 4). L_norm comes from the latency Spin reports, which
  includes the cold-start time of a model that is not warm.
- **Spin** (`src/pickspin/spin/`) keeps every model COLD, LOADING or WARM. A query routed to a cold model scales
  that model's Deployment to one replica and waits until the weights are loaded; a model with nothing in flight for
  300 s is scaled back to zero and its GPU released. Load times follow Eq. 5, and loads that overlap share the
  storage bandwidth. Spin accounts GPU-hours, GPU utilization and cold starts.

## Models

| Model | Tier | Weights (GB) | Cold start (s) |
|---|---|---|---|
| meta-llama/Llama-3.2-1B-Instruct | SIMPLE | 2 | 32 |
| Qwen/Qwen2.5-1.5B-Instruct | SIMPLE | 3 | 35 |
| google/gemma-2-2b-it | SIMPLE | 4 | 38 |
| meta-llama/Llama-3.2-3B-Instruct | SIMPLE | 6 | 32 |
| Qwen/Qwen2.5-7B-Instruct | MEDIUM | 14 | 48 |
| meta-llama/Llama-3.1-8B-Instruct | MEDIUM | 16 | 42 |
| google/gemma-2-9b-it | MEDIUM | 18 | 48 |
| Qwen/Qwen2.5-14B-Instruct | COMPLEX | 28 | 65 |
| google/gemma-3-27b-it | COMPLEX | 54 | 95 |

Cold-start times are the per-model values in `src/pickspin/config.py` (32-38 s small, 42-48 s medium, 65-95 s
large, at 1.2 GB/s shared storage). The live runner replaces them with the load times it measures.

## Install

Python 3.11 or later:

```bash
git clone https://github.com/bhanuprakashvangala/PickAndSpin.git
cd PickAndSpin
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest tests/unit                # about 10 seconds
```

This installs the `pick-and-spin` package (imported as `pickspin`) and the `pickspin` command. The base install
(matplotlib and requests) reproduces the paper, runs the simulator, the static baseline and the judge, and builds the
classifier labels. Extras add the rest:

| Extra | Installs | Needed for |
|---|---|---|
| `live` | kubernetes | `pickspin live`, except with `--static` |
| `classifier` | torch, transformers | the DistilBERT stage: `pickspin live`, `pickspin classifier evaluate`, `pickspin simulate --reclassify` |
| `train` | `classifier` plus datasets, accelerate, scikit-learn, seaborn, numpy | `pickspin classifier train` |
| `dev` | pytest, pytest-cov, ruff, mypy, types-requests, pre-commit | tests and checks (see Development) |

Combine them as needed, for example `pip install -e ".[dev,live,classifier]"`. For CPU-only torch, install it first
with `pip install torch --index-url https://download.pytorch.org/whl/cpu`, as `deploy/Dockerfile` does.

## Commands

```
pickspin [--root DIR] [-v | -q] [--version] <command> ...

pickspin reproduce              the paper's tables and figures from results/traces/, compared with the paper
pickspin simulate               trace-driven simulation of the four deployment policies
pickspin live                   a live run on a Kubernetes cluster
pickspin baseline run           the static baseline: every query on every model
pickspin baseline judge         the LLM judge's labels for the static baseline responses
pickspin classifier labels      DistilBERT training labels and the 80/20 split
pickspin classifier train       fine-tune DistilBERT
pickspin classifier evaluate    accuracy of the keyword lists, DistilBERT and the hybrid classifier
```

Run the commands from the repository root, or give the global `--root DIR` (before the command) for the directory
that holds `data/`, `results/`, `models/` and `deploy/`. Every default path is resolved under it, while paths given
to a command's flags are used as typed. Progress goes to stderr (`-q` keeps only warnings, `-v` adds debug output),
and stdout carries only results: the reproduce comparison and the evaluate table. The exit status is 0 on success;
1 for a missing input file, a missing extra or a missing setting, each reported on one `pickspin: error:` line, and
for a reproduction that does not match the paper; 2 for a usage error or a missing command, which prints the help to
stderr; 130 on Ctrl-C. `pickspin <command> --help` lists a command's flags, and `python -m pickspin` is the same
command.

The commands replace the scripts of version 1.1.0 and keep their flags and defaults. New are the global options,
flags for input and output paths, `live --cooldown` and `baseline judge --workers`; `CHANGELOG.md` lists every
change. Tag `v1.1.0` keeps the old layout.

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

## Reproduce

### Paper tables and figures from the traces (no GPU, about 5 seconds)

```bash
pickspin reproduce
```

This writes to `results/` (or to `--out DIR`):

- `static_baseline_summary.csv`, `static_baseline_per_benchmark_accuracy.csv`: Sec. VII-A and Fig. 2
- `table1_routing.csv`: Table I
- `table2_scored_routed_queries_by_size.csv`: routed queries per model size (Table II)
- `figures/fig2a_throughput.png`, `figures/fig2b_latency.png`, `figures/fig3_thompson.png`
- `verification.csv`: each value next to the one printed in the paper

It prints the comparison and exits non-zero if any value differs. The tables come out byte-identical to the committed
ones. The figures also depend on the plotting libraries: the committed ones were drawn with matplotlib 3.10.8 and
Pillow 11.3.0, and other versions render them slightly differently.

### Simulation (no GPU)

```bash
pickspin simulate                                    # 250 closed-loop clients, about 30 seconds
pickspin simulate --arrival-rate 8 4 2 1 0.5 0.25    # Poisson arrivals, queries per second, about 3 minutes
```

The simulator runs Pick and Spin (`src/pickspin/pick/`, `src/pickspin/spin/`) unchanged on a simulated clock. A
query routed to a model takes the latency and success recorded for that query on that model in the static baseline,
and its correctness is the judge label. Cold starts take the per-model time of Eq. 5, loads that overlap share
1.2 GB/s, and queries for a model that is loading wait for it. Each load is run with five seeds for four policies:

- `pick-and-spin`: scale to zero; Pick scores Spin's latency, including the cold-start time of a cold model
- `pick-and-spin-observed`: scale to zero; Pick scores the mean latency it has observed, cold-start waits included
- `unaware`: scale to zero; Pick scores inference latency only
- `static`: every model warm for the whole run

Results go to `results/simulation/<load>/` (per-seed summary, per model, cold starts per tier; `--write-queries`
adds a per-query trace) and `results/simulation/overview.csv`, which averages the seeds and compares each policy's
GPU-hours with the static deployment. Tiers come from `data/query_tiers.csv.gz`, the hybrid classifier's output for
every query; `--reclassify` recomputes it with the trained DistilBERT (`classifier` extra).

A model in the simulator serves any number of queries at once with their recorded latencies; `--max-concurrency`
caps that.

### Classifier labels and DistilBERT (GPU recommended)

```bash
pip install -e ".[train]"
pickspin classifier labels     # writes data/classifier/{train,val}.jsonl
pickspin classifier train      # writes models/distilbert-complexity-classifier
pickspin classifier evaluate   # writes results/classifier/evaluation.json
```

`classifier labels` labels each query with the smallest model group the judge marks correct: SIMPLE (1B to 3B),
MEDIUM (7B to 14B) or COMPLEX (Gemma-3-27B, Llama-3-70B, Kimi-K2, or no model correct), then splits 80/20 into
training and validation sets. `classifier train` fine-tunes distilbert-base-uncased on them, and
`classifier evaluate` reports the accuracy of the keyword lists, DistilBERT and the hybrid classifier on the
validation split.

The trained classifier is on Hugging Face:

```bash
hf download bhanuprakashvangala/pickspin-distilbert-complexity --local-dir models/distilbert-complexity-classifier
```

### Live experiments on a Kubernetes cluster

You need NVIDIA GPUs (an 80 GB GPU for Gemma-3-27B), the NVIDIA device plugin, a ReadWriteMany storage class, and a
Hugging Face token with access to the Llama and Gemma weights.

```bash
kubectl create namespace pick-and-spin
kubectl -n pick-and-spin create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
helm install pick-and-spin deploy/helm/pick-and-spin -n pick-and-spin \
  --set storage.storageClass=<your-rwx-class>
```

Every Deployment starts at zero replicas. Start the router Job; it runs `pickspin live` as the ServiceAccount the
chart creates, which is allowed to scale the model Deployments. Its image, `ghcr.io/bhanuprakashvangala/pick-and-spin`,
is built from `deploy/Dockerfile` by CI on every push to `main` and holds the package, the classifier and the queries;
to use your own build, set it in `deploy/router-job.yaml`:

```bash
kubectl -n pick-and-spin apply -f deploy/router-job.yaml
docker build -f deploy/Dockerfile -t <registry>/pick-and-spin . && docker push <registry>/pick-and-spin   # optional
```

`pickspin live` scales every model to zero, routes the queries with 250 workers, and writes
`results/live/pick_spin_<time>.jsonl` and a summary with the GPU-hours, utilization and cold starts that Spin
measured. `--static` keeps every model running instead (install the chart with `--set startReplicas=1`), and
`--limit 100` gives a quick check. Outside the image it needs the `live` and `classifier` extras. The static
baseline and the judge:

```bash
pickspin baseline run --endpoints deploy/endpoints.example.json   # 9 x 31,019 runs
cp .env.example .env              # set JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL
set -a; . ./.env; set +a          # export them
pickspin baseline judge
```

## Run it as a service

`pickspin serve` runs Pick and Spin as an OpenAI-compatible gateway in front of the model servers. Send it chat
completions as you would to any OpenAI-compatible server; with `"model": "auto"` Pick classifies the last user
message and picks a model, and Spin starts a cold model's server when a request needs it and stops servers that stay
idle for the cooldown. A request may also name one of the models directly.

```bash
pickspin serve --servers deploy/nautilus/servers.json --namespace <ns> --port 8080
curl http://localhost:8080/v1/chat/completions -H 'Content-Type: application/json'   -d '{"model": "auto", "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational."}]}'
```

The response is the model server's reply plus a `pickspin` object with the chosen model, the tier, whether the request
hit a cold start, and how long it waited. `GET /v1/models` lists the models, `GET /stats` shows the GPU-hours,
utilization, cold starts, measured load times and the state of every model, and `GET /healthz` reports liveness.
Streaming is not supported.

The gateway sends a query to a model of its tier that is already up whenever there is one, and starts a cold model
only when none is; `--explore-cold` lets Pick choose any model of the tier as in the paper, at the price of cold
starts while another model could have answered.

The gateway keeps itself running. A request waits up to `--max-wait` seconds (600 by default) for a cold model and
then gets `503` with `Retry-After` while the model keeps loading. A model whose server fails to load, or dies later
(for example when its Job reaches its deadline), goes back to COLD and the next request starts it again. The gateway
deletes every model's server when it starts and when it stops, so no GPU is left held by a server nobody routes to.

### Choosing the models

The models are not fixed: they come from the server file. Each entry names the Kubernetes Job that serves the model
and its GPU needs; a model outside the paper's pool also gives its Hugging Face id, tier, weight size and expected
cold-start time:

```json
"mistral_7B": {"name": "bhanu-pickspin-mistral-7b", "hf_id": "mistralai/Mistral-7B-Instruct-v0.3",
               "tier": "MEDIUM", "weight_gb": 15, "cold_start_s": 45, "memory": "24Gi"}
```

Add, remove or replace entries, rerun `python deploy/nautilus/render.py` to write the matching Services and endpoint
map, apply `deploy/nautilus/services.json`, and restart the gateway. Pick only chooses between the models of a
query's tier, so keep at least one model in every tier.

An entry may also set `env` (environment variables for the server), `extra_args` (more vLLM arguments) and
`placements`, other ways to run the model on the hardware there is. If no node fits the current placement for
`schedule_patience_s` seconds, the gateway recreates the server in the next one:

```json
"qwen2.5_14B": {"name": "bhanu-pickspin-qwen25-14b", "memory": "32Gi", "placements": [
  {"gpus": 1, "gpu_products": ["NVIDIA-RTX-A6000", "NVIDIA-L40", "NVIDIA-L40S"]},
  {"gpus": 2, "gpu_products": ["NVIDIA-GeForce-RTX-3090", "NVIDIA-GeForce-RTX-4090"], "env": {"NCCL_P2P_DISABLE": "1"}}]}
```

### On NRP Nautilus

Nautilus does not allow Deployments that request GPUs, so each model server runs as a Job that the gateway creates
and deletes (`src/pickspin/live/jobs.py`); only the CPU-only gateway is a Deployment. `deploy/nautilus/` holds
everything: `storage.yaml` (weight cache and results volumes), `download-weights.yaml` (fetches the weights of every
model in the server files, so cold starts read them from Ceph), `router-rbac.yaml` (lets the gateway and router
manage the server Jobs), `services.json`, `gateway.yaml`, `replay-job.yaml` (a load test of the gateway with
`pickspin replay`) and `router-job.yaml` (the benchmark replay with `pickspin live --servers`).

There are two catalogs. `servers.json`, which the gateway serves, keeps every model on one GPU that is common on
the cluster: the COMPLEX tier uses the AWQ builds of Qwen2.5-14B and Qwen2.5-32B, because 48 GB GPUs are rarely free.
`servers-paper.json` holds the paper's nine models, with Qwen2.5-14B and Gemma-3-27B on 48 GB GPUs or several 24 GB
ones, for `router-job.yaml`. After editing either file, run `python deploy/nautilus/render.py` and apply
`services.json`. The gateway reads its catalog from the ConfigMap `bhanu-pickspin-servers`, so after changing
`servers.json`, update the ConfigMap, rerun the download Job for new models, and restart the gateway; no new image is
needed. Every manifest runs the published image, so the gateway starts in seconds.

```bash
kubectl apply -f deploy/nautilus/storage.yaml -f deploy/nautilus/router-rbac.yaml
kubectl create secret generic <hf-secret> --from-literal=HF_TOKEN=...   # set its name in servers.json
kubectl create configmap bhanu-pickspin-servers --from-file=servers.json=deploy/nautilus/servers.json \
  --dry-run=client -o yaml | kubectl apply -f -                  # the gateway's catalog
kubectl apply -f deploy/nautilus/download-weights.yaml
kubectl apply -f deploy/nautilus/services.json -f deploy/nautilus/gateway.yaml
kubectl port-forward svc/bhanu-pickspin-gateway 8080:8080   # then call http://localhost:8080 as above
kubectl apply -f deploy/nautilus/replay-job.yaml            # optional: 200 benchmark queries through the gateway
```

## Package and paper

| Paper | Code in `src/pickspin/` |
|---|---|
| Hybrid classifier: keyword lists, then DistilBERT (Eq. 1) | `pick/classifier.py`, `pick/distilbert.py` |
| Beta posteriors, HTS blend and the routing score (Eqs. 2-4) | `pick/sampler.py`, `pick/router.py` |
| Spin's COLD/LOADING/WARM lifecycle, scale to zero, GPU accounting (Sec. V) | `spin/lifecycle.py` |
| Cold-start time with loads sharing the storage bandwidth (Eq. 5) | `spin/storage.py` |
| Queries waiting for a model that is loading (Eq. 6) | `simulation/engine.py` (simulated clock), `live/runner.py` (cluster) |
| The nine models, the tiers and the parameters (Sec. VI-A) | `config.py` |
| Tables I and II, Figs. 2 and 3, and the comparison with the paper (Sec. VII) | `paper/` |

The rest of the package runs the experiments:

- `simulation/`: the trace-driven simulator: the four policies, its inputs (traces and tier cache), the event loop
  and the CSV outputs
- `live/`: the live runner, its vLLM client and the Kubernetes actuator that scales the model Deployments
- `baseline/`: the static baseline and the LLM judge, with the prompts that produced the released labels
- `training/`: DistilBERT training labels, fine-tuning and evaluation
- `cli/`: the `pickspin` command; `paths.py` holds the repository layout and `data.py` reads the query file

## Layout

```
data/queries.jsonl.gz           the 31,019 prompts from 8 benchmarks, as sent to the models
data/query_tiers.csv.gz         the hybrid classifier's tier for every prompt (input to the simulator)
results/traces/                 experiment traces (no model responses)
  static_baseline.csv.gz        every query on every model: success, latency, token counts
  judgments.csv.gz              correct/incorrect label per (query, model) from the judge
  pick_spin_routed.csv.gz       the routed run: tier, chosen model, latency, cold-start flag, in arrival order
src/pickspin/                   the pickspin package and command (see Package and paper)
legacy/recorded_run/            the runner that recorded results/traces/pick_spin_routed.csv.gz (see Results)
deploy/                         Helm chart (nine vLLM Deployments, router RBAC), endpoint map, router Job and image
tests/unit/                     hermetic unit tests
tests/integration/              tests on the released data and against local stub servers
tests/data/golden/              digests of the version 1.1.0 simulator's outputs
scripts/make_goldens.py         regenerates those digests from tag v1.1.0
```

`legacy/recorded_run/` is kept byte for byte as it ran: its files are identical to `src/pickspin/` at tag `v1.0.0`,
and a test checks their git blob ids. It is not part of the package and is excluded from linting, type checking,
packaging and tests; `legacy/README.md` explains how it differs from the package and how to run it.

## Development

```bash
pip install -e ".[dev]"
pre-commit install                     # ruff and file checks on every commit
pytest tests/unit                      # hermetic unit tests, about 10 seconds
pytest -m "not slow"                   # plus the integration tests (released data, local stub servers), about 30 s
pytest -m slow                         # full-dataset runs: the simulator goldens (about a minute) and DistilBERT
ruff check . && ruff format --check .
mypy                                   # strict, over src/pickspin
```

The tests run against the installed package. The integration tests read `data/` and `results/` from the checkout,
skip without them, and write only to temporary directories. The DistilBERT test runs only with the `classifier`
extra and the trained model in `models/`.

The package keeps every number of version 1.1.0. `tests/data/golden/` holds SHA-256 digests of every output of the
1.1.0 simulator for three configurations (the default closed-loop grid, two Poisson rates, and a stress run with a
10 s cooldown and at most 8 queries per model), and full-precision fingerprints of four runs.
`tests/integration/test_simulation_golden.py` reruns them with `pickspin simulate` and must match bit for bit.
Bit-exact results depend on the CPython minor version and on the platform's math library, so the committed goldens
(Windows, CPython 3.12) are compared only there. Elsewhere the test skips unless `PICKSPIN_GOLDEN_DIR` names goldens
generated on that machine from tag `v1.1.0`, as the CI golden job does:

```bash
git worktree add ../pick-and-spin-v1.1.0 v1.1.0
python scripts/make_goldens.py --baseline ../pick-and-spin-v1.1.0 --out ../golden
PICKSPIN_GOLDEN_DIR=../golden pytest -m slow tests/integration/test_simulation_golden.py
```

Never regenerate the goldens, or the committed `results/*.csv`, from the code under test. A change that moves a
number belongs in its own release, with a new baseline tag.

CI (`.github/workflows/ci.yml`) runs ruff and mypy; the tests that are not slow on Linux with Python 3.11 to 3.13 and
on Windows with 3.12; `pickspin reproduce` on a base install without extras, which must leave `results/*.csv`
unchanged; the golden comparison against tag `v1.1.0`; and a package build.

## Results

`pickspin reproduce` computes every value below from `results/traces/`. Each one equals the value in the paper;
`results/verification.csv` lists all 84 comparisons.

The routed trace was recorded with the runner kept in `legacy/recorded_run/`: keyword rules only (no DistilBERT), the 14B
model in the medium tier, Pick scoring inference latency, cold starts recorded as a fixed per-model penalty while
all nine servers kept running, and google/gemma-2-27b-it as the 27B server. The static baseline used Gemma-3-27B.
Correctness comes from an LLM judge (gpt-oss-120b, prompts in `src/pickspin/baseline/judge.py`) applied to the static
baseline responses; a routed query is scored with the judge label of the model it was routed to (for the 27B server,
the Gemma-3-27B label).

| Runs | |
|---|---|
| Queries (8 benchmarks) | 31,019 |
| Static baseline runs (9 models x 31,019) | 279,171 |
| Total inference runs (static baseline + routed run) | 310,190 |

Static baseline (Fig. 2):

| Model | Throughput (tok/s) | Latency (s) |
|---|---|---|
| Llama-3.2-1B | 45.8 | 1.5 |
| Qwen2.5-1.5B | 26.4 | 1.5 |
| Gemma-2-2B | 24.9 | 2.1 |
| Llama-3.2-3B | 16.3 | 2.2 |
| Qwen2.5-7B | 7.5 | 4.8 |
| Llama-3.1-8B | 10.6 | 4.3 |
| Gemma-2-9B | 11.5 | 6.5 |
| Qwen2.5-14B | 5.6 | 7.6 |
| Gemma-3-27B | 4.6 | 21.7 |

Throughput falls 10x and latency rises 14x (1.51 s to 21.66 s) from Llama-3.2-1B to Gemma-3-27B. Judged accuracy:
Qwen2.5-14B 62.6%, Gemma-3-27B 56.6%, Qwen2.5-7B 56.3% (at 4.83 s, 4.5x lower latency than Gemma-3-27B).

Routed run (Table I). Latency includes the cold-start penalty.

| Model | Queries | Share (%) | Accuracy (%) | Latency (s) |
|---|---|---|---|---|
| Llama-3.2-1B | 5,319 | 17.1 | 23.2 | 1.82 |
| Qwen2.5-1.5B | 11 | 0.04 | 27.3 | 4.22 |
| Gemma-2-2B | 7 | 0.02 | 28.6 | 3.34 |
| Llama-3.2-3B | 5 | 0.02 | 20.0 | 8.29 |
| Qwen2.5-7B | 8,101 | 26.1 | 63.3 | 28.53 |
| Llama-3.1-8B | 13,462 | 43.4 | 54.5 | 28.87 |
| Gemma-2-9B | 499 | 1.6 | 73.5 | 30.52 |
| Qwen2.5-14B | 347 | 1.1 | 72.0 | 44.63 |
| Gemma-2-27B | 633 | 2.0 | 75.2 | 13.28 |
| Timed out | 2,635 | 8.5 | - | - |
| Total | 31,019 | 100 | 49.7 | 23.4 |

| Routed run | |
|---|---|
| Oracle accuracy (a query counts if any of the nine models is correct) | 86.8% |
| Queries that no model answers correctly | 13.2% |
| Share of the oracle reached by routing | 57.2% |
| Queries routed to the 14B and 27B models | 3.2% |
| Scored routed queries on 1-3B / 7-9B / 14-27B models (Table II) | 5,342 / 22,062 / 980 (total 28,384) |
| Cumulative selection rate at 31,000 queries, Llama-3.1-8B / Qwen2.5-7B / Llama-3.2-1B / Gemma-2-9B (Fig. 3) | 46.1% / 27.8% / 19.6% / 2.5% |

## Related

[Efficient Multi-Model Orchestration for Self-Hosted Large Language Models](https://github.com/bhanuprakashvangala/MultiModelOrchestration)
(DAI Workshop at AAAI 2026), the earlier version of this system.

## Citation

```bibtex
@inproceedings{vangala2026pickandspin,
  title     = {Pick and Spin: Cold-Start-Aware Routing for Self-Hosted {LLM} Serving},
  author    = {Vangala, Bhanu Prakash and Malik, Tanu},
  booktitle = {Proceedings of the IEEE International Conference on Cloud Computing (CLOUD)},
  year      = {2026}
}
```

## License

MIT. The benchmark prompts in `data/` come from HumanEval, MBPP, GSM8K, MATH, TruthfulQA, MMLU-Pro, ARC and
HellaSwag and keep their original licenses.
This work was supported by NASA (NASA-AIST-21-0095) and the National Science Foundation (CNS-1846418).
