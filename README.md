# Pick and Spin

Code and experiment traces for **Pick and Spin: Cold-Start-Aware Routing for Self-Hosted LLM Serving**
Bhanu Prakash Vangala, Tanu Malik.
IEEE CLOUD 2026. [Paper (PDF)](https://bhanuprakashvangala.github.io/files/papers/pick-and-spin.pdf) |
[Code](https://github.com/bhanuprakashvangala/PickAndSpin)

Pick and Spin serves nine self-hosted LLMs (1B to 27B parameters) with vLLM on Kubernetes.

- **Pick** (`src/pickspin/pick.py`, `classifier.py`) puts each query into a tier: three keyword lists first, and a
  fine-tuned DistilBERT for queries that match no list (Eq. 1). Within the tier it samples each model's success rate
  from a Beta posterior blended with the tier's posterior (Eqs. 2-3) and picks the model with the highest
  S(m) = 0.7 * mu_HTS + 0.3 * L_norm + 0.1 / sqrt(n + 1) (Eq. 4). L_norm comes from the latency Spin reports, which
  includes the cold-start time of a model that is not warm.
- **Spin** (`src/pickspin/spin.py`) keeps every model COLD, LOADING or WARM. A query routed to a cold model scales
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

## Layout

```
data/queries.jsonl.gz           the 31,019 prompts from 8 benchmarks, as sent to the models
data/query_tiers.csv.gz         the hybrid classifier's tier for every prompt (input to the simulator)
results/traces/                 experiment traces (no model responses)
  static_baseline.csv.gz        every query on every model: success, latency, token counts
  judgments.csv.gz              correct/incorrect label per (query, model) from the judge
  pick_spin_routed.csv.gz       the routed run: tier, chosen model, latency, cold-start flag, in arrival order
results/classifier/             label counts of the DistilBERT training data
scripts/reproduce.py            regenerates the paper's tables and figures from results/traces/
src/pickspin/                   Pick and Spin: classifier, pick, spin, config, simulate.py, run_live.py
src/recorded_run/               the runner that recorded results/traces/pick_spin_routed.csv.gz (see Results)
src/baseline/                   static baseline runner and the LLM judge
src/classifier/                 complexity labels and DistilBERT training
deploy/                         Helm chart (nine vLLM Deployments, router RBAC), endpoint map, router Job and image
tests/                          unit tests and a live-runner test against a stub cluster
```

## Setup

```bash
git clone https://github.com/bhanuprakashvangala/PickAndSpin.git
cd PickAndSpin
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest tests           # about 5 seconds
```

## Reproduce

### Paper tables and figures from the traces (no GPU, about 10 seconds)

```bash
python scripts/reproduce.py
```

This writes to `results/`:

- `static_baseline_summary.csv`, `static_baseline_per_benchmark_accuracy.csv`: Sec. VII-A and Fig. 2
- `table1_routing.csv`: Table I
- `table2_scored_routed_queries_by_size.csv`: routed queries per model size (Table II)
- `figures/fig2a_throughput.png`, `figures/fig2b_latency.png`, `figures/fig3_thompson.png`
- `verification.csv`: each value next to the one printed in the paper

It prints the comparison and exits non-zero if any value differs.

### Simulation (no GPU, about 10 minutes)

```bash
python src/pickspin/simulate.py                                   # 250 closed-loop clients
python src/pickspin/simulate.py --arrival-rate 8 4 2 1 0.5 0.25    # Poisson arrivals, queries per second
```

The simulator runs the code in `src/pickspin` unchanged on a simulated clock. A query routed to a model takes the
latency and success recorded for that query on that model in the static baseline, and its correctness is the judge
label. Cold starts take the per-model time of Eq. 5, loads that overlap share 1.2 GB/s, and queries for a model that
is loading wait for it. Each load is run with five seeds for four policies:

- `pick-and-spin`: scale to zero; Pick scores Spin's latency, including the cold-start time of a cold model
- `pick-and-spin-observed`: scale to zero; Pick scores the mean latency it has observed, cold-start waits included
- `unaware`: scale to zero; Pick scores inference latency only
- `static`: every model warm for the whole run

Results go to `results/simulation/<load>/` (per-seed summary, per model, cold starts per tier; `--write-queries`
adds a per-query trace) and `results/simulation/overview.csv`, which averages the seeds and compares each policy's
GPU-hours with the static deployment. Tiers come from `data/query_tiers.csv.gz`, the hybrid classifier's output for
every query; `--reclassify` recomputes it with the trained DistilBERT.

A model in the simulator serves any number of queries at once with their recorded latencies; `--max-concurrency`
caps that.

### Classifier labels and DistilBERT (GPU recommended)

```bash
pip install -r requirements-classifier.txt
python src/classifier/generate_labels.py   # writes data/classifier/{train,val}.jsonl
python src/classifier/train_distilbert.py  # writes models/distilbert-complexity-classifier
```

`generate_labels.py` labels each query with the smallest model group the judge marks correct: SIMPLE (1B to 3B),
MEDIUM (7B to 14B) or COMPLEX (Gemma-3-27B, Llama-3-70B, Kimi-K2, or no model correct), then splits 80/20 into
24,815 training and 6,204 validation queries. `train_distilbert.py` fine-tunes distilbert-base-uncased on them, and
`evaluate.py` reports the accuracy of the keyword lists, DistilBERT and the hybrid classifier on the validation split.

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

Every Deployment starts at zero replicas. Build the router image (`deploy/Dockerfile`, after training or copying the
classifier into `models/`), set it in `deploy/router-job.yaml` and start the Job, which runs as the ServiceAccount the
chart creates and is allowed to scale the model Deployments:

```bash
docker build -f deploy/Dockerfile -t <registry>/pick-and-spin-router . && docker push <registry>/pick-and-spin-router
kubectl -n pick-and-spin apply -f deploy/router-job.yaml
```

`run_live.py` scales every model to zero, routes the queries with 250 workers, and writes
`results/live/pick_spin_<time>.jsonl` and a summary with the GPU-hours, utilization and cold starts that Spin
measured. `--static` keeps every model running instead (install the chart with `--set startReplicas=1`), and
`--limit 100` gives a quick check. The static baseline and the judge:

```bash
python src/baseline/run_static_baseline.py --endpoints deploy/endpoints.example.json   # 9 x 31,019 runs
cp .env.example .env    # set JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL
python src/baseline/llm_judge.py
```

## Results

`scripts/reproduce.py` computes every value below from `results/traces/`. Each one equals the value in the paper;
`results/verification.csv` lists all 84 comparisons.

The routed trace was recorded with the runner kept in `src/recorded_run/`: keyword rules only (no DistilBERT), the 14B
model in the medium tier, Pick scoring inference latency, cold starts recorded as a fixed per-model penalty while
all nine servers kept running, and google/gemma-2-27b-it as the 27B server. The static baseline used Gemma-3-27B.
Correctness comes from an LLM judge (gpt-oss-120b, prompts in `src/baseline/llm_judge.py`) applied to the static
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
| Unscored (no judge label) | 2,635 | 8.5 | - | - |
| Total | 31,019 | 100 | 49.7 | 23.4 |

The Total accuracy and latency are over the 29,781 queries that have a label for at least one of the nine models.

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
