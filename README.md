# Pick and Spin

Code and experiment traces for **Pick and Spin: Cold-Start-Aware Routing for Self-Hosted LLM Serving**
Bhanu Prakash Vangala, Tanu Malik.
IEEE CLOUD 2026. [Paper (PDF)](https://bhanuprakashvangala.github.io/files/papers/pick-and-spin.pdf) |
[Code](https://github.com/bhanuprakashvangala/PickAndSpin)

Pick and Spin routes each query to one of nine self-hosted LLMs (1B to 27B parameters) served with vLLM.
Pick puts a query into a complexity tier with keyword rules and picks a model in that tier with Thompson Sampling,
scoring each model on its sampled success rate, its mean inference latency and an exploration bonus; the reward is
whether the request succeeded. Spin keeps a WARM/COLD state per model (a model turns cold after 300 s without
traffic) and, when a query reaches a cold model, adds that model's cold-start penalty (`src/pickspin/config.py`)
to the query's total latency.

## Models

The routed run used nine vLLM servers, one per model. The static baseline ran every query on the same eight smaller
models and on Gemma-3-27B.

| Size | Routed run | Static baseline | Pick tier |
|---|---|---|---|
| 1B | meta-llama/Llama-3.2-1B-Instruct | same | SIMPLE |
| 1.5B | Qwen/Qwen2.5-1.5B-Instruct | same | SIMPLE |
| 2B | google/gemma-2-2b-it | same | SIMPLE |
| 3B | meta-llama/Llama-3.2-3B-Instruct | same | SIMPLE |
| 7B | Qwen/Qwen2.5-7B-Instruct | same | MEDIUM |
| 8B | meta-llama/Llama-3.1-8B-Instruct | same | MEDIUM |
| 9B | google/gemma-2-9b-it | same | MEDIUM |
| 14B | Qwen/Qwen2.5-14B-Instruct | same | MEDIUM |
| 27B | google/gemma-2-27b-it | Gemma-3-27B | COMPLEX |

Correctness comes from an LLM judge (gpt-oss-120b, prompts in `src/baseline/llm_judge.py`) applied to the static
baseline responses. A routed query is scored with the judge label of the model it was routed to (for the 27B server,
the Gemma-3-27B label). Routed queries whose model has no label for that query are reported as unscored.

## Layout

```
data/queries.jsonl.gz           the 31,019 prompts from 8 benchmarks, as sent to the models
results/traces/                 experiment traces (no model responses)
  static_baseline.csv.gz        every query on every model: success, latency, token counts
  judgments.csv.gz              correct/incorrect label per (query, model) from the judge
  pick_spin_routed.csv.gz       the routed run: tier, chosen model, latency, cold-start flag, in arrival order
results/classifier/             label counts of the DistilBERT training data
scripts/reproduce.py            regenerates the paper's tables and figures from results/traces/
src/pickspin/                   Pick (pick.py), Spin (spin.py), config, live runner
src/baseline/                   static baseline runner and the LLM judge
src/classifier/                 complexity labels and DistilBERT training
deploy/                         Helm chart for the nine vLLM servers (written for this release), endpoint map, router Job
```

The static baseline and judge traces also contain Llama-3-70B and Kimi-K2, which are used only to build the
classifier labels.

## Setup

```bash
git clone https://github.com/bhanuprakashvangala/PickAndSpin.git
cd PickAndSpin
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Reproduce

### Tables and figures from the traces (no GPU, about 10 seconds)

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

### Classifier labels and DistilBERT (GPU recommended)

```bash
pip install -r requirements-classifier.txt
python src/classifier/generate_labels.py   # writes data/classifier/{train,val}.jsonl
python src/classifier/train_distilbert.py  # writes models/
```

`generate_labels.py` labels each query with the smallest model group the judge marks correct: SIMPLE (1B to 3B),
MEDIUM (7B to 14B) or COMPLEX (Gemma-3-27B, Llama-3-70B, Kimi-K2, or no model correct), then splits 80/20 into
24,815 training and 6,204 validation queries. `train_distilbert.py` fine-tunes distilbert-base-uncased on them.
The router in `src/pickspin` uses keyword rules and does not load this model. The trained weights (255 MB) are not
included; they are available on request or can be retrained with the script.

### Live experiments on a Kubernetes cluster

You need NVIDIA GPUs, the NVIDIA device plugin, a ReadWriteMany storage class, and a Hugging Face token
with access to the Llama and Gemma weights.

```bash
kubectl create namespace pick-and-spin
kubectl -n pick-and-spin create secret generic hf-token --from-literal=HF_TOKEN=$HF_TOKEN
helm install pick-and-spin deploy/helm/pick-and-spin -n pick-and-spin \
  --set storage.storageClass=<your-rwx-class>
```

`deploy/endpoints.example.json` maps model keys to the in-cluster Service names. From a pod in the namespace
(for example with `deploy/router-job.yaml` after building an image from this repository), or from your machine
after `kubectl port-forward` and an edited endpoint file:

```bash
python src/baseline/run_static_baseline.py --endpoints deploy/endpoints.example.json   # 9 x 31,019 runs
cp .env.example .env    # set JUDGE_API_BASE, JUDGE_API_KEY, JUDGE_MODEL
python src/baseline/llm_judge.py
python src/pickspin/run_live.py --endpoints deploy/endpoints.example.json --workers 250
```

Use `--limit 100` on either runner for a quick check. Outputs go to `results/live/`.

## Results

`scripts/reproduce.py` computes every value below from `results/traces/`. Each one equals the value in the paper;
`results/verification.csv` lists all 84 comparisons.

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
