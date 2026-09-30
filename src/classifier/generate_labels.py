"""
Generate Complexity Labels from Accuracy Data

This script creates labeled training data for DistilBERT classifier.
Labels are derived from which model tier successfully answers each query.

Label Logic:
- SIMPLE: At least one small model (1-3B) gets it correct
- MEDIUM: No small model correct, but medium model (7-14B) correct
- COMPLEX: Only large models (27B+) can answer correctly
"""

import csv
import gzip
import json
import os
from collections import defaultdict
from pathlib import Path
import random

# Model tier definitions
SMALL_MODELS = ['llama3.2_1B', 'llama3.2_3B', 'gemma2_2B', 'qwen2.5_1.5B']
MEDIUM_MODELS = ['llama3.1_8B', 'gemma2_9B', 'qwen2.5_7B', 'qwen2.5_14B']
LARGE_MODELS = ['gemma3_27B', 'llama3_70B', 'kimi_1T_MoE']

def load_accuracy_data(judgments_csv):
    """Load judge labels from results/traces/judgments.csv.gz: {(id, model): is_correct}"""
    accuracy_data = {}
    with gzip.open(judgments_csv, "rt", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            accuracy_data[(row["id"], row["model"])] = row["is_correct"] == "1"
    print(f"Loaded {len(accuracy_data)} accuracy entries")
    return accuracy_data

def load_queries(queries_file):
    """Load prompts from data/queries.jsonl.gz: {id: {query, benchmark}}"""
    queries = {}
    with gzip.open(queries_file, "rt", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            queries[d["id"]] = {"query": d["query"], "benchmark": d["benchmark"]}
    print(f"Loaded {len(queries)} unique queries")
    return queries

def get_complexity_label(query_id, accuracy_data):
    """
    Determine complexity label based on which model tier gets it correct.

    Returns: 'SIMPLE', 'MEDIUM', or 'COMPLEX'
    """
    # Check if any small model gets it correct
    small_correct = any(
        accuracy_data.get((query_id, model), False)
        for model in SMALL_MODELS
    )
    if small_correct:
        return 'SIMPLE'

    # Check if any medium model gets it correct
    medium_correct = any(
        accuracy_data.get((query_id, model), False)
        for model in MEDIUM_MODELS
    )
    if medium_correct:
        return 'MEDIUM'

    # Check if any large model gets it correct
    large_correct = any(
        accuracy_data.get((query_id, model), False)
        for model in LARGE_MODELS
    )
    if large_correct:
        return 'COMPLEX'

    # No model got it correct - label as COMPLEX (hardest)
    return 'COMPLEX'

def generate_labeled_dataset(judgments_csv, queries_file, output_dir):
    """Generate labeled dataset for DistilBERT training"""

    # Load data
    accuracy_data = load_accuracy_data(judgments_csv)
    queries = load_queries(queries_file)

    # Generate labels
    labeled_data = []
    label_counts = defaultdict(int)
    benchmark_counts = defaultdict(lambda: defaultdict(int))

    for query_id, query_info in queries.items():
        label = get_complexity_label(query_id, accuracy_data)

        labeled_data.append({
            'id': query_id,
            'text': query_info['query'],
            'label': label,
            'benchmark': query_info['benchmark']
        })

        label_counts[label] += 1
        benchmark_counts[query_info['benchmark']][label] += 1

    # Print statistics
    print("\n" + "="*60)
    print("LABEL DISTRIBUTION")
    print("="*60)
    total = len(labeled_data)
    for label in ['SIMPLE', 'MEDIUM', 'COMPLEX']:
        count = label_counts[label]
        pct = count / total * 100
        print(f"{label}: {count:,} ({pct:.1f}%)")

    print("\n" + "="*60)
    print("DISTRIBUTION BY BENCHMARK")
    print("="*60)
    for benchmark in sorted(benchmark_counts.keys()):
        print(f"\n{benchmark}:")
        for label in ['SIMPLE', 'MEDIUM', 'COMPLEX']:
            count = benchmark_counts[benchmark][label]
            print(f"  {label}: {count}")

    # Shuffle and split into train/val
    random.seed(42)
    random.shuffle(labeled_data)

    split_idx = int(len(labeled_data) * 0.8)
    train_data = labeled_data[:split_idx]
    val_data = labeled_data[split_idx:]

    print(f"\n" + "="*60)
    print(f"TRAIN/VAL SPLIT")
    print("="*60)
    print(f"Training samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")

    # Save datasets
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, 'train.jsonl')
    val_path = os.path.join(output_dir, 'val.jsonl')
    stats_path = os.path.join(output_dir, 'label_stats.json')

    with open(train_path, 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')

    with open(val_path, 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')

    # Save statistics
    stats = {
        'total_samples': total,
        'train_samples': len(train_data),
        'val_samples': len(val_data),
        'label_distribution': dict(label_counts),
        'benchmark_distribution': {k: dict(v) for k, v in benchmark_counts.items()},
        'model_tiers': {
            'small': SMALL_MODELS,
            'medium': MEDIUM_MODELS,
            'large': LARGE_MODELS
        }
    }

    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nSaved:")
    print(f"  - {train_path}")
    print(f"  - {val_path}")
    print(f"  - {stats_path}")

    return stats

if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    judgments_csv = root / "results" / "traces" / "judgments.csv.gz"
    queries_file = root / "data" / "queries.jsonl.gz"
    output_dir = root / "data" / "classifier"

    print("="*60)
    print("GENERATING COMPLEXITY LABELS FOR DISTILBERT")
    print("="*60)
    stats = generate_labeled_dataset(str(judgments_csv), str(queries_file), str(output_dir))
