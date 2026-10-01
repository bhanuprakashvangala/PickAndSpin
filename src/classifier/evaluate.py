"""Accuracy of the keyword lists, DistilBERT and the hybrid classifier on the validation split.

    python src/classifier/generate_labels.py   # writes data/classifier/val.jsonl
    python src/classifier/evaluate.py          # needs models/distilbert-complexity-classifier

Writes results/classifier/evaluation.json.
"""

import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "pickspin"))

from classifier import DistilBertTier, keyword_tier  # noqa: E402


def main():
    val = [json.loads(line) for line in open(ROOT / "data" / "classifier" / "val.jsonl", encoding="utf-8")]
    labels = [v["label"] for v in val]
    model = DistilBertTier().predict([v["text"] for v in val])
    keyword = [keyword_tier(v["text"]) for v in val]
    hybrid = [k or m for k, m in zip(keyword, model)]
    matched = [(k, y) for k, y in zip(keyword, labels) if k is not None]

    def acc(pred):
        return sum(p == y for p, y in zip(pred, labels)) / len(labels)

    result = {
        "validation_queries": len(val),
        "label_counts": dict(Counter(labels)),
        "distilbert_accuracy": acc(model),
        "hybrid_accuracy": acc(hybrid),
        "keyword_coverage": len(matched) / len(val),
        "keyword_accuracy_on_matched": sum(k == y for k, y in matched) / len(matched),
        "majority_class_accuracy": max(Counter(labels).values()) / len(labels),
    }
    out = ROOT / "results" / "classifier" / "evaluation.json"
    out.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    for k, v in result.items():
        print(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}")


if __name__ == "__main__":
    main()
