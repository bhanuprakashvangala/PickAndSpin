"""Accuracy of the keyword lists, DistilBERT and the hybrid classifier on the validation split.

The validation split is data/classifier/val.jsonl, written by pickspin.training.labels. DistilBERT
predicts all validation texts in one call. The hybrid prediction is the keyword tier where a keyword
list matches and DistilBERT's prediction elsewhere, as in Eq. 1. The figures are:

- validation_queries: the number of validation queries
- label_counts: the number of queries with each label, in order of first appearance
- distilbert_accuracy: the share of queries whose DistilBERT tier equals the label
- hybrid_accuracy: the same for the hybrid prediction
- keyword_coverage: the share of queries that a keyword list matches
- keyword_accuracy_on_matched: the share of those matched queries whose keyword tier equals the label
- majority_class_accuracy: the share of the most common label, the accuracy of always guessing it

Needs the [classifier] extra and a trained model. torch is imported only when evaluate() runs, so
importing this module is cheap.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pickspin.pick.classifier import keyword_tier

log = logging.getLogger(__name__)


def evaluate(val_path: Path, model_dir: Path) -> dict[str, Any]:
    """Return the accuracy figures of the classifier stages on the validation split in val_path.

    val_path is a JSONL file with a 'text' and a 'label' per line, and model_dir a directory written by
    `pickspin classifier train`. The keys of the result are in the order listed in the module
    docstring. The file must hold at least one query, and a keyword list must match at least one.
    Raises MissingDependencyError without the [classifier] extra and DataNotFoundError when model_dir
    holds no model.
    """
    # Imported here, not at the top, so that importing this module never loads torch.
    from pickspin.pick.distilbert import DistilBertTier

    with val_path.open(encoding="utf-8") as f:
        val: list[dict[str, Any]] = [json.loads(line) for line in f]
    log.debug("Read %d validation queries from %s", len(val), val_path)
    labels: list[str] = [v["label"] for v in val]
    model = DistilBertTier(model_dir).predict([v["text"] for v in val])
    keyword = [keyword_tier(v["text"]) for v in val]
    hybrid = [k or m for k, m in zip(keyword, model)]
    matched = [(k, y) for k, y in zip(keyword, labels) if k is not None]

    def acc(pred: Sequence[str]) -> float:
        return sum(p == y for p, y in zip(pred, labels)) / len(labels)

    return {
        "validation_queries": len(val),
        "label_counts": dict(Counter(labels)),
        "distilbert_accuracy": acc(model),
        "hybrid_accuracy": acc(hybrid),
        "keyword_coverage": len(matched) / len(val),
        "keyword_accuracy_on_matched": sum(k == y for k, y in matched) / len(matched),
        "majority_class_accuracy": max(Counter(labels).values()) / len(labels),
    }


def write_evaluation(result: Mapping[str, Any], out_path: Path) -> None:
    """Write the evaluation to out_path as JSON indented by one space, creating the directory if needed.

    The file ends with a newline and is written in text mode, so its line ends are the platform's.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    log.debug("Wrote %s", out_path)


def format_evaluation(result: Mapping[str, Any]) -> str:
    """Return the evaluation as a table: one line per figure, its name padded to 28 characters.

    Fractions are shown with four decimals; counts and the label counts are shown as they are. The
    text has no trailing newline.
    """
    return "\n".join(f"{k:28s} {v:.4f}" if isinstance(v, float) else f"{k:28s} {v}" for k, v in result.items())
