"""Fine-tune DistilBERT to classify query complexity (stage 2 of Pick's classifier).

Fine-tunes distilbert-base-uncased on the labels written by pickspin.training.labels, with a
class-weighted cross-entropy loss for the imbalanced classes and early stopping on the validation macro
F1. The hyperparameters, tokenization and outputs are those of the released model. train() writes:

- <out>/distilbert-complexity-classifier: the model and its tokenizer, which DistilBertTier loads
- <out>/training_results.json: the validation metrics and the training settings
- <out>/training_results.png: the confusion matrix and the per-class precision, recall and F1
- <out>/checkpoint-*: the Hugging Face checkpoints of each epoch

The [train] extra (torch, transformers, datasets, accelerate, scikit-learn, seaborn and numpy) is
imported when a function runs, never when this module is imported. train() imports all of it before
any work starts, so a missing package is reported at once, with the extra to install, instead of
failing after an epoch. For the same reason WeightedTrainer, a subclass of transformers.Trainer, is
defined when it is first used.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from functools import cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

from pickspin.errors import DataNotFoundError, import_optional
from pickspin.pick.distilbert import MAX_LENGTH

if TYPE_CHECKING:
    import datasets
    import torch
    import transformers
    from numpy.typing import NDArray

log = logging.getLogger(__name__)

LABEL2ID: Final[dict[str, int]] = {"SIMPLE": 0, "MEDIUM": 1, "COMPLEX": 2}
ID2LABEL: Final[dict[int, str]] = {v: k for k, v in LABEL2ID.items()}

# The modules that fine-tuning uses, in the order the original training script imported them, and
# accelerate, which transformers needs to run a Trainer.
_TRAIN_EXTRA: Final[tuple[str, ...]] = (
    "torch",
    "numpy",
    "sklearn.metrics",
    "matplotlib.pyplot",
    "seaborn",
    "transformers",
    "datasets",
    "accelerate",
)

_RULE: Final = "=" * 60


def _import_train_extra() -> None:
    """Import every module of the [train] extra, raising MissingDependencyError for the first one missing."""
    for module in _TRAIN_EXTRA:
        import_optional(module, "train")


@cache
def _weighted_trainer_class() -> type[transformers.Trainer]:
    """Define WeightedTrainer, once. It subclasses transformers.Trainer, so this imports transformers."""
    from torch import nn
    from transformers import Trainer

    class WeightedTrainer(Trainer):
        """A Trainer whose loss is cross-entropy weighted by class, for imbalanced labels.

        class_weights holds one weight per class id, or None for the unweighted loss. The other
        arguments are those of transformers.Trainer.
        """

        def __init__(self, class_weights: torch.Tensor | None = None, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        # The signature is the original one: Trainer passes num_items_in_batch by keyword, into **kwargs.
        def compute_loss(  # type: ignore[override,unused-ignore]
            self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any
        ) -> Any:
            """Return the weighted loss, and the model outputs too when return_outputs is true."""
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            if self.class_weights is not None:
                weight = self.class_weights.to(logits.device)
                loss_fct = nn.CrossEntropyLoss(weight=weight)
            else:
                loss_fct = nn.CrossEntropyLoss()

            # self.model is the model being trained, never None here, so it has a config.
            loss = loss_fct(
                logits.view(-1, self.model.config.num_labels),  # type: ignore[union-attr,unused-ignore]
                labels.view(-1),
            )
            return (loss, outputs) if return_outputs else loss

    # Named as the module attribute it is published as, for its repr and for pickle.
    WeightedTrainer.__qualname__ = "WeightedTrainer"
    return WeightedTrainer


if TYPE_CHECKING:
    WeightedTrainer: type[transformers.Trainer]
else:

    def __getattr__(name: str) -> Any:
        """Define WeightedTrainer on first access, so that importing this module does not import transformers."""
        if name == "WeightedTrainer":
            return _weighted_trainer_class()
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _load_dataset(path: Path) -> datasets.Dataset:
    """Read a JSONL split written by pickspin.training.labels into a Dataset of text and label id."""
    from datasets import Dataset

    data = []
    # The split is ASCII-only JSON, so reading it as UTF-8 gives the same text as any locale encoding.
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({"text": item["text"], "label": LABEL2ID[item["label"]]})
    return Dataset.from_list(data)


def _tokenize(examples: dict[str, list[Any]], tokenizer: Any, max_length: int = MAX_LENGTH) -> Any:
    """Tokenize a batch of texts, each padded or truncated to exactly max_length tokens."""
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=max_length)


def compute_metrics(eval_pred: tuple[NDArray[Any], NDArray[Any]]) -> dict[str, float]:
    """Return the accuracy and the macro F1 of the logits in eval_pred against its label ids."""
    import numpy as np
    from sklearn.metrics import f1_score

    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average="macro")
    return {"accuracy": accuracy, "macro_f1": macro_f1}


def plot_training_results(
    trainer: transformers.Trainer, output_dir: Path, val_dataset: datasets.Dataset
) -> dict[str, Any]:
    """Draw the validation results to output_dir/training_results.png and return the classification report.

    The left panel is the confusion matrix and the right panel the precision, recall and F1 of each
    class, with the overall accuracy. The report is scikit-learn's classification_report as a dict.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from sklearn.metrics import classification_report, confusion_matrix

    classes = list(LABEL2ID)  # the class names in id order

    # Predictions on the validation set
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    report: dict[str, Any] = classification_report(labels, preds, target_names=classes, output_dict=True)
    cm = confusion_matrix(labels, preds)

    # Large fonts
    plt.rcParams.update(
        {"font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18, "xtick.labelsize": 16, "ytick.labelsize": 16}
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: the confusion matrix
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax1,
        annot_kws={"size": 18},
        cbar_kws={"shrink": 0.8},
    )
    ax1.set_xlabel("Predicted", fontsize=18, fontweight="bold")
    ax1.set_ylabel("Actual", fontsize=18, fontweight="bold")
    ax1.set_title("Confusion Matrix", fontsize=20, fontweight="bold")

    # Right: the per-class metrics
    ax2 = axes[1]
    x = np.arange(len(classes))
    width = 0.25

    precision = [report[c]["precision"] for c in classes]
    recall = [report[c]["recall"] for c in classes]
    f1 = [report[c]["f1-score"] for c in classes]

    bars1 = ax2.bar(x - width, precision, width, label="Precision", color="#2ecc71")
    bars2 = ax2.bar(x, recall, width, label="Recall", color="#3498db")
    bars3 = ax2.bar(x + width, f1, width, label="F1-Score", color="#9b59b6")

    ax2.set_ylabel("Score", fontsize=18, fontweight="bold")
    ax2.set_xlabel("Complexity Class", fontsize=18, fontweight="bold")
    ax2.set_title("Classification Metrics", fontsize=20, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=16)
    ax2.legend(loc="lower right", fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(
        y=report["accuracy"], color="r", linestyle="--", linewidth=2, label=f"Accuracy: {report['accuracy']:.1%}"
    )

    # The value of each bar above it
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=12,
            )

    # The overall accuracy in the top left corner
    ax2.text(
        0.02,
        0.98,
        f"Overall Accuracy: {report['accuracy']:.1%}",
        transform=ax2.transAxes,
        fontsize=16,
        fontweight="bold",
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.tight_layout()

    fig_path = output_dir / "training_results.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    log.info("\nSaved figure: %s", fig_path)
    return report


def train(data_dir: Path, output_dir: Path, *, base_model: str = "distilbert-base-uncased") -> dict[str, Any]:
    """Fine-tune base_model on data_dir/train.jsonl, validate it on data_dir/val.jsonl and save the results.

    Trains for up to 5 epochs, with batches of 32 (64 for evaluation), learning rate 3e-5, weight decay
    0.01 and 10% warm-up; evaluates and saves a checkpoint after every epoch, stops after 3 epochs
    without a better validation macro F1, and keeps the best epoch. Uses mixed precision on a GPU. The
    seed is the Hugging Face default. Writes the files listed in the module docstring under output_dir
    and returns the dict saved as training_results.json.

    Raises MissingDependencyError without the [train] extra, and DataNotFoundError when train.jsonl or
    val.jsonl is missing.
    """
    _import_train_extra()
    import torch
    from transformers import (
        DistilBertForSequenceClassification,
        DistilBertTokenizer,
        EarlyStoppingCallback,
        TrainingArguments,
    )

    log.info("%s\nDISTILBERT FINE-TUNING FOR COMPLEXITY CLASSIFICATION\n%s", _RULE, _RULE)

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"
    for path in (train_path, val_path):
        if not path.exists():
            raise DataNotFoundError(f"Training data not found at {path}; run `pickspin classifier labels` first")

    # Only reported: the Trainer chooses the device itself.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("Using device: %s", device)
    if device == "cuda":
        log.info("GPU: %s", torch.cuda.get_device_name(0))

    log.info("\nLoading DistilBERT...")
    tokenizer = DistilBertTokenizer.from_pretrained(base_model)
    model = DistilBertForSequenceClassification.from_pretrained(
        base_model, num_labels=3, id2label=ID2LABEL, label2id=LABEL2ID
    )

    log.info("Loading datasets...")
    train_dataset = _load_dataset(train_path)
    val_dataset = _load_dataset(val_path)

    log.info("Training samples: %d", len(train_dataset))
    log.info("Validation samples: %d", len(val_dataset))

    # Inverse-frequency class weights for the imbalanced labels
    labels = [item["label"] for item in train_dataset]
    label_counts = Counter(labels)
    total = len(labels)
    num_classes = len(LABEL2ID)
    class_weights = torch.tensor(
        [total / (num_classes * label_counts[i]) for i in range(num_classes)], dtype=torch.float32
    )

    log.info("\nClass distribution: %s", dict(label_counts))
    log.info("Class weights: SIMPLE=%.2f, MEDIUM=%.2f, COMPLEX=%.2f", *class_weights.tolist())

    log.info("Tokenizing...")
    train_dataset = train_dataset.map(lambda x: _tokenize(x, tokenizer), batched=True, remove_columns=["text"])
    val_dataset = val_dataset.map(lambda x: _tokenize(x, tokenizer), batched=True, remove_columns=["text"])

    train_dataset.set_format("torch")
    val_dataset.set_format("torch")

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.1,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",  # weighs the three classes equally
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # mixed precision for speed
        dataloader_num_workers=4,
        report_to="none",
    )

    trainer = _weighted_trainer_class()(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    log.info("\n%s\nSTARTING TRAINING\n%s", _RULE, _RULE)
    start_time = datetime.now()

    trainer.train()

    training_time = datetime.now() - start_time
    log.info("\nTraining completed in: %s", training_time)

    log.info("\n%s\nEVALUATION\n%s", _RULE, _RULE)
    eval_results = trainer.evaluate()
    log.info("Validation Accuracy: %.4f", eval_results["eval_accuracy"])
    log.info("Validation Macro F1: %.4f", eval_results["eval_macro_f1"])

    log.info("\nGenerating results figure...")
    report = plot_training_results(trainer, output_dir, val_dataset)

    log.info("\nSaving model...")
    model_save_path = output_dir / "distilbert-complexity-classifier"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    results = {
        "training_time": str(training_time),
        "validation_accuracy": eval_results["eval_accuracy"],
        "validation_macro_f1": eval_results["eval_macro_f1"],
        "classification_report": report,
        "training_samples": len(train_dataset),
        "validation_samples": len(val_dataset),
        "model_name": base_model,
        "epochs": training_args.num_train_epochs,
        "batch_size": training_args.per_device_train_batch_size,
        "learning_rate": training_args.learning_rate,
        "class_weights": class_weights.tolist(),
        "weighted_loss": True,
    }

    # ASCII-only JSON, written in text mode (platform line ends) without a trailing newline.
    results_path = output_dir / "training_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    log.info(
        "\nSaved:\n  - Model: %s\n  - Results: %s\n  - Figure: %s",
        model_save_path,
        results_path,
        output_dir / "training_results.png",
    )
    log.info("\n%s\nTRAINING COMPLETE!\n%s", _RULE, _RULE)
    return results
