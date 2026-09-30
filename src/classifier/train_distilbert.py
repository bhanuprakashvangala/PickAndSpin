"""
Fine-tune DistilBERT for Query Complexity Classification (Pick Phase)

This script fine-tunes DistilBERT-base-uncased to classify queries into:
- SIMPLE: Can be answered by small models (1-3B)
- MEDIUM: Requires medium models (7-14B)
- COMPLEX: Requires large models (27B+)

Uses class-weighted loss to handle imbalanced dataset.
Optimized for fast training on GPU.
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset

# Label mapping
LABEL2ID = {'SIMPLE': 0, 'MEDIUM': 1, 'COMPLEX': 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


class WeightedTrainer(Trainer):
    """Custom Trainer with class-weighted loss for imbalanced data"""

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss_fct = nn.CrossEntropyLoss(weight=weight)
        else:
            loss_fct = nn.CrossEntropyLoss()

        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def load_dataset(filepath):
    """Load JSONL dataset"""
    data = []
    with open(filepath) as f:
        for line in f:
            item = json.loads(line)
            data.append({
                'text': item['text'],
                'label': LABEL2ID[item['label']]
            })
    return Dataset.from_list(data)

def tokenize_function(examples, tokenizer, max_length=256):
    """Tokenize text samples"""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=max_length
    )

def compute_metrics(eval_pred):
    """Compute accuracy and macro F1 for evaluation"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = (predictions == labels).mean()
    macro_f1 = f1_score(labels, predictions, average='macro')
    return {'accuracy': accuracy, 'macro_f1': macro_f1}

def plot_training_results(trainer, output_dir, val_dataset, tokenizer):
    """Generate training results figure"""

    # Get predictions on validation set
    predictions = trainer.predict(val_dataset)
    preds = np.argmax(predictions.predictions, axis=1)
    labels = predictions.label_ids

    # Classification report
    report = classification_report(
        labels, preds,
        target_names=['SIMPLE', 'MEDIUM', 'COMPLEX'],
        output_dict=True
    )

    # Confusion matrix
    cm = confusion_matrix(labels, preds)

    # Create figure with large fonts
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Confusion Matrix
    ax1 = axes[0]
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['SIMPLE', 'MEDIUM', 'COMPLEX'],
        yticklabels=['SIMPLE', 'MEDIUM', 'COMPLEX'],
        ax=ax1,
        annot_kws={'size': 18},
        cbar_kws={'shrink': 0.8}
    )
    ax1.set_xlabel('Predicted', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Actual', fontsize=18, fontweight='bold')
    ax1.set_title('Confusion Matrix', fontsize=20, fontweight='bold')

    # Plot 2: Per-class metrics
    ax2 = axes[1]
    classes = ['SIMPLE', 'MEDIUM', 'COMPLEX']
    x = np.arange(len(classes))
    width = 0.25

    precision = [report[c]['precision'] for c in classes]
    recall = [report[c]['recall'] for c in classes]
    f1 = [report[c]['f1-score'] for c in classes]

    bars1 = ax2.bar(x - width, precision, width, label='Precision', color='#2ecc71')
    bars2 = ax2.bar(x, recall, width, label='Recall', color='#3498db')
    bars3 = ax2.bar(x + width, f1, width, label='F1-Score', color='#9b59b6')

    ax2.set_ylabel('Score', fontsize=18, fontweight='bold')
    ax2.set_xlabel('Complexity Class', fontsize=18, fontweight='bold')
    ax2.set_title('Classification Metrics', fontsize=20, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(classes, fontsize=16)
    ax2.legend(loc='lower right', fontsize=14)
    ax2.set_ylim(0, 1.1)
    ax2.axhline(y=report['accuracy'], color='r', linestyle='--', linewidth=2, label=f'Accuracy: {report["accuracy"]:.1%}')

    # Add value labels on bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12)

    # Overall accuracy annotation
    ax2.text(0.02, 0.98, f'Overall Accuracy: {report["accuracy"]:.1%}',
            transform=ax2.transAxes, fontsize=16, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(output_dir, 'training_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nSaved figure: {fig_path}")

    return report

def main():
    """Main training function"""
    print("="*60)
    print("DISTILBERT FINE-TUNING FOR COMPLEXITY CLASSIFICATION")
    print("="*60)

    # Paths
    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data" / "classifier"   # written by generate_labels.py
    output_dir = root / "models"              # git-ignored

    train_path = data_dir / "train.jsonl"
    val_path = data_dir / "val.jsonl"

    # Check if data exists
    if not train_path.exists():
        print("ERROR: Training data not found. Run generate_labels.py first!")
        return

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load tokenizer and model
    print("\nLoading DistilBERT...")
    model_name = "distilbert-base-uncased"
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )

    # Load datasets
    print("Loading datasets...")
    train_dataset = load_dataset(train_path)
    val_dataset = load_dataset(val_path)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Calculate class weights for imbalanced data
    labels = [item['label'] for item in train_dataset]
    label_counts = Counter(labels)
    total = len(labels)
    num_classes = len(LABEL2ID)

    # Inverse frequency weighting
    class_weights = torch.tensor([
        total / (num_classes * label_counts[i])
        for i in range(num_classes)
    ], dtype=torch.float32)

    print(f"\nClass distribution: {dict(label_counts)}")
    print(f"Class weights: SIMPLE={class_weights[0]:.2f}, MEDIUM={class_weights[1]:.2f}, COMPLEX={class_weights[2]:.2f}")

    # Tokenize
    print("Tokenizing...")
    train_dataset = train_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )
    val_dataset = val_dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=['text']
    )

    train_dataset.set_format('torch')
    val_dataset.set_format('torch')

    # Training arguments - optimized for balanced performance
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
        metric_for_best_model="macro_f1",  # Use macro F1 for balanced classes
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Mixed precision for speed
        dataloader_num_workers=4,
        report_to="none"
    )

    # Weighted Trainer for class imbalance
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # Train
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    start_time = datetime.now()

    trainer.train()

    training_time = datetime.now() - start_time
    print(f"\nTraining completed in: {training_time}")

    # Evaluate
    print("\n" + "="*60)
    print("EVALUATION")
    print("="*60)

    eval_results = trainer.evaluate()
    print(f"Validation Accuracy: {eval_results['eval_accuracy']:.4f}")
    print(f"Validation Macro F1: {eval_results['eval_macro_f1']:.4f}")

    # Generate figure
    print("\nGenerating results figure...")
    report = plot_training_results(trainer, str(output_dir), val_dataset, tokenizer)

    # Save model
    print("\nSaving model...")
    model_save_path = output_dir / "distilbert-complexity-classifier"
    trainer.save_model(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))

    # Save results
    results = {
        'training_time': str(training_time),
        'validation_accuracy': eval_results['eval_accuracy'],
        'validation_macro_f1': eval_results['eval_macro_f1'],
        'classification_report': report,
        'training_samples': len(train_dataset),
        'validation_samples': len(val_dataset),
        'model_name': model_name,
        'epochs': training_args.num_train_epochs,
        'batch_size': training_args.per_device_train_batch_size,
        'learning_rate': training_args.learning_rate,
        'class_weights': class_weights.tolist(),
        'weighted_loss': True
    }

    results_path = output_dir / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved:")
    print(f"  - Model: {model_save_path}")
    print(f"  - Results: {results_path}")
    print(f"  - Figure: {output_dir / 'training_results.png'}")

    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()
