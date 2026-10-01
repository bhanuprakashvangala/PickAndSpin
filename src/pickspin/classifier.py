"""Hybrid complexity classifier for Pick (Sec. IV-A, Eq. 1).

Stage 1 matches three keyword lists (config.KEYWORDS) as lowercase substrings, checking the
COMPLEX list first, then MEDIUM, then SIMPLE. Queries that match no list fall through to
stage 2, the fine-tuned DistilBERT, which returns argmax_tau P(tau | q; theta).
"""

from config import CLASSIFIER_DIR, CLASSIFIER_MAX_LENGTH, KEYWORDS, TIER_ORDER


def keyword_tier(query):
    """Return the first tier (COMPLEX, MEDIUM, SIMPLE) whose list matches the query, else None."""
    q = query.lower()
    for tier in reversed(TIER_ORDER):
        if any(k in q for k in KEYWORDS[tier]):
            return tier
    return None


class DistilBertTier:
    """Stage 2: DistilBERT fine-tuned on the static-baseline labels (src/classifier)."""

    def __init__(self, model_dir=CLASSIFIER_DIR, device=None, batch_size=64):
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as e:
            raise SystemExit("DistilBERT needs torch and transformers: pip install -r requirements-classifier.txt") from e
        if not (model_dir / "config.json").exists():
            raise SystemExit(f"No classifier at {model_dir}. Train one with src/classifier/train_distilbert.py "
                             "or point $PS_CLASSIFIER at a trained model.")
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = batch_size

    def predict(self, queries):
        # Batch queries of similar length together so little time goes to padding.
        order = sorted(range(len(queries)), key=lambda i: len(queries[i]))
        tiers = [None] * len(queries)
        with self.torch.no_grad():
            for i in range(0, len(order), self.batch_size):
                idx = order[i:i + self.batch_size]
                batch = self.tokenizer([queries[j] for j in idx], truncation=True,
                                       max_length=CLASSIFIER_MAX_LENGTH, padding=True, return_tensors="pt")
                logits = self.model(**batch.to(self.device)).logits
                for j, k in zip(idx, logits.argmax(dim=-1).tolist()):
                    tiers[j] = self.id2label[int(k)]
        return tiers


class HybridClassifier:
    """classify(query) -> (tier, stage), where stage is "keyword" or "distilbert".

    With distilbert=None the ambiguous queries get `fallback_tier` instead (stage "default"),
    which is only meant for running without the trained model.
    """

    def __init__(self, distilbert="load", fallback_tier="MEDIUM"):
        self.distilbert = DistilBertTier() if distilbert == "load" else distilbert
        self.fallback_tier = fallback_tier

    def classify(self, query):
        tier = keyword_tier(query)
        if tier is not None:
            return tier, "keyword"
        if self.distilbert is None:
            return self.fallback_tier, "default"
        return self.distilbert.predict([query])[0], "distilbert"

    def classify_many(self, queries):
        """Batched version for offline use: one DistilBERT pass over all ambiguous queries."""
        out = [(keyword_tier(q), "keyword") for q in queries]
        todo = [i for i, (t, _) in enumerate(out) if t is None]
        if todo and self.distilbert is not None:
            for i, t in zip(todo, self.distilbert.predict([queries[i] for i in todo])):
                out[i] = (t, "distilbert")
        else:
            for i in todo:
                out[i] = (self.fallback_tier, "default")
        return out
