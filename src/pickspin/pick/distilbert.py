"""Stage 2 of the hybrid classifier: the fine-tuned DistilBERT (Sec. IV-A, Eq. 1).

torch and transformers come with the [classifier] extra and are imported only when a DistilBertTier is
created, so importing this module is cheap. Queries of similar length are batched together so that
little time goes to padding.
"""

from __future__ import annotations

import logging
import types
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Final

from pickspin.config import Tier
from pickspin.errors import DataNotFoundError, import_optional

log = logging.getLogger(__name__)

# The most tokens of a query that DistilBERT reads; longer queries are truncated.
MAX_LENGTH: Final = 256


class DistilBertTier:
    """Predicts query tiers with the DistilBERT fine-tuned on the static-baseline labels.

    model_dir is a directory written by `pickspin classifier train`: the model's config.json, whose
    id2label names the tier of each class, its weights and its tokenizer.
    """

    device: str
    batch_size: int
    max_length: int
    tokenizer: Any
    model: Any
    id2label: dict[int, str]
    _torch: types.ModuleType

    def __init__(
        self,
        model_dir: Path,
        *,
        device: str | None = None,
        batch_size: int = 64,
        max_length: int = MAX_LENGTH,
    ) -> None:
        """Load the tokenizer and the model, in evaluation mode, on device (CUDA when available, else the CPU).

        Raises MissingDependencyError when torch or transformers is not installed, and then
        DataNotFoundError when model_dir has no config.json.
        """
        torch = import_optional("torch", "classifier")
        transformers = import_optional("transformers", "classifier")
        if not (model_dir / "config.json").exists():
            raise DataNotFoundError(
                f"No classifier at {model_dir}. Train one with `pickspin classifier train` "
                "or pass --model-dir / set $PS_CLASSIFIER."
            )
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir)
        self.model = transformers.AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.batch_size = batch_size
        self.max_length = max_length
        log.debug("Loaded the DistilBERT classifier from %s on %s", model_dir, self.device)

    def predict(self, queries: Sequence[str]) -> list[Tier]:
        """Return the predicted tier of every query, in input order.

        The queries are sorted by length in characters, keeping the input order among equal lengths,
        and sent in batches of batch_size, each padded to its longest query and truncated to
        max_length tokens. A query's tier is the class with the highest logit. The padding depends on
        the batch, so a query whose top two classes are almost tied can get a different tier when it
        is predicted as part of a different list.
        """
        order = sorted(range(len(queries)), key=lambda i: len(queries[i]))
        tiers: dict[int, Tier] = {}
        with self._torch.no_grad():
            for i in range(0, len(order), self.batch_size):
                idx = order[i : i + self.batch_size]
                batch = self.tokenizer(
                    [queries[j] for j in idx],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                )
                logits = self.model(**batch.to(self.device)).logits
                for j, k in zip(idx, logits.argmax(dim=-1).tolist()):
                    tiers[j] = Tier(self.id2label[int(k)])
        return [tiers[j] for j in range(len(queries))]
