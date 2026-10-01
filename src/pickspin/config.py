"""Pick and Spin configuration (Sections IV-VI of the paper).

Model pool and tiers, the routing parameters of Eqs. 2-4, the cold-start model of Eq. 5
and Spin's cooldown. Endpoints are not stored here: they are read from a JSON file
(see deploy/endpoints.example.json).
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Nine models from three families in three tiers (Sec. VI-A). weight_gb is the size of the
# bf16 weights that a cold start loads from storage (Eq. 5); cold_start_s is the cold-start
# time stated for each model (Sec. V-B and Table II: 32-38 s small, 42-48 s medium,
# 65-95 s large).
MODELS = {
    "llama3.2_1B":  {"label": "Llama-3.2-1B", "tier": "SIMPLE",  "hf_id": "meta-llama/Llama-3.2-1B-Instruct",
                     "weight_gb": 2,  "cold_start_s": 32, "gpus": 1},
    "qwen2.5_1.5B": {"label": "Qwen2.5-1.5B", "tier": "SIMPLE",  "hf_id": "Qwen/Qwen2.5-1.5B-Instruct",
                     "weight_gb": 3,  "cold_start_s": 35, "gpus": 1},
    "gemma2_2B":    {"label": "Gemma-2-2B",   "tier": "SIMPLE",  "hf_id": "google/gemma-2-2b-it",
                     "weight_gb": 4,  "cold_start_s": 38, "gpus": 1},
    "llama3.2_3B":  {"label": "Llama-3.2-3B", "tier": "SIMPLE",  "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
                     "weight_gb": 6,  "cold_start_s": 32, "gpus": 1},
    "qwen2.5_7B":   {"label": "Qwen2.5-7B",   "tier": "MEDIUM",  "hf_id": "Qwen/Qwen2.5-7B-Instruct",
                     "weight_gb": 14, "cold_start_s": 48, "gpus": 1},
    "llama3.1_8B":  {"label": "Llama-3.1-8B", "tier": "MEDIUM",  "hf_id": "meta-llama/Llama-3.1-8B-Instruct",
                     "weight_gb": 16, "cold_start_s": 42, "gpus": 1},
    "gemma2_9B":    {"label": "Gemma-2-9B",   "tier": "MEDIUM",  "hf_id": "google/gemma-2-9b-it",
                     "weight_gb": 18, "cold_start_s": 48, "gpus": 1},
    "qwen2.5_14B":  {"label": "Qwen2.5-14B",  "tier": "COMPLEX", "hf_id": "Qwen/Qwen2.5-14B-Instruct",
                     "weight_gb": 28, "cold_start_s": 65, "gpus": 1},
    "gemma3_27B":   {"label": "Gemma-3-27B",  "tier": "COMPLEX", "hf_id": "google/gemma-3-27b-it",
                     "weight_gb": 54, "cold_start_s": 95, "gpus": 1},
}

TIER_ORDER = ["SIMPLE", "MEDIUM", "COMPLEX"]
TIERS = {t: [m for m, c in MODELS.items() if c["tier"] == t] for t in TIER_ORDER}

ROUTING = {
    "alpha_prior": 1.0,          # Beta(1, 1) prior for every model and tier (Eq. 2)
    "beta_prior": 1.0,
    "tier_weight": 0.3,          # w in Eq. 3
    "latency_weight": 0.3,       # lambda in Eq. 4
    "exploration_bonus": 0.1,    # epsilon in Eq. 4
}

SPIN = {
    "cooldown_s": 300,           # T_cooldown: a WARM model idle this long is scaled to zero
    "storage_gbps": 1.2,         # aggregate bandwidth of the shared weight volume (Eq. 5)
}

# Stage 1 of the hybrid classifier (Sec. IV-A): three keyword lists matched as lowercase
# substrings, COMPLEX first, then MEDIUM, then SIMPLE. Queries with no match go to DistilBERT.
KEYWORDS = {
    "SIMPLE": [
        "what is", "define", "who is", "when was", "where is", "true or false", "which of",
        "select the", "name the", "list the", "is it true", "yes or no",
    ],
    "MEDIUM": [
        "calculate", "how many", "how much", "solve", "compute", "write a function",
        "write a python function", "find the value", "what will be", "complete the",
    ],
    "COMPLEX": [
        "prove", "derive", "implement step by step", "analyze", "explain why",
        "compare and contrast", "design", "evaluate", "synthesize", "critique", "justify",
        "hypothesize", "formulate",
    ],
}

# Stage 2 of the hybrid classifier: the fine-tuned DistilBERT written by
# src/classifier/train_distilbert.py. Override with $PS_CLASSIFIER.
CLASSIFIER_DIR = Path(os.environ.get("PS_CLASSIFIER") or ROOT / "models" / "distilbert-complexity-classifier")
CLASSIFIER_MAX_LENGTH = 256

QUERIES_FILE = ROOT / "data" / "queries.jsonl.gz"
TRACES_DIR = ROOT / "results" / "traces"


def load_endpoints(path=None):
    """Return {model_key: {"base_url": ..., "model": served_model_name, "deployment": ...}}.

    Path comes from the argument, then $PS_ENDPOINTS, then the example file.
    """
    path = Path(path or os.environ.get("PS_ENDPOINTS") or ROOT / "deploy" / "endpoints.example.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
