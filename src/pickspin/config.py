"""Pick and Spin configuration.

Model pool, tiers, routing parameters and cold-start penalties as used for the
routed run in results/traces/pick_spin_routed.csv.gz. Endpoints are not stored
here: they are read from a JSON file (see deploy/endpoints.example.json).
"""

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# 9 models from 3 families. The routed run served the 27B model as
# google/gemma-2-27b-it under the key "gemma2_27B".
MODELS = {
    "llama3.2_1B":  {"family": "Llama", "size": "1B",   "tier": "SIMPLE",  "hf_id": "meta-llama/Llama-3.2-1B-Instruct"},
    "qwen2.5_1.5B": {"family": "Qwen",  "size": "1.5B", "tier": "SIMPLE",  "hf_id": "Qwen/Qwen2.5-1.5B-Instruct"},
    "gemma2_2B":    {"family": "Gemma", "size": "2B",   "tier": "SIMPLE",  "hf_id": "google/gemma-2-2b-it"},
    "llama3.2_3B":  {"family": "Llama", "size": "3B",   "tier": "SIMPLE",  "hf_id": "meta-llama/Llama-3.2-3B-Instruct"},
    "qwen2.5_7B":   {"family": "Qwen",  "size": "7B",   "tier": "MEDIUM",  "hf_id": "Qwen/Qwen2.5-7B-Instruct"},
    "llama3.1_8B":  {"family": "Llama", "size": "8B",   "tier": "MEDIUM",  "hf_id": "meta-llama/Llama-3.1-8B-Instruct"},
    "gemma2_9B":    {"family": "Gemma", "size": "9B",   "tier": "MEDIUM",  "hf_id": "google/gemma-2-9b-it"},
    "qwen2.5_14B":  {"family": "Qwen",  "size": "14B",  "tier": "MEDIUM",  "hf_id": "Qwen/Qwen2.5-14B-Instruct"},
    "gemma2_27B":   {"family": "Gemma", "size": "27B",  "tier": "COMPLEX", "hf_id": "google/gemma-2-27b-it"},
}

# Routing tiers used by Pick in the routed run.
TIERS = {
    "SIMPLE":  ["llama3.2_1B", "qwen2.5_1.5B", "gemma2_2B", "llama3.2_3B"],
    "MEDIUM":  ["qwen2.5_7B", "llama3.1_8B", "gemma2_9B", "qwen2.5_14B"],
    "COMPLEX": ["gemma2_27B"],
}

# Cold-start penalty (seconds) that Spin adds when a query hits a COLD model.
# These are the values used in the routed run.
COLD_START_TIMES = {
    "llama3.2_1B": 12, "qwen2.5_1.5B": 15, "gemma2_2B": 18, "llama3.2_3B": 22,
    "qwen2.5_7B": 38, "llama3.1_8B": 42, "gemma2_9B": 48, "qwen2.5_14B": 65,
    "gemma2_27B": 95,
}

ROUTING = {
    "alpha_prior": 1.0,
    "beta_prior": 1.0,
    "tier_weight": 0.3,          # w in Eq. 3
    "latency_weight": 0.3,       # lambda in Eq. 4
    "exploration_bonus": 0.1,    # epsilon in Eq. 4
    "confidence_threshold": 0.6,
    "max_escalations": 2,
    "cooldown_seconds": 300,     # T_cooldown
}

QUERIES_FILE = ROOT / "data" / "queries.jsonl.gz"


def load_endpoints(path=None):
    """Return {model_key: {"base_url": ..., "model": served_model_name}}.

    Path comes from the argument, then $PS_ENDPOINTS, then the example file.
    If $VLLM_API_KEY is set it is sent as a bearer token.
    """
    path = Path(path or os.environ.get("PS_ENDPOINTS") or ROOT / "deploy" / "endpoints.example.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)
