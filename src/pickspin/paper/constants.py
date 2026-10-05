"""Facts about the published experiment and the values as printed in the paper.

These are deliberately not derived from pickspin.config: they describe the recorded experiment, which
differs from the current configuration. In particular the routed run served Gemma-2-27B in the 27B
slot.
"""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# The nine model slots, smallest first. The static baseline ran Gemma-3-27B in the
# 27B slot ("gemma3_27B"); the routed run served google/gemma-2-27b-it there and
# logged it as "gemma2_27B". A routed query is scored with the static-baseline
# judge label of the model slot it was routed to.
PAPER_MODELS: Final[tuple[str, ...]] = (
    "llama3.2_1B",
    "qwen2.5_1.5B",
    "gemma2_2B",
    "llama3.2_3B",
    "qwen2.5_7B",
    "llama3.1_8B",
    "gemma2_9B",
    "qwen2.5_14B",
    "gemma3_27B",
)

DISPLAY_NAME: Final[Mapping[str, str]] = MappingProxyType(
    {
        "llama3.2_1B": "Llama-3.2-1B",
        "qwen2.5_1.5B": "Qwen2.5-1.5B",
        "gemma2_2B": "Gemma-2-2B",
        "llama3.2_3B": "Llama-3.2-3B",
        "qwen2.5_7B": "Qwen2.5-7B",
        "llama3.1_8B": "Llama-3.1-8B",
        "gemma2_9B": "Gemma-2-9B",
        "qwen2.5_14B": "Qwen2.5-14B",
        "gemma3_27B": "Gemma-3-27B",
    }
)

# Names for the routed run (Table I), where the 27B slot was served by Gemma-2-27B.
ROUTED_DISPLAY_NAME: Final[Mapping[str, str]] = MappingProxyType(dict(DISPLAY_NAME, gemma3_27B="Gemma-2-27B"))

SIZE_CLASS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "llama3.2_1B": "1-3B",
        "qwen2.5_1.5B": "1-3B",
        "gemma2_2B": "1-3B",
        "llama3.2_3B": "1-3B",
        "qwen2.5_7B": "7-9B",
        "llama3.1_8B": "7-9B",
        "gemma2_9B": "7-9B",
        "qwen2.5_14B": "14-27B",
        "gemma3_27B": "14-27B",
    }
)

BENCHMARK_COLUMNS: Final[tuple[str, ...]] = (
    "humaneval",
    "mbpp",
    "gsm8k",
    "math",
    "truthfulqa",
    "mmlu_pro",
    "arc",
    "hellaswag",
)

# The routed run logged the 27B slot as gemma2_27B; for scoring it maps to the static baseline's slot.
ROUTED_MODEL_ALIAS: Final[Mapping[str, str]] = MappingProxyType({"gemma2_27B": "gemma3_27B"})


def canonical_model(model: str) -> str:
    """Return the static-baseline model slot of a model key from the routed run.

    Only gemma2_27B changes (to gemma3_27B); every other key is returned as it is. This is used for
    scoring only: Fig. 3 counts the routed run's raw model keys.
    """
    return ROUTED_MODEL_ALIAS.get(model, model)


# The models whose cumulative selection rate Fig. 3 tracks, sampled every CONVERGENCE_STEP queries.
CONVERGENCE_MODELS: Final[tuple[str, ...]] = ("llama3.1_8B", "qwen2.5_7B", "llama3.2_1B", "gemma2_9B")
CONVERGENCE_STEP: Final = 500

# Table I as printed in the paper: queries, share of queries (%), accuracy (%) and latency (s) per model.
PAPER_TABLE1: Final[Mapping[str, tuple[int, float, float, float]]] = MappingProxyType(
    {
        "llama3.2_1B": (5319, 17.1, 23.2, 1.82),
        "qwen2.5_1.5B": (11, 0.04, 27.3, 4.22),
        "gemma2_2B": (7, 0.02, 28.6, 3.34),
        "llama3.2_3B": (5, 0.02, 20.0, 8.29),
        "qwen2.5_7B": (8101, 26.1, 63.3, 28.53),
        "llama3.1_8B": (13462, 43.4, 54.5, 28.87),
        "gemma2_9B": (499, 1.6, 73.5, 30.52),
        "qwen2.5_14B": (347, 1.1, 72.0, 44.63),
        "gemma3_27B": (633, 2.0, 75.2, 13.28),
    }
)

# Figs. 2a and 2b as printed in the paper, in PAPER_MODELS order.
PAPER_FIG2A_TOK_S: Final[tuple[str, ...]] = ("45.8", "26.4", "24.9", "16.3", "7.5", "10.6", "11.5", "5.6", "4.6")
PAPER_FIG2B_LATENCY_S: Final[tuple[str, ...]] = ("1.5", "1.5", "2.1", "2.2", "4.8", "4.3", "6.5", "7.6", "21.7")
