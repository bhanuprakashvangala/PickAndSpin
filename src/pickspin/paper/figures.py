"""Figs. 2a, 2b and 3 of the paper.

matplotlib is a base dependency but is imported only when the figures are drawn, with the Agg
backend. If it is missing, a warning is logged and the figures are skipped.

The figure sizes, colours, labels, text offsets and dpi are those of the published figures. The PNG
bytes also depend on the installed matplotlib build, which writes its version into the file.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pickspin.paper.analysis import ModelBaseline
from pickspin.paper.constants import DISPLAY_NAME, PAPER_MODELS, SIZE_CLASS

log = logging.getLogger(__name__)

SIZE_COLORS: Final[Mapping[str, str]] = MappingProxyType({"1-3B": "#4C72B0", "7-9B": "#DD8452", "14-27B": "#55A868"})


def make_figures(
    summary: Mapping[str, ModelBaseline], xs: Sequence[float], ys: Mapping[str, Sequence[float]], out_dir: Path
) -> bool:
    """Draw the three figures into out_dir; return False if matplotlib is not installed.

    Writes fig2a_throughput.png (throughput per model), fig2b_latency.png (latency per model) and
    fig3_thompson.png (the Fig. 3 selection rates ys over xs, one line per model in ys order). summary
    must hold every model of PAPER_MODELS, and every series in ys at least one point.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.warning("matplotlib not installed; skipping figures")
        return False
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [DISPLAY_NAME[m] for m in PAPER_MODELS]
    cols = [SIZE_COLORS[SIZE_CLASS[m]] for m in PAPER_MODELS]

    # Fig. 2a: horizontal bars, smallest model at the top.
    fig, ax = plt.subplots(figsize=(7, 4.5))
    vals = [summary[m].tok_s for m in PAPER_MODELS]
    ax.barh(names[::-1], vals[::-1], color=cols[::-1])
    for i, v in enumerate(vals[::-1]):
        ax.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=8)
    ax.set_xlabel("Throughput (tokens/second)")
    ax.set_title("Fig. 2a: throughput by model (static baseline)")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2a_throughput.png", dpi=200)
    plt.close(fig)

    # Fig. 2b: vertical bars in PAPER_MODELS order.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    vals = [summary[m].latency_s for m in PAPER_MODELS]
    ax.bar(names, vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.3, f"{v:.1f}s", ha="center", fontsize=8)
    ax.set_ylabel("Average latency (s)")
    ax.set_title("Fig. 2b: inference latency by model (static baseline)")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    fig.tight_layout()
    fig.savefig(out_dir / "fig2b_latency.png", dpi=200)
    plt.close(fig)

    # Fig. 3: cumulative selection rate of the tracked models; the legend shows each final rate.
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for m, series in ys.items():
        ax.plot(xs, series, label=f"{DISPLAY_NAME[m]} ({series[-1]:.1f}%)")
    ax.set_xlabel("Queries processed (thousands)")
    ax.set_ylabel("Cumulative selection rate (%)")
    ax.set_title("Fig. 3: Thompson Sampling selection over the routed run")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "fig3_thompson.png", dpi=200)
    plt.close(fig)
    return True
