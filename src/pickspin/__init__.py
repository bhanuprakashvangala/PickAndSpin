"""Pick and Spin: cold-start-aware routing for self-hosted LLM serving.

The package follows the paper:

- pickspin.config: the model pool, tiers and parameters (Sec. VI-A, Eqs. 2-5)
- pickspin.pick: complexity classification and Thompson-sampling routing (Sec. IV, Eqs. 1-4)
- pickspin.spin: the COLD/LOADING/WARM lifecycle and the cold-start model (Sec. V, Eq. 5)
- pickspin.simulation: the trace-driven simulator
- pickspin.live: live runs on a Kubernetes deployment
- pickspin.baseline: the static baseline and the LLM judge
- pickspin.training: the DistilBERT training pipeline
- pickspin.paper: the reproduction of the paper's tables and figures (Sec. VII)
- pickspin.cli: the `pickspin` command

Importing pickspin imports nothing else, so it never loads an optional dependency.
"""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__: str = version("pick-and-spin")
except PackageNotFoundError:
    __version__ = "0+unknown"
