"""Where the inputs and outputs live in the repository.

This is the only module that knows the repository layout. Every default location is relative to a
root directory, which defaults to the current directory; the command line sets it with --root.
Data never ships inside the package and nothing is located through __file__.

It also provides the two helpers the command line uses for paths: resolve_path, which picks an
explicit flag over an environment variable over the default, and require_file, which checks that an
input exists before any work starts.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from pickspin.errors import DataNotFoundError


@dataclass(frozen=True, slots=True)
class Paths:
    """The default input and output locations under a repository root."""

    root: Path = Path()  # the current directory

    @property
    def queries(self) -> Path:
        """The benchmark prompts (data/queries.jsonl.gz)."""
        return self.root / "data" / "queries.jsonl.gz"

    @property
    def tier_cache(self) -> Path:
        """The hybrid classifier's tier for every query (data/query_tiers.csv.gz)."""
        return self.root / "data" / "query_tiers.csv.gz"

    @property
    def classifier_data(self) -> Path:
        """The DistilBERT training and validation split (data/classifier)."""
        return self.root / "data" / "classifier"

    @property
    def traces(self) -> Path:
        """The released experiment traces (results/traces)."""
        return self.root / "results" / "traces"

    @property
    def results(self) -> Path:
        """The paper reproduction's tables and figures (results)."""
        return self.root / "results"

    @property
    def simulation(self) -> Path:
        """The simulator's outputs (results/simulation)."""
        return self.root / "results" / "simulation"

    @property
    def live(self) -> Path:
        """The live runner's outputs (results/live)."""
        return self.root / "results" / "live"

    @property
    def static_baseline(self) -> Path:
        """The static baseline's responses (results/live/static)."""
        return self.root / "results" / "live" / "static"

    @property
    def judgments(self) -> Path:
        """The LLM judge's labels (results/live/judgments)."""
        return self.root / "results" / "live" / "judgments"

    @property
    def classifier_evaluation(self) -> Path:
        """The classifier accuracy report (results/classifier/evaluation.json)."""
        return self.root / "results" / "classifier" / "evaluation.json"

    @property
    def models(self) -> Path:
        """Where training writes the model and its checkpoints (models)."""
        return self.root / "models"

    @property
    def classifier_model(self) -> Path:
        """The fine-tuned DistilBERT (models/distilbert-complexity-classifier)."""
        return self.root / "models" / "distilbert-complexity-classifier"

    @property
    def endpoints_example(self) -> Path:
        """The example endpoint map (deploy/endpoints.example.json)."""
        return self.root / "deploy" / "endpoints.example.json"


def resolve_path(flag: Path | None, default: Path, *, env_var: str | None = None) -> Path:
    """Return the path given on the command line, else the one in env_var, else the default.

    An environment variable that is set but empty counts as unset. Paths from a flag or the
    environment are used as given, so relative ones stay relative to the current directory.
    """
    if flag is not None:
        return flag
    if env_var is not None and (value := os.environ.get(env_var)):
        return Path(value)
    return default


def require_file(path: Path, what: str) -> Path:
    """Return path if it exists, else raise DataNotFoundError naming what is missing and where."""
    if not path.exists():
        raise DataNotFoundError(
            f"{what} not found: {path} (run from the repository root, pass --root DIR, or give the path explicitly)"
        )
    return path
