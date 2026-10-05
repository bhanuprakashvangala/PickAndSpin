"""Fixtures shared by the unit and integration tests.

pickspin is imported inside the fixture bodies, never at module level, so that collecting the tests
does not depend on any one module of the package.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pickspin.simulation.inputs import TierAssignment, TraceData
    from pickspin.spin import Spin


@pytest.fixture(autouse=True)
def restore_pickspin_logger() -> Iterator[None]:
    """Put the 'pickspin' logger back as it was after every test.

    pickspin.cli.main.main() configures that logger for the whole process: it installs a handler on the
    sys.stderr of the moment, which pytest closes after a test that captures output, and turns
    propagation off, which hides later records from caplog.
    """
    logger = logging.getLogger("pickspin")
    handlers, level, propagate = logger.handlers[:], logger.level, logger.propagate
    yield
    for handler in logger.handlers:
        if handler not in handlers:
            handler.close()
    logger.handlers[:] = handlers
    logger.setLevel(level)
    logger.propagate = propagate


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """The root of the checkout that holds tests/."""
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def released_data(repo_root: Path) -> Path:
    """The repository root when the released data is present; skips the test otherwise (as in an sdist)."""
    if not (repo_root / "data" / "queries.jsonl.gz").exists():
        pytest.skip("the released data (data/queries.jsonl.gz) is not available")
    return repo_root


class FixedBeta(random.Random):
    """Every Beta sample is 0.5, so Eq. 4 depends only on latency and exploration."""

    def betavariate(self, alpha: float, beta: float) -> float:
        return 0.5


@pytest.fixture
def fixed_beta() -> type[FixedBeta]:
    """The FixedBeta class: a random.Random whose Beta samples are all 0.5. Call it to get an rng."""
    return FixedBeta


@pytest.fixture
def serve() -> Callable[[Spin, str, float, float, float], float]:
    """serve(spin, model, t_request, t_ready, infer_s) -> end time of one query.

    Routes one query to the model at t_request; a cold model is ready at t_ready, when the query
    starts and then runs for infer_s seconds.
    """
    from pickspin.spin import ModelState

    def _serve(spin: Spin, model: str, t_request: float, t_ready: float, infer_s: float) -> float:
        before = spin.request(model, t_request)
        if before == ModelState.COLD:
            spin.loaded(model, t_ready)
        spin.start(model, t_ready)
        spin.finish(model, t_ready + infer_s, infer_s, t_ready + infer_s - t_request)
        return t_ready + infer_s

    return _serve


@pytest.fixture
def make_tiny_trace() -> Callable[..., tuple[TraceData, dict[str, TierAssignment]]]:
    """make_tiny_trace(n=40, latency=2.0) -> (TraceData, tiers) for n synthetic queries.

    Every query succeeds on every model after `latency` seconds, the judge marks query i correct on
    every model when i is even, and every query is SIMPLE by keyword.
    """
    from pickspin.config import MODELS, Tier
    from pickspin.data import Query
    from pickspin.pick.classifier import Stage
    from pickspin.simulation.inputs import TraceData

    def _make(n: int = 40, latency: float = 2.0) -> tuple[TraceData, dict[str, TierAssignment]]:
        queries = [Query(f"q{i}", "b", "x", "", "") for i in range(n)]
        runs = {(q.id, m): (True, latency) for q in queries for m in MODELS}
        correct = {(q.id, m): i % 2 == 0 for i, q in enumerate(queries) for m in MODELS}
        tiers = {q.id: (Tier.SIMPLE, Stage.KEYWORD) for q in queries}
        return TraceData(queries=queries, runs=runs, correct=correct), tiers

    return _make
