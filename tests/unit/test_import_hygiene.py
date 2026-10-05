"""Importing pickspin never loads an optional or heavy dependency, and neither does `pickspin --help`.

The [live], [classifier] and [train] extras, matplotlib (needed only to draw figures) and requests
(needed only by the HTTP clients) are imported inside the functions that use them. This keeps the
base install working without the extras and keeps `pickspin --help` fast.

Each check runs in a fresh interpreter (sys.executable), because this test process has already
imported whatever the other tests needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Sequence
from typing import Any, Final

import pytest

# Modules that importing the core of the package or running `pickspin --help` must not load.
HEAVY: Final = ("torch", "transformers", "kubernetes", "matplotlib", "requests", "datasets", "sklearn", "seaborn")
# Modules that the HTTP clients and the classifier evaluation must not load (requests is allowed there).
EXTRAS: Final = ("torch", "transformers", "kubernetes", "matplotlib")

# Imports the module named by argv[1], then prints which of the modules named by argv[2:] are loaded.
IMPORT_PROBE: Final = """\
import importlib, json, sys
importlib.import_module(sys.argv[1])
print(json.dumps({"loaded": sorted(m for m in sys.argv[2:] if m in sys.modules)}))
"""

# Runs pickspin.cli.main.main(argv) with argv given as JSON in argv[1], then prints its exit status, what
# it wrote to stdout, and which of the modules named by argv[2:] are loaded.
MAIN_PROBE: Final = """\
import contextlib, io, json, sys
from pickspin.cli.main import main
stdout = io.StringIO()
with contextlib.redirect_stdout(stdout):
    try:
        status = main(json.loads(sys.argv[1]))
    except SystemExit as e:
        status = e.code
loaded = sorted(m for m in sys.argv[2:] if m in sys.modules)
print(json.dumps({"status": status, "stdout": stdout.getvalue(), "loaded": loaded}))
"""


def probe(code: str, argument: str, watched: Sequence[str]) -> dict[str, Any]:
    """Run code in a fresh interpreter with argument and the watched module names; return its report."""
    done = subprocess.run(
        [sys.executable, "-c", code, argument, *watched], capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    report: dict[str, Any] = json.loads(done.stdout.splitlines()[-1])
    return report


@pytest.mark.parametrize(
    "module",
    [
        "pickspin",
        "pickspin.cli.main",
        "pickspin.pick",
        "pickspin.spin",
        "pickspin.simulation.experiment",
        "pickspin.paper.reproduce",
        "pickspin.training.labels",
        # The [train] extra is imported when train() runs, so that a missing package is reported with the extra.
        "pickspin.training.finetune",
    ],
)
def test_the_core_modules_load_no_heavy_dependency(module: str) -> None:
    assert probe(IMPORT_PROBE, module, HEAVY) == {"loaded": []}


@pytest.mark.parametrize(
    "module",
    ["pickspin.live.runner", "pickspin.baseline.static", "pickspin.baseline.judge", "pickspin.training.evaluate"],
)
def test_the_http_clients_and_the_evaluation_load_no_extra(module: str) -> None:
    assert probe(IMPORT_PROBE, module, EXTRAS) == {"loaded": []}


@pytest.mark.parametrize(
    "argv",
    [["--help"], ["simulate", "--help"], ["live", "--help"], ["classifier", "train", "--help"]],
    ids=" ".join,
)
def test_help_loads_no_heavy_dependency(argv: list[str]) -> None:
    report = probe(MAIN_PROBE, json.dumps(argv), HEAVY)
    assert report["status"] == 0
    prog = " ".join(["pickspin", *argv[:-1]])
    assert report["stdout"].startswith(f"usage: {prog} ")
    assert report["loaded"] == []
