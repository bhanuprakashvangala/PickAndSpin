"""The simulator must reproduce the frozen v1.1.0 simulator bit for bit.

tests/data/golden holds digests of what the v1.1.0 simulator produced, written by scripts/make_goldens.py:
outputs.sha256 lists every output file of the configurations in manifest.json, and manifest.json also
holds full-precision fingerprints of four runs. The first test reruns those configurations through
`pickspin simulate` with the current code and digests its outputs the same way. The second recomputes
the fingerprints with pickspin.simulation.engine.simulate. The CSVs are rounded, so they catch changes
in routing and accounting; the fingerprints also catch drift in the last bit of any time or total.

Bit-exact results depend on the CPython minor version (float sum() is compensated since 3.12) and on
the platform's math library (random.expovariate and betavariate), so the committed goldens are only
compared on the platform and Python recorded in their manifest, and the tests skip elsewhere. To run
them on another machine, generate goldens there from tag v1.1.0 and name their directory in
PICKSPIN_GOLDEN_DIR, as the CI golden job does:

    git worktree add ../pick-and-spin-v1.1.0 v1.1.0
    python scripts/make_goldens.py --baseline ../pick-and-spin-v1.1.0 --out <dir>
    PICKSPIN_GOLDEN_DIR=<dir> pytest -m slow tests/integration/test_simulation_golden.py

Never generate goldens from the code under test.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from pickspin.cli.main import main
from pickspin.config import MODELS
from pickspin.paths import Paths
from pickspin.simulation.engine import SimulationResult, SimulationSettings, simulate
from pickspin.simulation.inputs import load_trace_data, query_tiers
from pickspin.simulation.policies import Policy

pytestmark = pytest.mark.slow

GOLDEN_DIR_ENV = "PICKSPIN_GOLDEN_DIR"
MANIFEST = "manifest.json"
OUTPUT_DIGESTS = "outputs.sha256"


@dataclass(frozen=True)
class Golden:
    """A golden directory: its manifest and the digest of every output file by relative path."""

    path: Path
    manifest: dict[str, Any]
    outputs: dict[str, str]

    @property
    def provenance(self) -> str:
        """Where the goldens come from (commit, platform and Python), for messages."""
        m = self.manifest
        return f"{self.path}, generated from {m['baseline_commit'][:7]} on {m['platform']} / Python {m['python']}"


def read_output_digests(path: Path) -> dict[str, str]:
    """Read an outputs.sha256 file, one '<sha256>  <relative path>' line per output file."""
    digests: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, rel = line.split("  ", 1)
        assert rel not in digests, f"{path} lists {rel} twice"
        digests[rel] = digest
    return digests


def file_digest(path: Path) -> str:
    """SHA-256 of a file, taken after decompression for a .gz file (its gzip header holds a timestamp)."""
    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    return hashlib.sha256(data).hexdigest()


def output_digests(run_dir: Path, base: Path) -> dict[str, str]:
    """Digest every file under run_dir, keyed as scripts/make_goldens.py keys them.

    The key is the file's POSIX path relative to base, without a trailing '.gz'.
    """
    return {
        path.relative_to(base).as_posix().removesuffix(".gz"): file_digest(path)
        for path in sorted(run_dir.rglob("*"))
        if path.is_file()
    }


def fingerprint(result: SimulationResult) -> str:
    """The full-precision fingerprint of a run, in exactly the format of scripts/make_goldens.py.

    SHA-256 of UTF-8 lines, each followed by a newline: the repr of one tuple per query record in
    result order, then the repr of the run's totals and makespan, then one per model in MODELS order.
    repr() writes every float with all its digits, so a change in the last bit changes the fingerprint.
    """
    lines = [
        repr(
            (
                r.order,
                r.id,
                r.model,
                str(r.tier),
                str(r.stage),
                r.arrive,
                r.start,
                r.end,
                r.infer_s,
                r.wait_s,
                r.total_s,
                r.cold_start,
                r.success,
                r.correct,
                r.worker,
            )
        )
        for r in result.records
    ]
    usage = result.usage
    lines.append(
        repr(
            (
                usage.gpu_hours,
                usage.busy_gpu_hours,
                usage.gpu_utilization,
                usage.cold_starts,
                usage.cold_start_rate,
                result.makespan_s,
            )
        )
    )
    for m in MODELS:
        pm = usage.per_model[m]
        lines.append(repr((m, pm.gpu_hours, pm.busy_gpu_hours, pm.loading_gpu_hours, pm.cold_starts)))
    return hashlib.sha256("".join(line + "\n" for line in lines).encode("utf-8")).hexdigest()


def describe_differences(produced: Mapping[str, str], expected: Mapping[str, str], golden: Golden) -> str:
    """List the output files that are missing, unexpected or different, for an assertion message."""
    lines = [f"the outputs differ from the goldens ({golden.provenance}):"]
    lines += [f"  missing:    {rel}" for rel in sorted(expected.keys() - produced.keys())]
    lines += [f"  unexpected: {rel}" for rel in sorted(produced.keys() - expected.keys())]
    lines += [
        f"  different:  {rel}" for rel in sorted(expected.keys() & produced.keys()) if produced[rel] != expected[rel]
    ]
    return "\n".join(lines)


@pytest.fixture(scope="module")
def golden(repo_root: Path) -> Golden:
    """The goldens to compare with: the directory in $PICKSPIN_GOLDEN_DIR if set, else tests/data/golden.

    The committed goldens are only valid on the platform and Python recorded in their manifest, so the
    tests skip elsewhere. A directory named in $PICKSPIN_GOLDEN_DIR is taken to have been generated on
    this machine. Either way the directory must hold both files with something to compare, so a broken
    setup fails instead of skipping or passing vacuously.
    """
    override = os.environ.get(GOLDEN_DIR_ENV)
    path = Path(override) if override else repo_root / "tests" / "data" / "golden"
    missing = [name for name in (MANIFEST, OUTPUT_DIGESTS) if not (path / name).is_file()]
    if missing:
        pytest.fail(f"{path} has no {' or '.join(missing)}; generate goldens with scripts/make_goldens.py")
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    here = (sys.platform, f"{sys.version_info.major}.{sys.version_info.minor}")
    if not override and here != (manifest["platform"], manifest["python"]):
        pytest.skip(
            f"the committed goldens are for {manifest['platform']} / Python {manifest['python']}, not "
            f"{here[0]} / Python {here[1]}; set {GOLDEN_DIR_ENV} to goldens generated here from v1.1.0"
        )
    outputs = read_output_digests(path / OUTPUT_DIGESTS)
    if not (manifest["configs"] and manifest["fingerprints"] and outputs):
        pytest.fail(f"the goldens in {path} are empty; generate them with scripts/make_goldens.py")
    return Golden(path=path, manifest=manifest, outputs=outputs)


# main() configures the 'pickspin' logger for the whole process; the autouse fixture
# restore_pickspin_logger in tests/conftest.py puts it back after each test.


def test_simulate_command_reproduces_golden_outputs(golden: Golden, released_data: Path, tmp_path: Path) -> None:
    """`pickspin simulate` writes every golden output file, byte for byte, and nothing else."""
    produced: dict[str, str] = {}
    for name, args in golden.manifest["configs"].items():
        out = tmp_path / name
        status = main(["--root", str(released_data), "simulate", *args, "--out", str(out)])
        assert status == 0, f"pickspin simulate {' '.join(args)} exited with status {status}"
        produced |= output_digests(out, tmp_path)
    assert produced == golden.outputs, describe_differences(produced, golden.outputs, golden)


def test_simulator_reproduces_golden_fingerprints(golden: Golden, released_data: Path) -> None:
    """Every fingerprinted run gives the same records, totals and per-model usage, to the last bit."""
    paths = Paths(released_data)
    data = load_trace_data(paths.queries, paths.traces)
    tiers = query_tiers(data.queries, paths.tier_cache)
    mismatches = []
    for name, entry in golden.manifest["fingerprints"].items():
        # The manifest records the old simulate() keyword arguments, which are SimulationSettings' fields.
        settings = dict(entry["run"])
        policy, seed = Policy(settings.pop("policy")), settings.pop("seed")
        result = simulate(policy, seed, data, tiers, SimulationSettings(**settings))
        if (actual := fingerprint(result)) != entry["sha256"]:
            mismatches.append(f"  {name}: {actual}, expected {entry['sha256']}")
    assert not mismatches, f"fingerprints differ from the goldens ({golden.provenance}):\n" + "\n".join(mismatches)
