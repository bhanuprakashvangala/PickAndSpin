"""legacy/recorded_run is still byte for byte the code that recorded results/traces/pick_spin_routed.csv.gz.

Its four files must hash to the git blob ids of src/pickspin/ at tag v1.0.0, which legacy/README.md
lists. A blob id is the SHA-1 of 'blob <size>\\0' followed by the content as git stores it, with LF line
ends, so the working files are hashed after turning CRLF into LF: a Windows checkout holds them with
CRLF. The package must not depend on this code: no module under src/pickspin imports it, under its
directory names or its flat module names.
"""

import ast
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

import pytest

# The git blob ids of src/pickspin/*.py at tag v1.0.0 (commit 10e751f).
V1_0_0_BLOBS = {
    "config.py": "c5d30ba9d3a0ec8ffe6c1e510a08e242363e0444",
    "pick.py": "8a450a0bcf09f8c66bebe1bb72614bc81e9da143",
    "run_live.py": "7c0d93b136cd30b4e4ff40d45d75298dc39f5ce1",
    "spin.py": "b1ee62e18f7a6ebd28879b14494436027f10338d",
}
# What src/pickspin must never import: the legacy directories and the recorded runner's flat module names.
FORBIDDEN_IMPORTS = {"legacy", "recorded_run", "config", "pick", "spin", "run_live"}
# Calls that import a module named by a string.
IMPORT_CALLS = {"import_module", "import_optional", "__import__"}


@pytest.fixture(scope="module")
def recorded_run(repo_root: Path) -> Path:
    """legacy/recorded_run. It must exist in a git checkout; elsewhere, as in an sdist, the tests skip."""
    path = repo_root / "legacy" / "recorded_run"
    if not path.is_dir():
        if (repo_root / ".git").exists():
            pytest.fail(f"{path} is missing from this checkout")
        pytest.skip("legacy/ is not in this tree")
    return path


def git_blob_id(path: Path) -> str:
    """The id git gives the file's content once its line ends are LF."""
    data = path.read_bytes().replace(b"\r\n", b"\n")
    return hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest()


def test_recorded_runner_has_exactly_the_four_files(recorded_run):
    assert sorted(path.name for path in recorded_run.glob("*.py")) == sorted(V1_0_0_BLOBS)


@pytest.mark.parametrize("name", sorted(V1_0_0_BLOBS))
def test_recorded_runner_file_is_the_v1_0_0_blob(recorded_run, name):
    assert git_blob_id(recorded_run / name) == V1_0_0_BLOBS[name]


def test_readme_lists_the_v1_0_0_blob_ids_of_the_files(recorded_run):
    readme = (recorded_run.parent / "README.md").read_text(encoding="utf-8")
    # The README gives abbreviated ids, as `c5d30ba9` config.py.
    listed = {name: blob for blob, name in re.findall(r"`([0-9a-f]{7,40})`\s+([\w.]+\.py)", readme)}
    assert sorted(listed) == sorted(V1_0_0_BLOBS)
    for name, blob in listed.items():
        assert V1_0_0_BLOBS[name].startswith(blob), f"legacy/README.md lists {blob} for {name}"
        assert git_blob_id(recorded_run / name).startswith(blob), f"{name} is not the blob legacy/README.md lists"


def test_pinned_blob_ids_are_those_of_tag_v1_0_0(repo_root):
    """Checked against the tag itself where git and the tag are available (CI checkouts have no tags)."""
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    tagged = {}
    for name in V1_0_0_BLOBS:
        result = subprocess.run(
            [git, "rev-parse", "--verify", "--quiet", f"v1.0.0:src/pickspin/{name}"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            pytest.skip(f"git cannot read tag v1.0.0 here ({result.stderr.strip() or 'the tag is not in this clone'})")
        tagged[name] = result.stdout.strip()
    assert tagged == V1_0_0_BLOBS


def imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    """The absolute module names a module imports, with their line numbers.

    Relative imports name modules of the package itself and are left out. Calls such as
    importlib.import_module('x') with a literal name count as imports of x.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [(node.lineno, alias.name) for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Call) and node.args:
            func = node.func
            name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
            first = node.args[0]
            if name in IMPORT_CALLS and isinstance(first, ast.Constant) and isinstance(first.value, str):
                found.append((node.lineno, first.value))
    return found


def test_the_package_never_imports_the_recorded_runner(repo_root):
    package = repo_root / "src" / "pickspin"
    modules = sorted(package.rglob("*.py"))
    assert modules, f"no modules under {package}"
    offending = [
        f"{path.relative_to(repo_root).as_posix()}:{line}: {name}"
        for path in modules
        for line, name in imported_modules(ast.parse(path.read_bytes(), filename=str(path)))
        if name.split(".")[0] in FORBIDDEN_IMPORTS
    ]
    assert not offending
