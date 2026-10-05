"""Stage 2 of the classifier, DistilBertTier: its optional dependencies, model loading and batching.

Most tests swap torch and transformers for small fakes that record every call, so they run without the
[classifier] extra and pin how queries are batched, tokenized and mapped back to their tiers.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import types
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pickspin.config import Tier
from pickspin.errors import DataNotFoundError, MissingDependencyError
from pickspin.pick.classifier import HybridClassifier, Stage
from pickspin.pick.distilbert import MAX_LENGTH, DistilBertTier

# The id2label of the trained model's config.json.
ID2LABEL = {"0": "SIMPLE", "1": "MEDIUM", "2": "COMPLEX"}


class FakeLibraries:
    """Fake torch and transformers modules that log, in order, the calls DistilBertTier makes.

    The fake model predicts, for each text, the class whose id is the text's last character.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.cuda_available = False
        self.grad_enabled = True
        self.torch = types.ModuleType("torch")
        self.torch.cuda = types.SimpleNamespace(is_available=lambda: self.cuda_available)  # type: ignore[attr-defined]
        self.torch.no_grad = self.no_grad  # type: ignore[attr-defined]
        self.transformers = types.ModuleType("transformers")
        self.transformers.AutoTokenizer = types.SimpleNamespace(  # type: ignore[attr-defined]
            from_pretrained=self.load_tokenizer
        )
        self.transformers.AutoModelForSequenceClassification = types.SimpleNamespace(  # type: ignore[attr-defined]
            from_pretrained=self.load_model
        )

    @contextlib.contextmanager
    def no_grad(self) -> Iterator[None]:
        self.grad_enabled = False
        try:
            yield
        finally:
            self.grad_enabled = True

    def load_tokenizer(self, path: Path) -> FakeTokenizer:
        self.calls.append(("AutoTokenizer.from_pretrained", path))
        return FakeTokenizer(self)

    def load_model(self, path: Path) -> FakeModel:
        self.calls.append(("AutoModelForSequenceClassification.from_pretrained", path))
        return FakeModel(self)


class FakeBatch(dict[str, Any]):
    """A tokenized batch: the fake model reads the texts back from input_ids."""

    def __init__(self, libs: FakeLibraries, texts: list[str]) -> None:
        super().__init__(input_ids=texts)
        self.libs = libs

    def to(self, device: str) -> FakeBatch:
        self.libs.calls.append(("batch.to", device))
        return self


class FakeTokenizer:
    def __init__(self, libs: FakeLibraries) -> None:
        self.libs = libs

    def __call__(self, texts: list[str], **options: Any) -> FakeBatch:
        self.libs.calls.append(("tokenize", list(texts), options))
        return FakeBatch(self.libs, list(texts))


class FakeLogits:
    def __init__(self, ids: list[int]) -> None:
        self.ids = ids

    def argmax(self, dim: int) -> types.SimpleNamespace:
        assert dim == -1
        return types.SimpleNamespace(tolist=lambda: list(self.ids))


class FakeModel:
    def __init__(self, libs: FakeLibraries) -> None:
        self.libs = libs
        self.config = types.SimpleNamespace(id2label=dict(ID2LABEL))

    def to(self, device: str) -> FakeModel:
        self.libs.calls.append(("model.to", device))
        return self

    def eval(self) -> FakeModel:
        self.libs.calls.append(("model.eval",))
        return self

    def __call__(self, *, input_ids: list[str]) -> types.SimpleNamespace:
        self.libs.calls.append(("forward", list(input_ids), self.libs.grad_enabled))
        return types.SimpleNamespace(logits=FakeLogits([int(text[-1]) for text in input_ids]))


@pytest.fixture
def fakes(monkeypatch: pytest.MonkeyPatch) -> FakeLibraries:
    """Installs fake torch and transformers modules for the duration of a test."""
    libs = FakeLibraries()
    monkeypatch.setitem(sys.modules, "torch", libs.torch)
    monkeypatch.setitem(sys.modules, "transformers", libs.transformers)
    return libs


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    """A directory with the config.json that DistilBertTier looks for."""
    (tmp_path / "config.json").write_text(json.dumps({"id2label": ID2LABEL}), encoding="utf-8")
    return tmp_path


def tokenized(batches: list[list[str]], *, device: str = "cpu", max_length: int = 256) -> list[tuple[Any, ...]]:
    """The calls predict() makes for these batches: tokenize, move to the device, run without gradients."""
    options = {"truncation": True, "max_length": max_length, "padding": True, "return_tensors": "pt"}
    calls: list[tuple[Any, ...]] = []
    for texts in batches:
        calls += [("tokenize", texts, options), ("batch.to", device), ("forward", texts, False)]
    return calls


def test_without_torch_the_error_names_the_classifier_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setitem(sys.modules, "torch", None)
    # tmp_path has no config.json either: the missing dependency is reported first.
    with pytest.raises(MissingDependencyError, match=r"pip install 'pick-and-spin\[classifier\]'") as excinfo:
        DistilBertTier(tmp_path)
    assert str(excinfo.value).startswith("torch is required")
    assert isinstance(excinfo.value, ImportError)
    with pytest.raises(MissingDependencyError, match=r"pick-and-spin\[classifier\]"):
        HybridClassifier.from_pretrained(tmp_path)


def test_without_transformers_the_error_names_the_classifier_extra(
    monkeypatch: pytest.MonkeyPatch, fakes: FakeLibraries, model_dir: Path
) -> None:
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(MissingDependencyError, match=r"^transformers is required .*pick-and-spin\[classifier\]"):
        DistilBertTier(model_dir)


def test_a_directory_without_a_model_is_reported(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    with pytest.raises(DataNotFoundError, match="No classifier at"):
        DistilBertTier(tmp_path)


def test_a_directory_without_config_json_is_reported_before_loading(fakes: FakeLibraries, tmp_path: Path) -> None:
    with pytest.raises(DataNotFoundError) as excinfo:
        DistilBertTier(tmp_path)
    assert str(excinfo.value) == (
        f"No classifier at {tmp_path}. Train one with `pickspin classifier train` "
        "or pass --model-dir / set $PS_CLASSIFIER."
    )
    assert isinstance(excinfo.value, FileNotFoundError)
    assert fakes.calls == []


@pytest.mark.parametrize(
    ("cuda_available", "device", "expected"),
    [(False, None, "cpu"), (True, None, "cuda"), (True, "cpu", "cpu"), (False, "cuda:1", "cuda:1")],
)
def test_the_model_loads_in_eval_mode_on_the_chosen_device(
    fakes: FakeLibraries, model_dir: Path, cuda_available: bool, device: str | None, expected: str
) -> None:
    fakes.cuda_available = cuda_available
    stage2 = DistilBertTier(model_dir, device=device)
    assert stage2.device == expected
    assert fakes.calls == [
        ("AutoTokenizer.from_pretrained", model_dir),
        ("AutoModelForSequenceClassification.from_pretrained", model_dir),
        ("model.to", expected),
        ("model.eval",),
    ]
    assert stage2.id2label == {0: "SIMPLE", 1: "MEDIUM", 2: "COMPLEX"}
    assert (stage2.batch_size, stage2.max_length) == (64, MAX_LENGTH)
    assert MAX_LENGTH == 256


def test_predict_batches_queries_by_length_and_keeps_the_input_order(fakes: FakeLibraries, model_dir: Path) -> None:
    stage2 = DistilBertTier(model_dir, batch_size=3)
    fakes.calls.clear()
    tiers = stage2.predict(["ccc2", "a0", "bb1", "dd2", "e1", "ffff0", "g2"])
    assert tiers == [Tier.COMPLEX, Tier.SIMPLE, Tier.MEDIUM, Tier.COMPLEX, Tier.MEDIUM, Tier.SIMPLE, Tier.COMPLEX]
    assert all(type(tier) is Tier for tier in tiers)
    # Sorted by length; queries of equal length keep their input order.
    assert fakes.calls == tokenized([["a0", "e1", "g2"], ["bb1", "dd2", "ccc2"], ["ffff0"]])
    assert fakes.grad_enabled


def test_predict_uses_the_device_and_max_length_it_was_given(fakes: FakeLibraries, model_dir: Path) -> None:
    stage2 = DistilBertTier(model_dir, device="cuda:1", batch_size=2, max_length=16)
    fakes.calls.clear()
    assert stage2.predict(["xx1", "y0"]) == [Tier.MEDIUM, Tier.SIMPLE]
    assert fakes.calls == tokenized([["y0", "xx1"]], device="cuda:1", max_length=16)


def test_predict_without_queries_runs_no_batch(fakes: FakeLibraries, model_dir: Path) -> None:
    stage2 = DistilBertTier(model_dir)
    fakes.calls.clear()
    assert stage2.predict([]) == []
    assert fakes.calls == []


def test_from_pretrained_sends_only_unmatched_queries_to_distilbert(fakes: FakeLibraries, model_dir: Path) -> None:
    clf = HybridClassifier.from_pretrained(model_dir, device="cpu", batch_size=2)
    assert isinstance(clf.stage2, DistilBertTier)
    fakes.calls.clear()
    assert clf.classify_many(["foo 2", "What is x? 0", "bar baz 1", "q 0"]) == [
        (Tier.COMPLEX, Stage.DISTILBERT),
        (Tier.SIMPLE, Stage.KEYWORD),
        (Tier.MEDIUM, Stage.DISTILBERT),
        (Tier.SIMPLE, Stage.DISTILBERT),
    ]
    assert fakes.calls == tokenized([["q 0", "foo 2"], ["bar baz 1"]])
    fakes.calls.clear()
    assert clf.classify("baz 2") == (Tier.COMPLEX, Stage.DISTILBERT)
    assert fakes.calls == tokenized([["baz 2"]])


def test_importing_stage2_does_not_import_torch_or_transformers() -> None:
    code = (
        "import sys\n"
        "sys.modules['torch'] = sys.modules['transformers'] = None  # importing either now fails\n"
        "import pickspin.pick.distilbert\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
