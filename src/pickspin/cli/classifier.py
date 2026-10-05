"""`pickspin classifier labels|train|evaluate`: the DistilBERT pipeline.

`classifier labels` writes the training labels (standard library only), `classifier train` fine-tunes
DistilBERT (the [train] extra), and `classifier evaluate` prints the accuracy of the classifier stages
on the validation split (the [classifier] extra). Without a subcommand the group prints its help.
"""

from __future__ import annotations

import argparse
import functools
import logging
import sys
from pathlib import Path
from typing import Final

from pickspin.errors import MissingDependencyError
from pickspin.paths import Paths, require_file, resolve_path

log = logging.getLogger(__name__)

_RULE: Final = "=" * 60

# The descriptions are wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
The offline pipeline behind stage 2 of Pick's classifier (Sec. IV-A). Run the
commands in this order: 'labels' derives complexity labels from the judge's
results, 'train' fine-tunes DistilBERT on them, and 'evaluate' measures the
keyword lists, DistilBERT and the hybrid classifier on the validation split."""

_LABELS_DESCRIPTION: Final = """\
Label every query with the smallest model group the judge marks correct:
SIMPLE (a 1-3B model), MEDIUM (a 7-14B model) or COMPLEX (only Gemma-3-27B,
Llama-3-70B or Kimi-K2, or no model at all). These groups are not the routing
tiers. The labelled queries are shuffled with a fixed seed and split 80/20
into <out>/train.jsonl and <out>/val.jsonl, with the label counts in
<out>/label_stats.json. Needs only the standard library."""

_TRAIN_DESCRIPTION: Final = """\
Fine-tune distilbert-base-uncased on <data>/train.jsonl with a class-weighted
loss, keeping the epoch with the best macro F1 on <data>/val.jsonl. Writes
the model to <out>/distilbert-complexity-classifier, the validation metrics
to <out>/training_results.json and .png, and a checkpoint per epoch to
<out>/checkpoint-*. A GPU is recommended. Needs the [train] extra."""

_EVALUATE_DESCRIPTION: Final = """\
Measure the accuracy of the keyword lists, DistilBERT and the hybrid
classifier on the validation split. Prints the figures and writes them to
--out as JSON. Needs the [classifier] extra and a trained model."""

_EVALUATE_EPILOG: Final = """\
environment:
  PS_CLASSIFIER  the DistilBERT directory when --model-dir is not given"""


def _print_help(parser: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """Print the group's help to stderr and return 2: the group was given without a subcommand."""
    parser.print_help(sys.stderr)
    return 2


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add the classifier command group with its labels, train and evaluate subcommands."""
    parser = subparsers.add_parser(
        "classifier",
        help="build the complexity labels, train and evaluate DistilBERT",
        description=_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.set_defaults(func=functools.partial(_print_help, parser))
    commands = parser.add_subparsers(title="commands", metavar="<command>")

    labels = commands.add_parser(
        "labels",
        help="derive complexity labels from the judge's results, split 80/20",
        description=_LABELS_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    labels.add_argument(
        "--judgments",
        type=Path,
        metavar="FILE",
        help="the judge's labels (default: <root>/results/traces/judgments.csv.gz)",
    )
    labels.add_argument(
        "--queries", type=Path, metavar="FILE", help="the benchmark queries (default: <root>/data/queries.jsonl.gz)"
    )
    labels.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/data/classifier)")
    labels.set_defaults(func=run_labels)

    train = commands.add_parser(
        "train",
        help="fine-tune DistilBERT on the labels",
        description=_TRAIN_DESCRIPTION,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train.add_argument(
        "--data",
        type=Path,
        metavar="DIR",
        help="where `pickspin classifier labels` wrote train.jsonl and val.jsonl (default: <root>/data/classifier)",
    )
    train.add_argument("--out", type=Path, metavar="DIR", help="output directory (default: <root>/models)")
    train.set_defaults(func=run_train)

    evaluate = commands.add_parser(
        "evaluate",
        help="measure the classifier's accuracy on the validation split",
        description=_EVALUATE_DESCRIPTION,
        epilog=_EVALUATE_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate.add_argument(
        "--val",
        type=Path,
        metavar="FILE",
        help="the validation split (default: <root>/data/classifier/val.jsonl)",
    )
    evaluate.add_argument(
        "--model-dir",
        type=Path,
        metavar="DIR",
        help="the fine-tuned DistilBERT (default: $PS_CLASSIFIER, else <root>/models/distilbert-complexity-classifier)",
    )
    evaluate.add_argument(
        "--out",
        type=Path,
        metavar="FILE",
        help="where the figures are written as JSON (default: <root>/results/classifier/evaluation.json)",
    )
    evaluate.set_defaults(func=run_evaluate)


def run_labels(args: argparse.Namespace) -> int:
    """Write the training labels and return the exit status."""
    paths = Paths(args.root)
    judgments = resolve_path(args.judgments, paths.traces / "judgments.csv.gz")
    queries = resolve_path(args.queries, paths.queries)
    out = resolve_path(args.out, paths.classifier_data)
    require_file(judgments, "judge-label trace")
    require_file(queries, "queries file")

    from pickspin.training.labels import generate_labeled_dataset

    log.info("%s\nGENERATING COMPLEXITY LABELS FOR DISTILBERT\n%s", _RULE, _RULE)
    generate_labeled_dataset(judgments, queries, out)
    return 0


def run_train(args: argparse.Namespace) -> int:
    """Fine-tune DistilBERT and return the exit status."""
    paths = Paths(args.root)
    data = resolve_path(args.data, paths.classifier_data)
    out = resolve_path(args.out, paths.models)
    require_file(data / "train.jsonl", "training data from `pickspin classifier labels`")
    require_file(data / "val.jsonl", "validation data from `pickspin classifier labels`")

    try:
        from pickspin.training import finetune
    except MissingDependencyError:
        raise  # it already names the missing module and the extra that provides it
    except ImportError as e:
        if e.name is not None and e.name.partition(".")[0] == "pickspin":
            raise  # a module of this package is missing: a bug or a broken install, not a missing extra
        missing = e.name or "a package of the [train] extra"
        raise MissingDependencyError(
            f"{missing} is required for this command: pip install 'pick-and-spin[train]'"
        ) from e

    finetune.train(data, out)
    return 0


def run_evaluate(args: argparse.Namespace) -> int:
    """Evaluate the classifier, print the results and return the exit status."""
    paths = Paths(args.root)
    val = resolve_path(args.val, paths.classifier_data / "val.jsonl")
    model_dir = resolve_path(args.model_dir, paths.classifier_model, env_var="PS_CLASSIFIER")
    out = resolve_path(args.out, paths.classifier_evaluation)
    require_file(val, "validation data from `pickspin classifier labels`")
    require_file(model_dir, "DistilBERT classifier")

    from pickspin.training.evaluate import evaluate, format_evaluation, write_evaluation

    result = evaluate(val, model_dir)
    write_evaluation(result, out)
    print(format_evaluation(result))
    return 0
