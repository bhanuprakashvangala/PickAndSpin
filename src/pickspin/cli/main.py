"""The `pickspin` command: argument parsing, logging, dispatch and exit codes.

    pickspin [--root DIR] [-v | -q] [--version] <command> ...

The global options come before the command. --root (default: the current directory) is the directory
that holds data/, results/, models/ and deploy/; every default path is resolved under it. Logging goes
to stderr and stdout carries only command results. Exit codes: 0 on success; 1 for a PickSpinError (a
missing input file or extra, or bad configuration) or a reproduction mismatch; 2 for usage errors or a
missing command; 130 on Ctrl-C.
"""

import argparse
import logging
import sys
import types
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Final

from pickspin import __version__
from pickspin.cli import baseline, classifier, live, reproduce, simulate
from pickspin.errors import PickSpinError

log = logging.getLogger(__name__)

# The command modules, in the order `pickspin --help` lists them. Each one has register(subparsers),
# which adds its parser and sets func to the handler that runs it.
COMMANDS: Final[tuple[types.ModuleType, ...]] = (reproduce, simulate, live, baseline, classifier)

# The logger configure_logging sets up; every module of the package logs to a child of it.
_LOGGER: Final = "pickspin"
# The name of the handler configure_logging installs, so that a later call can find and replace it.
_HANDLER: Final = "pickspin.cli"

# The help texts are wrapped by hand to fit an 80-column terminal.
_DESCRIPTION: Final = """\
Pick and Spin: cold-start-aware routing for self-hosted LLM serving.

Reproduce the paper's tables and figures, simulate the deployment policies on
the released traces, run live experiments on Kubernetes, run the static
baseline and its LLM judge, and train the complexity classifier."""

_EPILOG: Final = """\
The global options come before the command; run 'pickspin <command> --help'
for the options of a command. Logging goes to stderr and stdout carries only
command results. Exit status: 0 on success, 1 for an error (such as a missing
input file or optional dependency) or a reproduction that does not match the
paper, 2 for a usage error, 130 on Ctrl-C."""


def build_parser() -> argparse.ArgumentParser:
    """Return the parser with the global options and every command's subcommand."""
    parser = argparse.ArgumentParser(
        prog="pickspin",
        description=_DESCRIPTION,
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(),
        metavar="DIR",
        help="the directory that holds data/, results/, models/ and deploy/; every default path is under it, "
        "while paths given to a flag are used as typed (default: the current directory)",
    )
    verbosity = parser.add_mutually_exclusive_group()
    verbosity.add_argument(
        "-v", "--verbose", action="count", default=0, help="also log debug messages, with times and logger names"
    )
    verbosity.add_argument("-q", "--quiet", action="store_true", help="log only warnings and errors")
    parser.add_argument("--version", action="version", version=f"pickspin {__version__}")
    subparsers = parser.add_subparsers(title="commands", dest="command", metavar="<command>")
    for module in COMMANDS:
        module.register(subparsers)
    return parser


def configure_logging(verbosity: int) -> None:
    """Send the 'pickspin' logger to stderr: WARNING for -1, INFO for 0 and DEBUG for 1 or more.

    At INFO and WARNING only the message is shown; at DEBUG each line also has the time, the level and
    the logger's name. Records do not propagate to the root logger. A later call replaces the handler
    an earlier call installed and leaves any other handler alone, so repeated calls never stack up
    handlers.
    """
    if verbosity < 0:
        level, fmt = logging.WARNING, "%(message)s"
    elif verbosity == 0:
        level, fmt = logging.INFO, "%(message)s"
    else:
        level, fmt = logging.DEBUG, "%(asctime)s %(levelname)s %(name)s: %(message)s"
    logger = logging.getLogger(_LOGGER)
    # Close the old handler before naming the new one: closing a handler unregisters its name.
    for old in [h for h in logger.handlers if h.get_name() == _HANDLER]:
        logger.removeHandler(old)
        old.close()
    handler = logging.StreamHandler(sys.stderr)
    handler.set_name(_HANDLER)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command line and return the exit status.

    argv defaults to sys.argv[1:]. Without a command the help goes to stderr and the status is 2. A
    PickSpinError is reported as one line, 'pickspin: error: <message>', on stderr with status 1 (with
    -v the traceback is logged too), and Ctrl-C gives 130. Otherwise the status is the command's own.
    Usage errors, --help and --version exit from argparse with SystemExit (2, 0 and 0).
    """
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(-1 if args.quiet else args.verbose)
    handler: Callable[[argparse.Namespace], int] | None = getattr(args, "func", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    try:
        return handler(args)
    except PickSpinError as e:
        log.debug("pickspin %s failed", args.command, exc_info=True)
        print(f"pickspin: error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
