"""Errors that the command line reports as a one-line message, and the optional-dependency helper.

Library code never calls sys.exit or raises SystemExit. It raises a PickSpinError instead, and
pickspin.cli.main prints 'pickspin: error: <message>' to stderr and returns exit status 1. The
specific errors also derive from the matching built-in exceptions, so callers can still catch
FileNotFoundError, ImportError or ValueError.
"""

import importlib
import types


class PickSpinError(Exception):
    """Base class of the errors the command line turns into exit status 1."""


class DataNotFoundError(PickSpinError, FileNotFoundError):
    """A required input file or directory does not exist."""


class MissingDependencyError(PickSpinError, ImportError):
    """An optional dependency is not installed; the message names the extra that provides it."""


class ConfigError(PickSpinError, ValueError):
    """The configuration is invalid, for example a required environment variable is not set."""


def import_optional(module: str, extra: str) -> types.ModuleType:
    """Import and return an optional dependency, or explain which extra to install.

    module is the full module name (for example 'kubernetes.client') and extra the pick-and-spin
    extra that provides it (for example 'live').
    """
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingDependencyError(
            f"{module} is required for this command: pip install 'pick-and-spin[{extra}]'"
        ) from e
