"""Run the command line as a module: `python -m pickspin ...` is the same as `pickspin ...`."""

from pickspin.cli.main import main

if __name__ == "__main__":
    raise SystemExit(main())
