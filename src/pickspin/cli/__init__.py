"""The `pickspin` command line.

main builds the argument parser from the command modules (reproduce, simulate, live, baseline and
classifier), configures logging once, dispatches to the command and turns errors into exit codes.
Command modules import nothing heavy at module level; each imports its implementation when it runs.
"""
