"""The static baseline and the LLM judge.

static sends every query to every model, with no routing, and can resume an interrupted run. judge
labels the static baseline's responses CORRECT or INCORRECT with an LLM behind an OpenAI-compatible
API. The released traces in results/traces/ were built from their outputs. Import from the modules.
"""
