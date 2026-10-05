"""The trace-driven simulator of Pick and Spin (standard library only).

Pick and Spin run unchanged on a simulated clock. A query routed to model m takes the latency and
success recorded for that (query, model) pair in results/traces/static_baseline.csv.gz, and its
correctness is the judge label in results/traces/judgments.csv.gz. Spin tracks COLD/LOADING/WARM per
model, cold starts share the storage bandwidth (Eq. 5), queries routed to a model that is not warm wait
for its load (Eq. 6), and a model with nothing in flight for T_cooldown is scaled to zero.

The modules are policies (the four policies compared), inputs (the recorded traces and the tier cache),
engine (the discrete-event loop), metrics (one run as CSV rows) and experiment (a policy x seed grid
and its output files). Import from the modules.
"""
