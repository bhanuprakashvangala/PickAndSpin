# Golden simulator fingerprints

These files pin the simulator to the code at tag `v1.1.0` (commit `094f537`). They hold SHA-256 digests only:

- `outputs.sha256`: every output file of the three configurations in `manifest.json` (`configs`), with per-query
  traces digested after decompression;
- `manifest.json`: the baseline commit, platform and Python version, and full-precision fingerprints of four runs
  (every field of every query record, the run summary and the per-model GPU accounting).

They were generated on Windows (`win32`) with CPython 3.12 by

```bash
git worktree add ../pick-and-spin-v1.1.0 v1.1.0
python scripts/make_goldens.py --baseline ../pick-and-spin-v1.1.0
```

`tests/integration/test_simulation_golden.py` (marked `slow`) reruns the configurations with the current code and
compares digests. Bit-exact results depend on the CPython minor version and the platform's math library, so on any
other platform or Python the test skips unless `PICKSPIN_GOLDEN_DIR` points at goldens regenerated there from
`v1.1.0` with the command above (CI does this). Never regenerate goldens from the code under test.
