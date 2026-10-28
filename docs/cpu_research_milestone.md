# CPU GNSS research checkpoint: moving reflections and receiver initialization

This checkpoint collects the CPU simulation and receiver research developed since
`42be392`. It includes the preceding CPU signal, polarization, rain/wet-antenna,
acquisition, calibration and uncertainty work that the latest experiments depend
on. CUDA remains the default urban-simulator backend. Experimental path fitting
and new initialization strategies are not automatically selected in production.

## Latest result and its limits

In 800 ms synthetic, known-two-path observations at 49 dB-Hz, a receiver with
common carrier drift produced roughly 3 m range bias when relative Doppler
crossed zero. Adding independent echo drift alone did not eliminate that bias.
Broader initialization improved 17 of 32 cases; half-window frequency seeds
improved the remaining 15. Final median absolute errors were 0.636 m and 0.545 m
in the two crossing cells; the largest absolute error was 1.900 m.

These are retrospective diagnostics on the same observations, not held-out
validation or evidence of real-world positioning performance. The next milestone
is a frozen-procedure replication on fresh data, including compute cost and null
controls. No claim of novelty over published research is made.

- [Physically coupled moving echoes](cpu_moving_echo_pilot.md)
- [Echo acceleration](cpu_accelerating_echo_pilot.md)
- [Sign reversal](cpu_echo_sign_reversal.md)
- [Independent echo drift](cpu_relative_drift_pilot.md)
- [Broader search](cpu_relative_drift_search.md)
- [Half-window initialization](cpu_window_drift_search.md)

The small report/audit/figure snapshots are under `assets/cpu_research/`, with
SHA-256 hashes in its manifest. Bulk generated observations remain in ignored
`output/` storage. Historical diagnosis scripts need their preceding protocol and
observation files; this repository snapshot alone does not contain that full
artifact chain. The source and unit tests can be used without those outputs.

## Historical source integrity

Merge preparation made formatting and typing/compatibility changes to research
modules. The original worktree and its frozen experiment outputs were not changed.
Exact pre-preparation versions of changed modules are retained under
`experiments/frozen_cpu_sources/`, indexed by original/current SHA-256 in
`manifest.json`. These files are provenance records, not a second import path.
Do not rewrite old protocol hashes to make old experiments appear to have run
with edited sources. To replay old experiments, use an isolated checkout with the
matching archived versions and the original artifact chain; old absolute source
paths may also require the original workspace layout.

## Development checks

Use Python 3.12 for the research tests and install numpy, scipy, numba, pytest and
matplotlib. From the repository root:

```bash
PYTHONPATH=python:. python -m pytest -q tests/test_window_seed_relative_drift.py tests/test_relative_drift_paths.py tests/test_broad_relative_drift.py
python -m ruff check python/ --ignore=E501,F401
python -m pyright
bash scripts/ci/run_python_smoke.sh
```

The broader suite also includes native CUDA tests and historical source-hash
checks. Those requirements are separate from a successful CPU-only smoke run.
