# GSDC2023 Post-PR55 Status - 2026-05-10

PR #55 is merged into `main`.

- PR: <https://github.com/rsasaki0109/gnss_gpu/pull/55>
- Merge commit: `bd63c08d5da3ed909b909f56d3c0383e5ee22cc6`
- Latest PR head before merge: `94995f5`
- CI at merge: `workflow-lint`, `lint`, `test-python-smoke`, `site-smoke`, and `build-cuda` passed.

## MATLAB Parity State

The MATLAB/reference final CSV is numerically reproduced by Python.

```bash
PYTHONPATH=.:python python3 experiments/reproduce_gsdc2023_matlab_reference_final.py \
  --output-dir experiments/results/source_selection_lowbaseline_submission_probe_20260430/matlab_reference_final_reproduction_require_exact_20260509 \
  --require-exact
```

Recorded real-data result:

```text
rows=71936
p95_delta_m=0
max_delta_m=0
missing_bridge_timestamp_rows=24
```

The reproduced final CSV matches the original MATLAB/reference leaderboard score: `4.056` public / `5.141` private.

## Python Best State

The current best private-floor Python submission family is not MATLAB-reference identical.

```text
Python private-floor best family: 3.686 public / 4.710 private
MATLAB/reference reproduced final: 4.056 public / 5.141 private
```

Interpretation:

- MATLAB final-output reproduction is solved for provenance/parity.
- The better Python Kaggle score is a separate leaderboard-optimization track.
- Do not use "MATLAB equivalent" to imply that the best Python leaderboard CSV is identical to the MATLAB/reference final CSV.

## Submit-Readiness Gates

Submit readiness can require both:

- `--require-matlab-equivalence`
- `--matlab-final-reproduction-summary .../matlab_reference_final_reproduction_require_exact_20260509/summary.json`
- `--require-matlab-final-reproduction`

The real `p6p0_prevsafe_candidate_20260508` dry run with both gates recorded:

```text
prepared=3 candidates
matlab_final_reproduction_gate.passed=true
rows=71936
max_delta_m=0
changed_rows_gt_1e_9m=0
changed_rows_gt_0p01m=0
```

## Next Reasonable Work

1. Small documentation cleanup PRs only if the public README/internal docs drift again.
2. If leaderboard work resumes, treat it as Kaggle optimization, not MATLAB parity.
3. If productionizing reproduction, reduce audit-derived GSDC scripts into a smaller command surface and retain `--require-exact` as the regression gate.
