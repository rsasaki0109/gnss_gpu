# PR Summary

## Summary

- Add GSDC2023 pre-submit manifest generation and submit readiness gates.
- Add ready report, ready report audit, and one-shot prepare flow for P6P0 candidate submissions.
- Add factor mask, residual value, PR proxy risk, and VD residual audit tooling for MATLAB migration parity.
- Improve observation matrix, GNSS log residual parity, VD guard metrics, and chunk selection controls.
- Add candidate build/export tooling and release notes for the submit readiness artifacts.

## Submit State

- Kaggle submission was not run.
- P6P0 ready artifacts were prepared and audited.
- Worktree was clean after splitting the commits.

## Test Plan

```bash
PYTHONPATH=.:python pytest -q tests/test_build_gsdc2023_pre_submit_manifest.py tests/test_submit_gsdc2023_pixel5_candidate_queue.py tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_clock_state.py tests/test_diagnose_gsdc2023_epoch_errors.py
# 58 passed
```

Additional focused checks run while splitting commits:

```text
pre-submit manifest: 2 passed
submit readiness gates: 20 passed
observation matrix parity support: 53 passed
VD guard metrics: 122 passed
parity/risk audit tools: 15 passed
candidate build/export tools: 10 passed
GNSS log residual parity: 90 passed
chunk selection controls incl. real observation mask: 38 passed
diagnostic comparison helpers: 11 passed
```

## Key Artifacts

```text
internal_docs/gsdc2023_submit_readiness_release_20260505.md
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/submit_readiness.md
```
