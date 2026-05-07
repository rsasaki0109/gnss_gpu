# GSDC2023 Submit Readiness Release - 2026-05-05

## Commits

```text
f062584 Add GSDC2023 pre-submit manifest builder
7cb0dd3 Add GSDC2023 submit readiness gates
52469e3 Update GSDC2023 migration plan
84a1e59 Improve GSDC2023 observation matrix parity support
2c9a89b Add GSDC2023 VD guard metrics
7a36fc5 Add GSDC2023 parity and risk audit tools
0cb0bbf Add GSDC2023 candidate build tools
110f64b Improve GSDC2023 GNSS log residual parity
6465c6b Refine GSDC2023 chunk selection controls
892ec8b Update GSDC2023 diagnostic comparison helpers
```

## Scope

- MATLAB migration parity audit support for factor masks and residual values.
- Raw bridge VD seed guard metrics and rejection records.
- PR proxy risk reports and submit-time risk gates.
- P6P0 pre-submit manifest, ready report, ready audit, and readiness doc generation.
- Candidate build and source chunk export tools.
- GNSS log residual parity improvements.
- Chunk selection and reproduction controls.
- Focused tests for the above.

## Verification

```bash
PYTHONPATH=.:python pytest -q tests/test_build_gsdc2023_pre_submit_manifest.py tests/test_submit_gsdc2023_pixel5_candidate_queue.py tests/test_gsdc2023_chunk_selection.py tests/test_gsdc2023_clock_state.py tests/test_diagnose_gsdc2023_epoch_errors.py
# 58 passed
```

Additional group checks during commit split:

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

## Submit State

- Kaggle submit was not run.
- P6P0 previous-safe artifacts were prepared and audited, but duplicate-SHA guard identifies them as byte-identical to already-ready non-P6P0 candidates.
- Full-window MATLAB writer equivalence is cached and re-auditable from the submit-readiness doc.
- `phone_data` artifact compatibility is re-auditable from the submit-readiness doc; CSV sidecars are covered and `phone_data.mat` is intentionally deferred.
- `git status --short` was clean after the commit split.

## P6P0 Ready Artifacts

```text
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/pre_submit_manifest.json
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/pre_submit_candidate_manifest.csv
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/pre_submit_trip_delta_checks.csv
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/submit_ready_report.json
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/submit_ready_report.csv
experiments/results/source_selection_lowbaseline_submission_probe_20260430/p6p0_clean_candidate_20260505/submit_readiness.md
```

Current P6P0 gate state:

```text
ready_count=3
ready_csv_rows=3
pre_submit_manifest_candidates=3
candidate_actionable_risky_chunks=0
max_risky_pixel6pro_input_changed_rows=1444
max_risky_pixel6pro_input_delta_m=0.8
matlab_equivalence=matlab_equivalent
cached_matlab_equivalence_validation=passed
duplicate_sha_candidates=3
duplicate_sha_matches=3
phone_data_artifact_compatibility=passed
phone_data_mat_decision=defer
```
