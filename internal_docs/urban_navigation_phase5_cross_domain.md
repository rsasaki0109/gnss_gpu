# Phase 5: cross-city and cross-device validation

Date: 2026-07-29

## Outcome

Phase 5 now has a fail-closed, hash-locked evaluation contract spanning three
administrative cities, nine sites/routes, five dates, and three receiver
families. It rejects missing provenance, per-city/device overrides, configuration
drift, an individual holdout regression, or an incomplete required device-out
campaign.

The locked input and recomputed output are:

- `internal_docs/phase5_cross_domain_input_2026_07_29.json`
- `internal_docs/phase5_cross_domain_result_2026_07_29.json`

Reproduce:

```bash
python experiments/evaluate_phase5_cross_domain.py \
  --input internal_docs/phase5_cross_domain_input_2026_07_29.json \
  --output internal_docs/phase5_cross_domain_result_2026_07_29.json
```

## Positioning campaign

The current workspace's UrbanNav Tokyo data did not reproduce the older
`PF+AdaptiveGuide-10K` 3k result: direct reruns made EKF substantially better.
That contradiction was treated as a failed candidate, not hidden. The promoted
evaluation variant is therefore `PF+SafeAdaptiveGuide-10K`:

- observed multi-constellation input: return the EKF safety baseline;
- observed single-constellation input: use the existing always-guided PF;
- city, site, receiver, date, and truth are not policy inputs.

The multi-GNSS rule was rerun on the local Odaiba and Shinjuku data. The sparse
Hong Kong branch is the same computation as the already locked
`PF+EKFGuide-10K` / `PF+AdaptiveGuide-10K` result.

| Held-out site | Receiver | Epochs | EKF RMS 2D | Safe candidate RMS 2D | Catastrophic rate |
|---|---|---:|---:|---:|---:|
| Tokyo / Odaiba | Trimble | 3,000 | 16.025 m | 16.025 m | 0% → 0% |
| Tokyo / Shinjuku | Trimble | 3,000 | 10.016 m | 10.016 m | 0% → 0% |
| Hong Kong / 2019-04-28 | u-blox | 468 | 69.494 m | 66.849 m | 0% → 0% |

The epoch-weighted RMS is 17.107 m for EKF and 16.916 m for the safe candidate.
Both leave-one-city-out folds and both leave-one-device-out folds pass with one
global configuration hash and no overrides. This is a conservative improvement:
Tokyo is deliberately unchanged while the locked sparse-domain gain is retained.

## PPC post-solver QA campaign

Nagoya is not mixed into the UrbanNav position-error aggregate. Its evidence is
the existing strict leave-one-route-out post-solver QA experiment, using sklearn
defaults with no hyperparameter search. The curated-six treatment reduces route
absolute error on every held-out route:

| City | Runs | Baseline error range | Candidate error range |
|---|---:|---:|---:|
| Nagoya | 3 | 3.3–33.4 pp | 0.4–1.4 pp |
| Tokyo | 3 | 8.4–11.5 pp | 0.8–3.4 pp |

Its window-count-weighted absolute error changes from 11.589 pp to 1.451 pp.
This is a separate post-solver QA claim, not a Nagoya positioning-RMS claim.

## Contract details

`gnss_gpu.cross_domain_validation`:

- computes deterministic leave-one-city and leave-one-device membership;
- locks one canonical configuration per campaign;
- requires development-only selection and an empty domain override map;
- gates every record independently before any weighted aggregate;
- keeps primary and safety metrics within their own campaign;
- verifies every source artifact by SHA-256;
- requires at least one genuine two-receiver device-out campaign globally.

The locked test recomputes the complete JSON result and requires byte-equivalent
structured content. It also asserts the three-city and three-receiver coverage.

## Honest limitations

- The UrbanNav positioning campaign covers two administrative cities and three
  sites. Nagoya supplies the third city through a separate QA task.
- Hong Kong raw data is not present in this checkout; its tracked summary is
  hash-locked, but it was not rerun in this phase.
- The safe multi-GNSS branch equals EKF by construction. It establishes
  non-degradation, not a new Tokyo positioning gain.
- The PPC QA model consumes post-solver state and cannot be presented as a
  pre-solver navigation estimator.
