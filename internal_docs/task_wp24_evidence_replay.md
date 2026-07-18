# WP24 — evidence ledger and deterministic FIX replay

Date: 2026-07-18. Parent: `internal_docs/pf_only_rtk_scaleup_plan_2026_07_18.md`.

## Objective

Create the truth-free evaluation contract required before multi-epoch ambiguity
inference. Every observation update must have auditable provenance, and every
trusted FIX decision must be reproducible from a compact epoch trace without
rerunning the estimator.

## Scope

1. Add a common truth-free epoch trace containing MAP assignment identity,
   gamma, support, guard separations, commit streak, and emitted position.
2. Add an evidence ledger keyed by epoch, posterior target, observation source,
   observation identity, and annealing stage.
3. Reject duplicate stages and beta totals other than one for every applied
   target/source/observation update.
4. Extract the trusted FIX streak and guard composition into a deterministic,
   replayable policy.
5. Integrate the contract into the WP23b runner without changing its default
   position or FIX outputs.

Truth and scoring errors must not appear in the replay trace or policy input.
Proposal generation is not measurement evidence and is not recorded as such.

## Gates

- Unit tests cover duplicate evidence, incomplete/over-consumed beta, replay
  equivalence, streak reset, and guard rejection.
- The runner asserts that the extracted policy agrees with the legacy basin
  posterior FIX state at every epoch.
- A Tokyo run2 smoke has zero online/replay mismatch and zero evidence-audit
  failure.
- Default output/FIX decisions match the pre-WP24 reference within numerical
  tolerance.
- Focused WP21–WP23 regressions remain green.

## Artifacts

- `python/gnss_gpu/rtk_evidence.py`;
- `tests/test_rtk_evidence.py`;
- opt-in truth-free trace and evidence CSVs from the WP23b runner;
- `results/wp24/PROGRESS.md` and `results/wp24/WP24_REPORT.md`.
