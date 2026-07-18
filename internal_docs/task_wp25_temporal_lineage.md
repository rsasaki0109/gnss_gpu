# WP25 — multi-epoch ambiguity lineage posterior

Date: 2026-07-18. Parent: `pf_only_rtk_scaleup_plan_2026_07_18.md`.

## Objective

Select persistent integer-ambiguity lineages using per-epoch evidence and
truth-free motion consistency, instead of sharpening DDPR or repeatedly
feeding the already cumulative basin posterior back into itself.

## First increment

1. Record each basin's current-epoch log-likelihood increment separately from
   its cumulative marginal likelihood.
2. Implement a normalized discrete transition filter over canonical ambiguity
   assignments. Transitions distinguish exact stay, compatible partial
   adopt/release, and incompatible birth/re-entry.
3. Include an optional constant-velocity motion transition, with an explicit
   covariance rather than a hard distance gate.
4. Retain immediate ancestors and a bounded Viterbi history for later
   fixed-lag `2/5/10 s` evaluation.
5. Wire the temporal posterior as a diagnostic-only runner arm. It must not
   change emitted position or trusted FIX decisions in this increment.

## Statistical constraints

- Observation input is the basin's current-epoch likelihood increment, not
  normalized cumulative weight.
- Every previous-state transition row and the birth proposal are normalized.
- Motion evidence is applied once in the transition and recorded separately
  from DDPR/DDCP observation evidence.
- A changed ambiguity generation cannot be treated as an exact stay.
- Temporal gamma is not a production confidence until calibration and
  false-fix gates pass on full runs.

## Gates

- Synthetic alternating-distractor tests show persistent correct lineage
  selection where single-epoch MAP fails.
- Slip, partial release/adopt, incompatible birth, normalization, and ancestry
  backtrace have focused tests.
- Default runner output remains identical with the arm disabled and enabled
  diagnostic output cannot affect the commit path.
- Real PPC diagnostics report temporal-vs-single-epoch oracle selection,
  lineage survival, gamma calibration, and lag ablations.
- Production integration requires improved selection on held-out runs with
  declared-FIX false rate `<=1%`.
