# WP24 Progress — evidence ledger and deterministic replay

## 2026-07-18 — evaluation contract

| Gate | Status | Evidence |
| --- | --- | --- |
| truth-free common trace | **pass** | 200 rows; no truth/reference/error field |
| evidence provenance | **pass** | 793 records, 793 updates, zero duplicate stage |
| likelihood consumption | **pass** | zero beta-total error |
| basin marginal evidence | **pass** | 273 mixture log-marginal records |
| deterministic FIX replay | **pass** | zero online/replay mismatch |
| default neutrality | **pass** | position/gamma/FIX exactly match `b3f1106` |

Implemented `gnss_gpu.rtk_evidence` with an append-only evidence ledger,
strict `(epoch,target,source,observation,stage)` uniqueness, beta-consumption
audit, stable versioned-assignment identities, a truth-free epoch trace, and a
stateful trusted FIX policy that can be replayed without rerunning the
estimator.

The WP23b runner records Doppler, DD pseudorange, and DD carrier applications
separately for the float KF, DDPR guard, and basin posterior. Basin updates now
return the mixture log marginal computed before normalization/capping. LAMBDA
candidate birth is deliberately not called evidence.

Tokyo run2/200 smoke result:

- 14/14 correct FIX, zero false;
- 793 evidence records and zero beta errors;
- 273 basin log-evidence values;
- zero online/replay FIX or streak mismatch;
- against a fresh detached `b3f1106` execution: zero position delta, zero
  gamma delta, zero FIX and gamma-FIX mismatch.

Focused validation passed with 19 tests after the final evidence integration.
Compact artifacts are `csv/wp24_smoke_run2_200_{summary,trace,evidence}` and
`csv/wp24_smoke_equivalence.json`.

## Next

WP25 will consume this trace contract to compare single-epoch MAP selection
against multi-epoch assignment-lineage posterior and motion consistency.
