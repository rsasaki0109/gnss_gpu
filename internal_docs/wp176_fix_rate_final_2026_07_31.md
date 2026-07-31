# WP176 library FIX-rate improvement audit

## Decision

Adopt the default-off satellite-PAR surplus holdout route with the strict
policy selected by blocked nested cross-validation:

- maximum surplus integer distance: 0.10 cycles;
- minimum fixed subset: 8 DD pairs;
- minimum PAR ratio: 1.4;
- NIS per observation at most 3.0;
- prefit residual RMS at most 50 m;
- every available surplus pair must pass;
- one-epoch satellite-PAR acquisition.

The route uses dropped satellites only as independent post-fit evidence. Each
dropped carrier DD is re-differenced against a fixed-set satellite from the
same constellation/frequency/reference domain, so its test does not reuse the
dropped satellite's float ambiguity. A surplus pass is additive with the
existing disjoint validator; a failure preserves the legacy route.

## Canonical full-run result

| Route | Monitor FIX | Adopted FIX | Gain | False/FIX | >1 m false | p95 |
|---|---:|---:|---:|---:|---:|---:|
| Tokyo run1 | 6,008/11,928 (50.37%) | 6,484/11,928 (54.36%) | +476 (+3.99 pt) | 3/6,484 (0.046%) | 0 | 62.25 ms |
| Nagoya run1 | 5,021/7,602 (66.05%) | 5,274/7,602 (69.38%) | +253 (+3.33 pt) | 0/5,274 | 0 | 46.72 ms |

The correct-FIX rates are 54.33% for Tokyo and 69.38% for Nagoya. Both meet
the integrity and 100 ms latency constraints, but do not meet the 70%/80%
stretch targets. The remaining gaps are 1,866 and 808 correct epochs.

## Selection integrity

Leave-one-city/time-block-out nested CV independently selected the same
0.10-cycle, 8-pair, ratio-1.4 policy in every fold. Its aggregate holdout was
672 correct candidates and zero wrong candidates. On the final monitor data,
the strict candidate set was:

| Route | Strict candidates | Correct | Wrong |
|---|---:|---:|---:|
| Tokyo | 2,658 | 2,658 | 0 |
| Nagoya | 1,364 | 1,364 | 0 |

Truth was used only for policy training folds and post-selection audits, never
by the runtime selector.

## Fault replay

Raw-RINEX cycle-slip, NLOS, satellite-loss, and outage replays were run for
1,000 epochs in both cities. All eight replays had zero false FIX globally and
inside the injected fault windows. Reacquisition p95 was at most 3.02 s.
All latency checks were below 100 ms after Tokyo NLOS was remeasured alone in
the final build (88.39 ms p95 instead of the four-way-contended 122.84 ms).

## Stretch-target boundary

The canonical truth oracle shows that signal exists above the stretch target,
but the unguarded candidates are not safe:

| Route/source | Oracle-correct nonfixed | Oracle-wrong nonfixed | Wrong share |
|---|---:|---:|---:|
| Tokyo source-PAR | 3,348 | 721 | 17.72% |
| Tokyo satellite-PAR | 2,838 | 673 | 19.17% |
| Tokyo causal arc | 2,639 | 402 | 13.22% |
| Nagoya source-PAR | 1,552 | 102 | 6.17% |
| Nagoya satellite-PAR | 1,438 | 120 | 7.70% |
| Nagoya causal arc | 1,250 | 99 | 7.34% |

An oracle could exceed 70%/80%, but every directly available high-coverage
candidate source is orders of magnitude above the 0.1% false-FIX budget.
Among the tested runtime-observable policies, the adopted surplus holdout is
the highest-availability configuration that remains inside the integrity and
latency constraints. Raising FIX further requires a new independent
discriminator rather than looser thresholds.

## Compatibility

Surplus validation and low-pair rescue are default-off. Monitor mode records
the evidence without promotion authority. The production Release target
builds on MSVC; monitor-mode 300-epoch position outputs are byte-identical to
the pre-promotion route. The previously merged PF/FGO GPU work remains
orthogonal: device-side PF reductions and cuSOLVER FGO retain automatic CPU
fallback and do not change this RTK integrity decision.

## Evidence

- `fix_rate_canonical_2026_07_31.json`
- `wp176_surplus_nested_cv_2026_07_31.json`
- `wp176_{tokyo,nagoya}_surplus_rediff_active_additive_streak1_full_v104_2026_07_31.json`
- `wp176_{tokyo,nagoya}_surplus_{cycle_slip,nlos,satellite_loss,outage}_fault_v104_2026_07_31.json`
- `wp176_tokyo_surplus_nlos_fault_v105_2026_07_31.json` (final isolated latency)
