# WP8 — NLOS hard exclusion + canyon float-divergence forensics + live-knob retune (tokyo run1-3)

## 0. Headline

| effect on run1 `<50cm_full%` | result |
|---|---|
| WP7/dead-knob baseline (`--preset low-cost --max-pos-jump-rate 2.3`) | 26.643% |
| NLOS hard exclusion, best of 6 coarse candidates | 21.66-21.78% (**regression, -4.9 to -5.0 pp**), FixRMS blown to 2.1-2.4 m (budget ≤0.5 m) |
| Live-knob retune, best of 12 candidates (`--arfilter-margin 0.0/0.2 --hold-ratio-threshold 2.0`) | 26.92% (**+0.277 pp, below the task's own >0.3 pp bar**) |
| Combination | not run — neither lever independently won |

**Both WP7 next-bottleneck recommendations tested here (hard exclusion, live-knob retune) are negative or inconclusive on run1**, so neither was generalized to run2/run3 or combined, per the task's own decision rules. The canyon-forensics work item, however, produced a definitive, code-verified mechanism verdict (§3) that WP7 could only speculate about — this is the most valuable result of WP8 and is the basis for the WP9 recommendation (§5).

## 1. Hard exclusion mode (work items 1-2)

### 1.1 Design

Added `NlosWeightMode::EXCLUDE` to `nlos_weights.hpp`/`.cpp` alongside two new pure functions:

- `nlosShouldExclude(los_prob, mode, threshold)` — per-satellite decision, true iff `mode == EXCLUDE && los_prob < threshold`.
- `nlosExclusionGuardAllows(total_sats, excluded_count, min_sats)` — epoch-level safety floor, true iff excluding `excluded_count` satellites would still leave `>= min_sats` behind.

Both are unit-tested in isolation (9 new cases in `test_nlos_weights.cpp`; also fixed a real bug caught while writing them — `nlosVarianceInflationFactor` had no explicit `EXCLUDE` branch and was falling through to the continuous-mode inflation formula, which would have double-penalized survivors that should be untouched).

Wired into `RTKProcessor::buildSelectionSnapshot()` (`rtk.cpp`) — confirmed via code reading to be the single shared choke point both `buildMeasurementBlocks()` (float KF) and `buildDoubleDifferencePairs()` (AR/LAMBDA candidate set) call for their satellite list, so exclusion applies identically to both consumers as the task requires. Two safety guards, both implemented in `buildSelectionSnapshot` (RTK-specific selection knowledge, kept out of the generic `nlos_weights` module):

1. **Epoch-level `nlos_min_sats` floor** — if excluding all NLOS-flagged satellites this epoch would drop the surviving count below `nlos_min_sats`, exclusion is skipped entirely for that epoch (everyone kept).
2. **Per-system last-reference-candidate guard** — a system's sole remaining reference-satellite candidate is never excluded, even if it is NLOS-flagged, so DD formation can still proceed for that system.

New CLI flags on `gnss_solve`: `--nlos-weight-mode exclude`, `--nlos-exclude-threshold` (default 0.5), `--nlos-min-sats` (default 5).

### 1.2 Verification

- 9 new `test_nlos_weights.cpp` unit tests (pure-function coverage of both new functions) — all pass.
- 3 new `test_rtk_smoke.cpp` fixture tests (absent-table bit-identical; `min_sats=0` collapses fixed count when every satellite is synthetically NLOS-flagged; `min_sats=999` guard reproduces the exact baseline fixed count) — these `GTEST_SKIP()` in this environment for the same pre-existing missing-fixture-data reason as 8 other `RTKSmokeTest` cases (not new).
- Full C++ suite: **289 tests, 239 passed, 0 failed, 50 skipped** (47 pre-existing + 3 new fixture-dependent skips).
- **Bit-identical absent-flag verification**: reran the exact WP7-baseline full-run1 command (`--preset low-cost --max-pos-jump-rate 2.3`, no WP8 flags) against the WP8-rebuilt `gnss_solve` and compared against `results/wp7/final/run1/d0_wp6_winner_rebuilt.pos` via SHA-256 over the **full** 11928-epoch output (not a slice):

```
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp8/verify/wp8_absentflag_check_run1.pos
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp7/final/run1/d0_wp6_winner_rebuilt.pos
```

Identical. The WP8 C++ diff is a strict no-op when the new flags are absent.

### 1.3 Coarse threshold × min-sats sweep on run1 (work item 2)

`experiments/sweep_libgnss_rtk_wp8.py --stage exclude_coarse` (6 candidates, `--nlos-exclude-threshold {0.3,0.5} × --nlos-min-sats {4,5,6}`, layered on the WP7/dead-knob baseline, full run1). Results in `results/wp8/sweep/run1_exclude_coarse.csv`:

| candidate | fixed | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc | canyon AllRMS | canyon fix% |
|---|---|---|---|---|---|---|---|---|---|
| **WP7 baseline (no exclusion)** | 1130 | 62.75 | 19.471 | 0.313 | 15.10 | **26.643** | 23.65 | 125.61 | 0.0 |
| thr=0.3, min_sats=4 | 1047 | 56.60 | 19.939 | **2.437** | 15.51 | 21.781 | 19.93 | 125.35 | 0.0 |
| thr=0.3, min_sats=5 | 1047 | 56.32 | 20.059 | **2.437** | 15.59 | 21.663 | 19.84 | 125.35 | 0.0 |
| thr=0.3, min_sats=6 | 1047 | 56.43 | 20.032 | **2.132** | 15.55 | 21.739 | 19.96 | 125.35 | 0.0 |
| thr=0.5, min_sats=4 | 1047 | 56.60 | 19.939 | **2.437** | 15.51 | 21.781 | 19.93 | 125.35 | 0.0 |
| thr=0.5, min_sats=5 | 1047 | 56.32 | 20.059 | **2.437** | 15.59 | 21.663 | 19.84 | 125.35 | 0.0 |
| thr=0.5, min_sats=6 | 1047 | 56.43 | 20.032 | **2.132** | 15.55 | 21.739 | 19.96 | 125.35 | 0.0 |

**Every one of the 6 candidates regresses every headline metric** relative to the WP7 baseline: fewer fixed solutions (1047 vs 1130), lower coverage (56.3-56.6% vs 62.75%), and — most importantly — FixRMS blows the ≤0.5 m budget by 4-8× (2.13-2.44 m). The canyon segment is completely unaffected in all 6 (`canyon_all_rms_m` = 125.35, `canyon_fix_pct` = 0.0, identical to the no-exclusion baseline) — the `min_sats`/last-reference-candidate guards fully suppress exclusion there because the canyon already runs on thin, guard-protected satellite counts (confirmed independently by the canyon forensics in §3, where `num_sats` frequently bounces to 0 in this window even without exclusion).

**Threshold 0.3 and 0.5 give byte-identical results for every `min_sats` value.** This is not a bug in the new exclusion code — it is a real property of the input data: `experiments/results/plateau_nlos_phase33/tokyo_run1_per_epoch_nlos.csv` (the phase-33 mask) only carries a boolean `is_los` column (`{0,1}`), no continuous probability. The lookup path maps this to `los_prob ∈ {0.0, 1.0}`, so any `--nlos-exclude-threshold` strictly between 0 and 1 produces the identical exclusion set — the coarse grid's "threshold" axis was a no-op by construction given this mask's binary nature. Effectively only 3 distinct configurations (by `min_sats`) were tested, and all 3 regress.

**Why exclusion regresses despite the WP7 report predicting it might help**: hard exclusion removes NLOS satellites from the DD candidate pool entirely rather than down-weighting them, which — for a run with already-thin fix supply (15% fix rate) — removes some of the geometric diversity that AR needs to reject *wrong* integer candidates. The FixRMS blowup to 2.1-2.4 m indicates exclusion mode is admitting some *wrong* fixes that survived AR with a smaller, less self-checking satellite set, not merely losing coverage.

### 1.4 Decision: no refinement, no generalization

Per the task's own contingency ("apply the best configuration verbatim to run2/run3" implicitly assumes a winning configuration exists): **no candidate in the coarse grid beats the WP7 baseline on any axis that matters** (coverage, AllRMS, FixRMS, `<50cm_full%`, ppc, or canyon behavior). There is no "best exclusion config" to refine or carry forward. This is reported as a clean, unambiguous negative result — refinement (bracketing `min_sats` more finely) and generalization to run2/run3 were both skipped, since applying a regressing configuration there would not test anything new.

## 2. Live-knob retune (work item 4)

`experiments/sweep_libgnss_rtk_wp8.py --stage retune` (12 candidates, `--arfilter-margin {0.0,0.2,0.35,0.5} × --hold-ratio-threshold {2.0,2.5,3.0}`, **no** NLOS flags, full run1). Results in `results/wp8/sweep/run1_retune.csv`:

| margin \ hold | 2.0 | 2.5 | 3.0 |
|---|---|---|---|
| 0.00 | fixed 1130, FixRMS 0.311, **26.920%**, ppc 24.19 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 |
| 0.20 | fixed 1130, FixRMS 0.311, **26.920%**, ppc 24.19 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 |
| 0.35 (baseline margin) | fixed 1130, FixRMS 0.313, 26.635%, ppc 23.64 | fixed 1130, FixRMS 0.313, **26.643% = WP7 baseline exactly**, ppc 23.65 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 |
| 0.50 | fixed 1130, FixRMS 0.313, 26.635%, ppc 23.64 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 | fixed 1130, FixRMS 0.313, 26.643%, ppc 23.65 |

(`margin=0.35, hold=2.5` is the WP7/dead-knob baseline's own default from `--preset low-cost` and reproduces its score exactly — 26.643%/ppc 23.65 — confirming the sweep driver is wired correctly.)

**Findings:**
- `--arfilter-margin` is a **complete no-op** across its entire tested range {0.0, 0.2, 0.35, 0.5} at every `hold-ratio-threshold` value: the fixed count never changes (always 1130) and scores are identical to 3-4 decimal places within each hold-value column. No candidate fix's ratio in this run ever falls in the range where the subset-AR margin changes an accept/reject decision.
- `--hold-ratio-threshold` **is** a real, wired lever (as established in WP7): 2.0 beats 2.5/3.0 (which are identical to each other) by **+0.277 pp** (`26.920%` vs `26.643%`), same fixed count, FixRMS still comfortably inside the ≤0.5 m budget (0.311 m vs 0.313 m).
- The **best improvement found (+0.277 pp) does not clear the task's own >0.3 pp bar** for triggering run2/run3 verification. This is a near-miss, not a clean win.

### Decision: no generalization, no combination

Per the task's explicit rule ("If >0.3 pp improvement over WP7 baseline, verify on run2/run3"), the retune sweep's best result (+0.277 pp) falls just short and was **not** carried forward to run2/run3. Since neither exclusion (§1, regression) nor retune (§2, near-miss) independently won on run1, the "combine both levers" stage (work item 5) was skipped — there is nothing worth combining.

## 3. Canyon float-divergence forensics (work item 3)

### 3.1 Method

Ran the WP7-baseline command (`--preset low-cost --max-pos-jump-rate 2.3`) on full run1 with the newly-added `--debug-epoch-log`, which now also emits (new `EpochDebugTelemetry` fields, `rtk.hpp`/`rtk.cpp`/`gnss_solve.cpp`): `float_update_observation_count`, `float_update_prefit_residual_rms_m`, `float_update_post_suppression_residual_rms_m`, `float_update_nis_per_observation`, `float_update_suppressed_outliers`, `float_position_covariance_trace_m2`. Extracted the window tow 188925-189075 (the specified 188985-189075 canyon segment plus the requested 60 s lead-in) — 751 epochs at 0.2 s cadence.

Built `experiments/diag_canyon_forensics_wp8.py` (9 unit-tested pure functions) to cross-reference four independent signals:

1. The engine's own slip counters from the debug log (`gf_slip_count`, `doppler_slip_*`, `lli_slip_*`, `ambiguity_reset_*`).
2. Raw RINEX LLI bits, parsed directly from `rover.obs` (self-contained scan, independent of the engine's own slip detector, in case it misses something the engine doesn't flag).
3. `float_position_covariance_trace_m2` (position-state 3×3 trace) plus the float update's own prefit/post-suppression residual RMS and NIS-per-observation.
4. Per-satellite raw pseudorange residuals against ground truth (median-subtracted per epoch as a model-free receiver-clock-bias proxy), cross-referenced against the phase-33 LOS/NLOS mask.

### 3.2 Findings

**(a) Cycle slips — present but not the primary driver.** Raw RINEX LLI: 15 of 22 tracked satellites in the window carry ≥1 slip-flagged (odd LLI) observation. Engine's own counters over the 751 epochs: `gf_slip_count`=23, `doppler_slip_l1`=1, `doppler_slip_l2`=11, `code_slip_l2`=4, `lli_slip_l1`=35, `lli_slip_l2`=14, `ambiguity_reset_{l1,l2}`={24,25}. This is elevated relative to a quiet segment but is far too infrequent (slip events on a few percent of epoch-satellite pairs) to explain a covariance regime that is wide-open on 73.5% of *all* epochs (below).

**(b) Float covariance — wide/dragged, not collapsing, and this is the actual mechanism.** Directly measured via the new `float_position_covariance_trace_m2` telemetry:

| segment | n epochs | frac. trace > 500 m² (wide/reset) | frac. trace < 1 m² (fully converged) |
|---|---|---|---|
| Canyon window (tow 188925-189075) | 751 | **73.5%** | 20.8% |
| Healthy contrast segment (tow 187470-187600, open sky) | 651 | **0.0%** | **100%** |

**Root cause, traced to exact code**: `RTKProcessor::resetPositionToSPP()` (`rtk.cpp:1473`) is called **unconditionally every single epoch** (`rtk.cpp:1623`, inside the main per-epoch processing path, not gated behind any of the disabled-by-default reset-streak knobs like `max_consecutive_nonfix_for_reset`/`max_consecutive_float_for_reset`, which are both 0/off in every preset). Every epoch it re-seeds the position state from a fresh SPP fix and **resets the position covariance's 3 diagonal terms to a wide prior** — `var_pos = 900.0` per axis (giving a 2700 m² trace) — **unless** the previous epoch's solution refreshed "trusted" status. Trust-refresh (`rememberSolution`, `rtk.cpp:3741`) requires either a FIXED solution, or a FLOAT solution with ≥5 satellites **and** a small jump versus the last trusted position (`rtk.cpp:3747-3762`); when trusted, `resetPositionToSPP`'s alternate branch reseeds from the tighter, still-recent trusted position instead (`rtk.cpp:1502-1510`, `var_pos` capped to 25 m²/axis for `dt≤1s`).

In the canyon, NLOS satellites carry far larger raw pseudorange errors than LOS satellites (common-mode-removed residual vs ground truth, phase-33-mask cross-referenced, n=20770 sat-epochs): **NLOS median 20.5 m / mean 51.4 m / max 641 m** vs **LOS median 7.2 m / mean 9.6 m / max 267 m**. This corrupts both the per-epoch SPP seed and each epoch's own DD update badly enough that a FLOAT solution rarely stays close enough to the last trusted position to refresh trust — so the *next* epoch resets wide again, from an equally-corrupted SPP seed, in a self-reinforcing loop with essentially zero cross-epoch position memory for ~92 continuous seconds. `reject_reason=adaptive_position_jump` fires on 478/751 epochs (63.6%) as a **downstream symptom** of this same loop (the exported solution keeps getting vetoed for jumping relative to the last accepted one), not an independent cause.

**(c) NLOS satellites and the biggest residuals — confirmed, and quantified above in (b).** 7526 of 20770 (36.2%) sat-epochs in the window are NLOS-flagged by the phase-33 mask, and those carry residuals ~2.8× the median / ~5.4× the mean magnitude of LOS satellites' residuals (and a 2.4× larger worst case). This is the trigger; (b)'s reset/trust-refresh policy is the mechanism that turns "some satellites have bad pseudoranges" into "the filter has no memory for 92 seconds."

### 3.3 Mechanism verdict

**Filter-tuning** (the always-on, per-epoch position-covariance reset/trust-refresh policy in `resetPositionToSPP`/`rememberSolution`) **triggered by NLOS pseudorange corruption** (which starves the policy of trust-refresh opportunities), with cycle slips as a real but secondary/minor contributor. Not primarily a slip-detection gap, and not a signal-outage problem in the literal sense (`float_update_observation_count` stays in the 20-36 range throughout the window — the receiver never stops tracking satellites; what collapses is the *filter's own retained confidence* in its position estimate between epochs, by design, once trust lapses).

### 3.4 Engine fix: documented for WP9, not attempted here

No cheap, safely flag-guarded fix was implemented. The reset/trust-refresh policy in `resetPositionToSPP`/`rememberSolution` is core, always-active logic exercised on **every epoch of every run** (not narrowly canyon-specific) — it is what keeps kinematic-mode position estimates from silently drifting when nothing is fixed. Loosening it (e.g., widening the trusted-position jump/`dt` gate, or substituting a lighter constant-velocity predict step for the full wide reset when trust lapses, or making the reset "wideness" scale down when many candidate satellites are NLOS-flagged rather than resetting to a fixed 900 m²/axis regardless) is a substantive, broadly-exercised engine-behavior change that needs regression testing across all three runs and the existing tuned-knob combinations (WP6/WP7's own AR-config and NLOS-weighting knobs) before it could be trusted not to regress the 92-108% of the timeline where the filter behaves well today. This exceeds this task's "cheap" bar; **recommending it as a concrete, code-cited WP9 investigation** (§5) rather than shipping an under-tested core-filter change.

## 4. Deliverables

- `results/wp8/WP8_REPORT.md` (this file)
- `results/wp8/final/run1/*.pos` (18 candidate trajectories: 6 exclusion coarse + 12 retune)
- `results/wp8/final/run1/scores/*.json` (score JSONs for the WP7-baseline reproduction, the least-bad exclusion candidate, and the retune winner)
- `results/wp8/sweep/run1_exclude_coarse.csv`, `run1_retune.csv` (+ matching `.log` files)
- `results/wp8/canyon/wp7_baseline_full_debuglog.{pos,csv}` (full-run1 debug-epoch-log)
- `results/wp8/canyon/canyon_forensics_summary.json` (structured forensics output)
- `results/wp8/verify/wp8_absentflag_check_run1.pos` (bit-identical verification artifact)
- `experiments/diag_canyon_forensics_wp8.py` + `tests/test_diag_canyon_forensics_wp8.py` (9 tests)
- `experiments/sweep_libgnss_rtk_wp8.py` + `tests/test_sweep_libgnss_rtk_wp8.py` (10 tests)
- C++ diffs: `nlos_weights.{hpp,cpp}` (EXCLUDE mode), `rtk.hpp`/`rtk.cpp` (exclusion wiring + new debug telemetry fields), `gnss_solve.cpp` (new CLI flags + debug-log columns) — all flag-guarded, bit-identical when absent (§1.2)
- New/extended C++ tests: `test_nlos_weights.cpp` (+9), `test_rtk_smoke.cpp` (+3, fixture-skipped in this environment)

## 5. Next-bottleneck recommendation (WP9 input)

1. **(Highest-conviction, code-cited)** Investigate loosening `resetPositionToSPP`/`rememberSolution`'s trust-refresh policy specifically for NLOS-heavy epochs — e.g., degrade gracefully (smaller reset, or a constant-velocity predict instead of a full SPP reseed) rather than unconditionally resetting to a 900 m²/axis prior every time trust lapses, and/or relax the FLOAT trust-refresh jump gate (`rtk.cpp:3747-3762`) when a large fraction of visible satellites are NLOS-flagged. This directly targets the mechanism found in §3 (73.5% of the canyon window sits in the wide/untrusted-reset regime vs 0% in a healthy segment) but needs broad regression testing (all 3 runs × existing AR-config knobs) before shipping, since the policy is exercised on every epoch of every run.
2. Hard exclusion (this WP) was a clean loss on run1 in its current "remove from DD entirely" form; a **softer variant worth testing before abandoning the idea** is NLOS-aware AR candidate diversity preservation — e.g., still down-weight (not remove) NLOS pseudoranges in the float KF (WP7's continuous/two-tier modes), but additionally require AR to see a minimum count of *LOS* satellites specifically (rather than total satellites) before accepting a fix, which might explain/fix the FixRMS blowup (§1.3) without losing the geometric diversity hard exclusion removed.
3. The retune near-miss (+0.277 pp, just under the bar) suggests `--hold-ratio-threshold 2.0` is a mild, real, low-risk win worth re-testing in combination with future NLOS/canyon fixes (item 1) rather than in isolation, since its effect size may compound once the canyon segment itself contributes fixed solutions (it currently contributes zero, `canyon_fix_pct=0.0` in every WP7/WP8 configuration tested to date).

## 6. Tests

- C++: `run_tests` — **289 tests, 239 passed, 0 failed, 50 skipped** (47 pre-existing + 3 new WP8 fixture-dependent skips, same missing-data gap as other pre-existing cases; unchanged from the WP8 build-verification step since no further C++ changes were made after it).
- Python: `tests/test_sweep_libgnss_rtk_wp8.py` (10), `tests/test_diag_canyon_forensics_wp8.py` (9), `tests/test_sweep_libgnss_rtk_wp7.py` (8) — **27 passed, 0 failed**.

No git commits made. All builds done in WSL. No changes to protected files (`validate_fgo_ppc.py`, `python/gnss_gpu/io/ppc.py`).
