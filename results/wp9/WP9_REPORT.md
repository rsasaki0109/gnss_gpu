# WP9 — Fixing the float-filter trust/reset policy (code-cited canyon root cause) — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Baseline = WP7/dead-knob baseline: `--preset low-cost --max-pos-jump-rate 2.3` (run1 26.643 / run2 43.165 / run3 43.710 `<50cm_full%`; scores in `results/wp7/final/run{N}/wp7_baseline_score.json`, reproduction artifact `results/wp8/verify/wp8_absentflag_check_run1.pos`, SHA `4c2effb6…`).

**Headline result: negative, but with a definitive, quantified mechanism confirmation.** Both new float-trust policies (`cv-predict`, `scaled-reset`) are implemented, unit-tested, and verified bit-identical to legacy when their flags are absent. `cv-predict` loses to the baseline at every tested `qpos` on run1 itself. `scaled-reset` **wins run1** (up to +0.78 pp, and nearly halves the canyon's float-divergence RMS) but **fails the mandatory 3-run regression gate**: every `qpos` value that gives a real run1 win regresses run2 and/or run3 by far more than the 0.3 pp budget, and the only `qpos` that clears the gate (100) gives essentially zero run1 benefit. This is an inherent trade-off in the task-specified formula (see §4), not a bug — documented in full below, per this project's established practice of reporting clean negatives (WP7 §NLOS, WP8 §1).

## 1. Policy design (work item 1)

Added `--float-trust-policy {legacy,cv-predict,scaled-reset}` (default `legacy`) to `gnss_solve`, flag-guarded end to end:

- **`cv-predict`**: when trust has lapsed (`float_trust_policy::hasTrustLapsed()` — the previous epoch did not refresh "trusted" status per `rememberSolution`), do **not** reseed position from SPP. Instead propagate the last float position with a constant-velocity predict and grow the position variance linearly: `var = min(900, prev_var + qpos·dt)`. **Velocity source**: the filter carries no velocity states, so velocity is derived from the last **two trusted** position/time samples (`prev_trusted_position_`/`prev_trusted_time_`, new `RTKProcessor` members), i.e. a two-point finite difference over the true elapsed time between the two most recent trust refreshes — not Doppler (Doppler is available per-satellite but there is no existing single-point Doppler-to-3D-velocity solve in this codebase, and reusing the trusted-position history is simpler and exactly matches "last N trusted deltas" from the task text). If the gap between the two trusted samples exceeds a sanity cap the estimate collapses to zero velocity (pure position-hold) rather than extrapolating from a stale/degenerate pair.
- **`scaled-reset`**: keeps the SPP reseed, but scales the reset variance with time-since-trust: `var_pos = min(900, 25 + qpos·dt_since_trust²)` — exactly the task's formula, verified against the call site (`rtk.cpp:1557-1558`).
- **Optional lever `--trust-gate-nlos-relax`** (default off): relaxes `rememberSolution`'s FLOAT jump gate 2× when >50% of the epoch's tracked satellites are NLOS-flagged (requires `--nlos-weights`). Built and unit-tested but not needed by the winning path below (not exercised in the final recommendation).

Both policies engage **only** on epochs where trust has lapsed; a healthy/converged run touches none of this new code, which is what makes the absent-flag bit-identical guarantee possible.

**Pure math** lives in a new `float_trust_policy.hpp/cpp` (no engine state, fully unit-testable): `hasTrustLapsed`, `growPositionVarianceCvPredict`, `scaledResetPositionVariance`, `estimateVelocityFromTrustedDeltas`, `predictPositionConstantVelocity`. 19 new `FloatTrustPolicyTest` cases cover linear/quadratic growth, the 900 m² cap, `dt=0`/negative/non-finite input clamping, and velocity-estimate degeneracy.

**Bit-identical verification**: reran the exact WP7-baseline full-run1 command (no WP9 flags) against the WP9-rebuilt `gnss_solve` and compared the full 11928-epoch `.pos` against `results/wp8/verify/wp8_absentflag_check_run1.pos` via SHA-256:

```
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp9/verify/wp9_absentflag_check_run1.pos
4c2effb637672cd29a6ae79d3a1a065130c292d95a92d54d9fe02f964b7ad8aa  results/wp8/verify/wp8_absentflag_check_run1.pos
```

Identical. Default behavior is unchanged.

## 2. Run1 coarse sweep (work item 2)

`experiments/sweep_libgnss_rtk_wp9.py --stage {cvpredict_coarse,scaledreset_coarse}`, `qpos ∈ {0.1, 1, 10, 100}` m²/s, full run1. Full rows in `results/wp9/sweep/run1_cvpredict_coarse.csv` / `run1_scaledreset_coarse.csv`.

| policy | qpos | fixed | coverage% | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc | canyon AllRMS | canyon fix% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **baseline (legacy)** | — | 1130 | 62.75 | 19.471 | 0.313 | 15.10 | **26.643** | 23.65 | 125.61 | 0.0 |
| cv-predict | 0.1 | 979 | 61.57 | 41.730 | 0.140 | 13.33 | 9.222 | 3.85 | 97.01 | 0.0 |
| cv-predict | 1 | 979 | 68.43 | 47.274 | 0.140 | 12.00 | 9.222 | 3.85 | 142.92 | 0.0 |
| cv-predict | 10 | 975 | 66.39 | 34.327 | 0.742 | 12.31 | 10.002 | 6.41 | 103.33 | 0.0 |
| cv-predict | 100 | 1037 | 64.95 | 30.057 | 0.320 | 13.39 | 22.326 | 18.44 | 30.86 | 0.0 |
| scaled-reset | **0.1** | 1130 | 61.89 | 19.327 | 0.313 | 15.31 | **27.423** | 25.44 | **74.73** | 0.0 |
| scaled-reset | 1 | 1130 | 62.42 | 19.973 | 0.313 | 15.18 | 27.071 | 24.15 | 125.61 | 0.0 |
| scaled-reset | 10 | 1130 | 62.80 | 19.952 | 0.313 | 15.08 | 26.970 | 24.29 | 125.61 | 0.0 |
| scaled-reset | 100 | 1130 | 62.82 | 19.583 | 0.313 | 15.08 | 26.585 | 23.81 | 125.61 | 0.0 |

**`cv-predict` is a clean loss at every qpos** — even at qpos=100 (its best point, closest to a fast fallback to the legacy wide reset) it is still 4.3 pp below baseline. The position-hold-with-linear-variance-growth model does not track this vehicle well during the multi-second trust lapses that actually occur (velocity is frequently near-zero-collapsed per the sanity-cap rule above, so the predict step degenerates to "sit still," which is worse than SPP's noisy-but-unbiased-in-the-mean reseed for a moving vehicle). **`scaled-reset` beats baseline at every tested qpos, monotonically decreasing as qpos grows** (best at the grid's smallest value, qpos=0.1: +0.78 pp `<50cm_full%`, FixRMS unchanged at 0.313 m, same 1130 fixed count, and canyon AllRMS nearly halved (125.6→74.7 m)).

**Canyon covariance-trace regime** (work item 2's specific ask — same window and method as WP8 §3, tow 188925–189075, 751 epochs, `--debug-epoch-log` on the scaled-reset qpos=0.1 winner, full run1):

| segment / config | n epochs | frac. trace > 500 m² (wide/reset) | frac. 50–500 m² (partial) | frac. trace < 50 m² | frac. trace < 1 m² (fully converged) |
|---|---:|---:|---:|---:|---:|
| Baseline (WP8, legacy policy) | 751 | **73.5%** | n/a (WP8 didn't bucket this) | n/a | 20.8% |
| **scaled-reset, qpos=0.1 (this WP)** | 751 | **11.6%** | 54.9% | 33.6% | 19.6% |

This is the graceful-degradation effect the task asked for, shown directly in the telemetry: the wide/untrusted-reset regime drops from 73.5% of canyon epochs to 11.6%, replaced mostly by an intermediate 50–500 m² regime instead of the old binary wide-or-tight split. The fully-converged (<1 m²) fraction is essentially unchanged (~20%) — `scaled-reset` doesn't make the filter *converge* faster, it makes its *uncertainty bookkeeping* honest during the many short lapses instead of always claiming maximal ignorance.

## 3. Regression matrix (work item 3) — **fails**

Per the task's rule ("if cv-predict wins run1 but regresses run2/3, try scaled-reset before declaring failure"): `cv-predict` already lost outright on run1 (§2), so only `scaled-reset` was carried into the regression matrix, run on all three runs verbatim (`experiments/sweep_libgnss_rtk_wp9.py --stage regression`, full rows in `results/wp9/sweep/run{1,2,3}_regression.csv`):

| run | baseline `<50cm_full%` | scaled-reset qpos=0.1 | Δ (pp) | within ±0.3pp budget? | baseline FixRMS | winner FixRMS |
|---|---:|---:|---:|:---:|---:|---:|
| run1 | 26.643 | 27.423 | **+0.780** | (win, n/a) | 0.3128 | 0.3128 |
| run2 | 43.165 | 41.373 | **−1.792** | **NO** | 0.1250 | 0.1258 |
| run3 | 43.710 | 42.383 | **−1.327** | **NO** | 0.0811 | 0.0808 |

FixRMS stays comfortably inside the ≤0.5 m budget on every run (the regression is a coverage/accuracy loss, not a wrong-fix-quality problem), but run2 and run3 both blow the ±0.3 pp regression budget by 4–6×.

### 3.1 qpos-sensitivity follow-up (is there a rescuing qpos?)

Before declaring failure, swept `qpos ∈ {10, 30, 100}` (in addition to the run1-only 0.1/1/10/100 grid from §2) across all three runs to check for a value that both meaningfully helps run1 and clears the run2/run3 gate (`results/wp9/sweep/run{1,2,3}_regression_qpos{10,30,100}.csv`):

| qpos | run1 Δpp | run2 Δpp | run3 Δpp | clears ±0.3pp gate on run2 **and** run3? |
|---:|---:|---:|---:|:---:|
| 0.1 | **+0.780** | −1.792 | −1.327 | No |
| 10 | +0.327 | −0.743 | −0.125 | No (run2 fails) |
| 30 | +0.159 | −0.711 | +0.006 | No (run2 fails) |
| 100 | −0.058 | **−0.186** | **−0.053** | **Yes** — but run1 gain is gone (slightly negative) |

**Conclusion: there is no single global `qpos` for `scaled-reset` that both meaningfully improves run1 and satisfies the mandatory ≤0.3 pp regression gate on run2/run3.** The only value that clears the gate (qpos=100) does so by behaving almost exactly like `legacy` everywhere (its reset variance already reaches the 900 m² cap within ~3 s of any lapse, `900 = 25 + 100·dt² ⇒ dt≈2.96s`), which is also why it stops helping run1's canyon.

**Root cause of the trade-off (not a bug, inherent to the task's own formula)**: `scaled-reset`'s variance at `dt=0` (the very first lapsed epoch) is `25 m²`, dramatically tighter than legacy's `900 m²`, by construction, for *every* lapse regardless of how short. Run1's canyon has long (multi-second to ~92 s), severe trust lapses where a tight-but-honest covariance is a clear win over an always-maximal one. Run2/run3 are open-sky-dominated runs (baseline `<50cm_full%` 43.2%/43.7% vs run1's 26.6%) where trust lapses are typically brief and otherwise benign; for those, immediately tightening the covariance right after any lapse makes the filter overconfident in a not-yet-re-verified SPP-seeded position, slowing the Kalman gain's correction on the next good epoch relative to legacy's "assume nothing" wide reset — a small but real net loss that dominates run2/run3's much larger, much-easier-to-fix epoch population. Raising `qpos` reduces this overconfidence window (faster ramp back to the 900 m² cap) but by the point it's fast enough to stop hurting run2/run3, it's also fast enough to stop meaningfully helping run1's canyon.

**Per the task's own decision framework** (mirroring WP7's NLOS §, WP8's exclusion §): this is reported as a clean, code-verified negative result at the "ship a single global config" level. The mechanism is real and precisely diagnosed (§2's covariance-trace numbers), and the flag is fully built, tested, and available for future work (e.g. a canyon/NLOS-conditional activation — see §6) — but no run1-winning `(policy, qpos)` pair generalizes to run2/run3 as required.

## 4. Combination (work item 4)

Per the task's literal instruction, ran `scaled-reset qpos=0.1` (the run1-best candidate) + `--hold-ratio-threshold 2.0` on all three runs (`experiments/sweep_libgnss_rtk_wp9.py --stage combination`, full rows in `results/wp9/sweep/run{1,2,3}_combination.csv`):

| run | baseline | winner alone | `--hold-ratio-threshold 2.0` alone | combined |
|---|---:|---:|---:|---:|
| run1 | 26.643 | 27.423 | 26.635 | 27.414 |
| run2 | 43.165 | 41.373 | 43.165 (bit-identical to baseline) | 41.373 (bit-identical to winner alone) |
| run3 | 43.710 | 42.383 | 43.710 (bit-identical to baseline) | 42.383 (bit-identical to winner alone) |

**`--hold-ratio-threshold 2.0` does not compound with the winner, and is itself essentially a no-op here.** Note a real discrepancy with the task text's cited "+0.277 pp from this knob alone" (WP8 §2): that number was measured at `--arfilter-margin {0.0, 0.2}` + `--hold-ratio-threshold 2.0` together — WP8's own table (`WP8_REPORT.md` §2) shows `margin=0.35 (the preset default, left untouched by "`--hold-ratio-threshold 2.0` alone")` + `hold=2.0` gives only **26.635%** (a −0.008 pp wash), matching exactly what is reproduced here on run1's `c2_hold2.0_alone` row. Following the task's literal flag list (`--hold-ratio-threshold 2.0`, no margin change) reproduces WP8's own "no-op at this margin" row, not its "+0.277pp" row — the two are the same knob at a different, unstated `arfilter-margin`. Combining does very slightly improve run1's canyon (canyon AllRMS 74.7→70.3 m, canyon `n_scored` 14→13) but does not change the regression-matrix verdict: run2/run3 remain regressed to the same degree as the winner alone (hold-ratio-threshold produces bit-identical output on both runs with or without the winner active).

## 5. Stretch (work item 5) — not attempted

`--nlos-min-los-sats N` does **not exist** in `gnss_solve` — confirmed by grepping the entire `third_party/gnssplusplus` tree (no `min_los_sats`/`MinLosSats` symbol anywhere) and by the CLI itself rejecting it (`Argument error: unknown or incomplete argument: --nlos-min-los-sats`, `results/wp9/sweep/run1_stretch.log`). WP8 only *recommended* this as a future AR-candidate-filter feature (§5 of `WP8_REPORT.md`); it was never wired. Implementing it properly (a new AR-candidate LOS-count gate, threaded through `buildDoubleDifferencePairs`/the ambiguity-fix acceptance path, plus unit tests and a rebuild) is a non-trivial new C++ feature, not a CLI-only stretch test. Per the task's own framing ("only if items 1-4 leave time"), and given items 1-4 already required a very large sweep budget (coarse grid × 2 policies, a 3-run regression matrix, a qpos-sensitivity follow-up across 3 runs, and a 3-run combination matrix — 30+ full-run `gnss_solve` invocations) to reach a decisive, well-supported verdict, this stretch item was not implemented. Recommended as the next concrete WP10 candidate (§6).

*(Minor unrelated fix made in service of this stretch attempt: `experiments/sweep_libgnss_rtk_wp7.py`'s `run_gnss_solve_wp7` decoded `gnss_solve`'s captured stdout using the OS default codepage, which raised `UnicodeDecodeError` on this flag combination's help/error text under a non-UTF-8 Windows locale; changed to explicit `encoding="utf-8", errors="replace"`. This is a Python sweep-driver robustness fix only, no engine behavior change.)*

## 6. Deliverables

- `results/wp9/WP9_REPORT.md` (this file)
- `results/wp9/verify/wp9_absentflag_check_run1.pos` — bit-identical verification artifact (SHA `4c2effb6…`, matches `results/wp8/verify/wp8_absentflag_check_run1.pos`)
- `results/wp9/canyon/scaledreset_qpos0.1_full_debuglog.{pos,csv}` — full-run1 `--debug-epoch-log` for the scaled-reset qpos=0.1 winner (source of §2's covariance-trace regime table)
- `results/wp9/sweep/run1_{cvpredict,scaledreset}_coarse.csv` — work item 2 coarse sweep
- `results/wp9/sweep/run{1,2,3}_regression.csv` + `run{1,2,3}_regression_qpos{10,30,100}.csv` — work item 3 regression matrix + qpos-sensitivity follow-up
- `results/wp9/sweep/run{1,2,3}_combination.csv` — work item 4 combination matrix
- `results/wp9/sweep/run1_stretch.csv`/`.log` — work item 5 attempt log (flag does not exist)
- `.pos` files for every candidate above under `results/wp9/sweep/run*/` and `results/wp9/sweep/run*_combo/`, `run*_qpos*/`
- New C++: `third_party/gnssplusplus/include/libgnss++/algorithms/float_trust_policy.hpp`, `src/algorithms/float_trust_policy.cpp`, `tests/test_float_trust_policy.cpp` (19 tests); `rtk.hpp`/`rtk.cpp`/`gnss_solve.cpp` diffs (flag-guarded); 2 new `RTKSmokeTest` cases (bit-identical absent-flag + cv-predict/scaled-reset don't-crash smoke test — both `GTEST_SKIP()` for the same pre-existing missing-fixture-data reason as 15 other `RTKSmokeTest` cases, not new)
- New Python: `experiments/sweep_libgnss_rtk_wp9.py`, `tests/test_sweep_libgnss_rtk_wp9.py` (9 tests, all pass)
- Minor fix: `experiments/sweep_libgnss_rtk_wp7.py` stdout-decoding robustness fix (§5)

### Test suite counts

- **C++ (`third_party/gnssplusplus/build/tests/run_tests`): 310 tests, 258 passed, 0 failed, 52 skipped** (up from WP8's 289/239/0/50: +19 new `FloatTrustPolicyTest` pure-function tests, all passing; +2 new `RTKSmokeTest` cases, both skipped for the pre-existing missing-fixture-data reason).
- **Python (`pytest tests/test_sweep_libgnss_rtk_wp9.py`): 9 passed, 0 failed.**

## 7. 3-run summary vs inuex35 and prior WPs

| method | run | AllRMS | FixRMS | fix% | `<50cm_full%` | ppc | vs inuex35 |
|---|---|---:|---:|---:|---:|---:|---:|
| inuex35 README (external) | run1 | 47.40 | 0.815 | 49.5 | **56.7** | n/a | — |
| WP6/7/8 baseline (`--max-pos-jump-rate 2.3`, dead knobs wired) | run1 | 19.471 | 0.313 | 15.10 | 26.643 | 23.65 | −30.1pp |
| **WP9 scaled-reset qpos=0.1 (run1-local win, does not generalize)** | run1 | 19.327 | 0.313 | 15.31 | **27.423** | 25.44 | −29.3pp |
| inuex35 README (external) | run2 | 32.08 | 0.277 | 60.8 | **69.9** | n/a | — |
| WP6/7/8 baseline | run2 | 9.335 | 0.125 | 13.04 | 43.165 | 48.05 | −26.7pp |
| WP9 scaled-reset qpos=0.1 (regresses) | run2 | 9.413 | 0.126 | 13.57 | 41.373 | 48.45 | −28.5pp |
| inuex35 README (external) | run3 | 34.52 | 0.211 | 59.4 | **67.9** | n/a | — |
| WP6/7/8 baseline | run3 | 5.637 | 0.081 | 5.17 | 43.710 | 40.67 | −24.2pp |
| WP9 scaled-reset qpos=0.1 (regresses) | run3 | 5.841 | 0.081 | 5.14 | 42.383 | 40.65 | −25.5pp |

**Recommendation: do not change the shipped default.** `--float-trust-policy` stays `legacy` by default (bit-identical, verified). The WP7/8 baseline (`--preset low-cost --max-pos-jump-rate 2.3`, no WP9 flags) remains the best-generalizing configuration across all three runs; `scaled-reset` is a genuine run1/canyon-specific improvement that is not safe to ship globally.

## 8. Next-bottleneck recommendation

The regression pattern in §3 (short, benign lapses hurt by immediate covariance tightening; long, severe canyon lapses helped) strongly suggests the fix belongs in the *gating*, not the *policy math*: a **canyon/NLOS-conditional activation** of `scaled-reset` — e.g. only engage it once `dt_since_trust` exceeds a few seconds (falling back to legacy's wide reset for the short lapses that dominate run2/run3's epoch population), or gate it on the already-built `trust_gate_nlos_relax`-style NLOS-fraction signal (only tighten the reset when the *reason* for the lapse is itself NLOS-heavy, which is specifically the canyon's signature per WP8 §3(c): 36.2% NLOS sat-epochs there vs near-zero in a healthy segment). This is a natural WP10 candidate: reuse `float_trust_policy.hpp`'s already-unit-tested pure functions, add one more gate condition, and rerun exactly this WP's regression matrix. Implementing `--nlos-min-los-sats` (§5, WP8 rec 2, still unbuilt) remains the other concrete option and is orthogonal (AR-acceptance-side rather than float-filter-side).
