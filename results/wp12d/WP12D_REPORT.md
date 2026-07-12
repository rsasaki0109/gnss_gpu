# WP12d — Quality-gated LAMBDA AR on the TC-FGO Schur float — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Builds on WP12c (`--schur-marginal`, open-sky float **0.04 m RMS**). WP12d implements the WP12c §6 mandate: offer LAMBDA only when a per-window float certificate passes, then run the validation stack (ratio 3.0 → subset-AR → DDPR cross-validation → fix-and-hold + post-AR cost gate).

**Headline result: stage-2 gate FAIL, but a blocking feedback defect is root-caused and fixed.** Before the fix, accepted AR changed the epoch *status label* while exporting the *float position* (AllRMS identical to 3 decimals, DDPR gate blind). After the fix, re-solve with held integers correctly moves positions (1 351 / 3 900 probe epochs shift; unit test confirms cm-level convergence on synthetic geometry). The campaign metric still does not move: `<50cm_full%` stays **6.8 %** on the 4k probe and **7.0 %** on full run1 because the certificate only fires in open-sky stretches where the float is *already* sub-meter — AR cannot manufacture fixes in the drift/canyon regions that dominate the denominator.

## 1. Gate verdicts

| gate | criterion | run1 result | pass? |
|---|---|---|:---:|
| **Stage 2 (AR quality)** | `<50cm_full%` **> 26.6** with FixRMS **≤ 0.8 m** | full run1 **7.0 %**, FixRMS **60.1 m** (best probe subset: FixRMS **0.70 m**, `<50cm_full%` still **6.8 %**) | **NO** |

Shipped config (best post-fix): `--recovery --anchor-fix --dynamic-dd-rebuild --dd-carrier --persistent-ambiguities --schur-marginal --lambda-ar --ar-cert-max-pos-sigma 0.15 --ar-cert-max-dd-pr-rms 1.0` (full validation stack on).

## 2. Feedback-defect postmortem (root cause + fix)

### 2.1 Symptom

Pre-fix ablation: AllRMS **156.417 m** on every AR stage (identical to 3 decimals); `<50cm_full%` frozen at **6.79 %**; DDPR cross-validation **0 rejects**; cert-tight FixRMS **0.627 m** with open-sky fixed truth RMS **0.036 m** — i.e. fixes were *labeling* epochs that were already at float accuracy, not tightening them.

Hypotheses (1) export timing and (2) missing re-solve were **false**: `run_tc_window_ar` already re-solves via `solve_tc_fgo_window` and the runner writes `result.states` back into `epoch_states` before export.

### 2.2 Root cause

**Hypothesis (3) confirmed — wrong held-integer position Jacobian in the LM linearizer.** `ambiguity_carrier_residual_and_jacobian` returned `jac_m` (meters) for held integers instead of `-jac_m / wavelength` (cycles w.r.t. position), inconsistent with the float-ambiguity branch and with `local_fgo._dd_carrier_fixed_error_fn` (the reference implementation).

```1063:1065:python/gnss_gpu/tc_fgo.py
    if held_integer is not None:
        residual = dd - expected_m / wavelength - float(held_integer)
        return float(residual), -jac_m / wavelength, None
```

(Pre-fix line returned `jac_m` without sign or wavelength scaling.)

With a broken gradient, the post-AR re-solve could not move position under tight held-carrier constraints; `pos_before ≈ pos_after` in DDPR cross-check → gate blind.

### 2.3 Secondary hardening

`build_ambiguity_layout` now also reads `DDCarrierEpoch.fixed_ambiguities` when `held_global` lacks an entry, so held integers folded by `apply_held_ambiguities_to_carrier` always enter `layout.held_map`.

### 2.4 Verification

| check | pre-fix | post-fix |
|---|---|---|
| Probe epochs with exported position shift vs float | **0** (bit-identical) | **1 351** (mean **0.18 m** on FIX epochs) |
| Fixed-epoch truth median (cert-tight probe) | **0.037 m** (float already there) | **0.043 m** (subset); open-sky fixed RMS **0.040 m** |
| AllRMS delta vs float (full precision) | 0 | **+0.0005 m** (drift tail dominates) |
| `<50cm_full%` | 6.79 % | **6.79 %** (unchanged) |
| Unit test `test_held_integer_resolv_tightens_position_to_truth` | would fail | **pass** (5 m float error → **< 5 cm** after held-N re-solve) |

**pytest `tests/test_tc_fgo.py`: 31 passed** (+2 held-Jacobian / re-solve acceptance tests).

## 3. Probe ablation (run1, first 4 000 rover epochs → 3 900 scored)

Certificate tight thresholds: `--ar-cert-max-pos-sigma 0.15 --ar-cert-max-dd-pr-rms 1.0` (from WP12c open-sky telemetry via `_calibrate_cert.py`).

| stage | AllRMS (m) | FixRMS (m) | fix % | `<50cm_full%` | n_fix |
|---|---:|---:|---:|---:|---:|
| float no AR | 156.417 | — | 0 | 6.79 | 0 |
| cert-tight | 156.417 | 1.23 | 18.1 | 6.79 | 704 |
| cert-tight + subset | 156.417 | **0.70** | 15.1 | 6.79 | 589 |
| cert-tight + hold (full stack) | 156.417 | 0.71 | 14.7 | 6.79 | 572 |

Artifact: `ablation_4000_summary.json`, telemetry `probe_cert_tight_*_4000_telemetry.csv`, pre-fix snapshot `probe_cert_tight_4000_PREBUG_telemetry.csv`.

**Mechanism:** certificate passes **~700 / 3 900** epochs (open-sky + post-recovery quiet); these overlap the **20.8 %** of scored epochs already `< 50 cm` at float. Fixing them tighter does not add new epochs to the full-rover denominator (11 928). Subset-AR improves FixRMS purity; hold + DDPR trims fixes slightly (589 → 572) now that DDPR sees real position movement.

## 4. Full-length 3-run table

| method | run | cov % | AllRMS (m) | FixRMS (m) | fix % | `<50cm_full%` |
|---|---|---:|---:|---:|---:|---:|
| inuex35 README | run1 | 100 | 47.4 | 0.815 | 49.5 | **56.7** |
| inuex35 README | run2 | 100 | 32.1 | 0.277 | 60.8 | **69.9** |
| inuex35 README | run3 | 100 | 34.5 | 0.211 | 59.4 | **67.9** |
| WP7 RTK | run1 | 62.0 | 19.9 | 0.084 | 10.5 | 25.4 |
| WP7 RTK | run2 | 70.7 | 9.3 | 0.049 | 12.6 | 43.2 |
| WP7 RTK | run3 | 83.9 | 5.3 | 0.048 | 6.4 | 43.7 |
| WP12a TC-FGO | run1 | 97.9 | 235.2 | — | 0 | 7.1 |
| WP12a TC-FGO | run2 | 100 | 69.1 | — | 0 | 9.5 |
| WP12a TC-FGO | run3 | 99.9 | 27.3 | — | 0 | 7.7 |
| WP12b TC-FGO+AR (naive) | run1 | 97.9 | 228.6 | 23.7 | 80.2 | 6.8 |
| WP12b TC-FGO+AR (naive) | run2 | 100 | 69.3 | — | — | 9.4 |
| WP12b TC-FGO+AR (naive) | run3 | 99.9 | 27.3 | — | — | 2.7 |
| **WP12d cert-tight+hold** | **run1** | **97.9** | **235.2** | **60.1** | **5.3** | **7.0** |
| **WP12d cert-tight+hold** | **run2** | **100.0** | **69.1** | **2.47** | **10.3** | **10.2** |
| **WP12d cert-tight+hold** | **run3** | **99.9** | **26.9** | **9.90** | **27.0** | **4.6** |

Scores: `full_run{1,2,3}.score.json`. Run2 improves slightly vs pre-fix orphan (+0.2 pp `<50cm_full%`); run1 full-length AR **hurts** in the drift tail (FixRMS 60 m on 621 fixes). Run3 fix rate 27 % but many wrong fixes (FixRMS 9.9 m).

### 4.1 Run1 canyon (tow 188990–189070)

| metric | value |
|---|---:|
| n epochs | 401 |
| n FIX | **0** |
| RMS (m) | **118.4** |
| median (m) | 62.3 |

Artifact: `run1_canyon_188990_189070.json`. Certificate never passes here; AR offers nothing in the canyon.

## 5. Honest conclusion

WP12d closes the **AR feedback loop** (the last missing wiring between WP12b's validation stack and exported positions) but does not close the **campaign gap**. inuex35 wins with **49.5 % FIX at 0.815 m** spread across the timeline; we certify and fix **~15 %** of probe epochs, almost all in open-sky where float error is already **0.04 m**. Quality gating prevents the WP4/WP12b naive-AR catastrophe but cannot raise `<50cm_full%` without **absolute FIX supply** in degraded geometry (the same RTK-engine bottleneck WP5–WP10 documented).

## 6. Recommended next workstream — WP12e

**Dense RTK-FIX anchoring + cert AR:** feed WP6/WP7 RTK FIX epochs (not just the first-200 s anchor block) as `position_anchor` factors in TC-FGO, then apply cert-tight AR on the improved float. WP5 machinery exists; WP6 raised run2 FIX rate to 43 %. This attacks the actual deficit — *where* fixes happen — rather than polishing open-sky epochs that already score `< 50 cm`.

## 7. Deliverables

- `results/wp12d/WP12D_REPORT.md` (this file)
- `ablation_4000_summary.json`, probe/full `*.pos`, `*.score.json`, `*_telemetry.csv`
- Helpers: `_run_ablation.py`, `_fix_truth_analysis.py`, `_calibrate_cert.py`, `_canyon_analysis.py`
- Code: `python/gnss_gpu/tc_fgo.py` (Jacobian fix, `fixed_ambiguities` layout fallback)
- Tests: `tests/test_tc_fgo.py` — **31 passed**
