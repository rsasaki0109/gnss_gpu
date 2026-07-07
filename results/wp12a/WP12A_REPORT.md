# WP12a — Stabilize TC-FGO float estimator (diagnostics + recovery) — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Builds on WP11 (`python/gnss_gpu/tc_fgo.py`, `experiments/wp11_run_tc_fgo.py`). WP12a adds flag-guarded stabilization (`experiments/wp12_run_tc_fgo.py`), per-epoch telemetry, recovery, and a diagnostic pass on the run1 4000-epoch probe before adding more knobs.

**Headline result: gate FAIL — diagnostics pinpoint why ablations were inert; recovery fix is the first lever that materially bounds drift, but run1 full-length AllRMS remains 235 m (probe best 173 m), not the < 20 m bar.** This is a quantified partial win: recovery+anchor-fix cuts WP11 run1 km-blow-up to hundreds of metres and raises `<50cm_full%` from 0.2 % to 7.1 %, yet the float graph still lacks carrier-phase AR and a trustworthy absolute core through canyon geometry.

## 1. WP12a gate verdict

| run | gate criterion | result | pass? |
|---|---|---|:---:|
| run1 | AllRMS < 20 m **and** coverage 100 % | AllRMS **234.7 m**, coverage **97.9 %** (11 676 / 11 928) | **NO** |
| run2 | (same bar) | AllRMS **103.1 m**, coverage **100.0 %** | **NO** |
| run3 | (same bar) | AllRMS **27.6 m**, coverage **99.9 %** | **NO** |

**Verdict: WP12a gate not met.** Closest to the accuracy bar is run3 (27.6 m); run1 — the headline gate — is still two orders of magnitude off despite recovery firing 1 688 times on the full trajectory.

Best shipped config: `--recovery --anchor-fix` (plus adaptive Huber disable when raw DDPR RMS > 15 m, default-on in `tc_fgo.py`).

## 2. Diagnostic centerpiece — question (b): are DD-PR factors present when km-wrong?

**Yes — DD factors are usually present; they do not pull the trajectory back because Huber downweighting and the sliding-window marginal prior dominate the factor graph, and recovery was broken until this WP.**

Per-epoch telemetry on the **WP11-equivalent baseline** probe (`results/wp12a/probe_baseline_4000_telemetry.csv`, onset window tow ~188 038 / epoch ~2 842):

| segment | n ep | mean pos err (m) | mean DD factor count | mean DDPR RMS raw (m) | mean DDPR RMS Huber (m) | frac GNSS-solved |
|---|---:|---:|---:|---:|---:|---:|
| pre-onset (ep < 2 742) | 2 742 | 115.6 | **31.8** | 81.0 | **4.6** | 99.7 % |
| onset ±100 ep | 200 | **6 418** | **7.9** | **3 351** | **70.5** | 59.5 % |
| post-onset | 958 | **36 020** | **14.0** | **20 235** | **169** | 72.3 % |

At the first epoch with > 100 m error (ep **2 506**, tow **187 971**): **30 DD factors**, raw DDPR RMS **57.7 m**, Huber RMS **9.7 m** — geometry is usable, but the robust kernel has already collapsed the effective GNSS pull.

Code path — Huber caps per-factor contribution (`local_fgo.py`):

```928:934:python/gnss_gpu/local_fgo.py
def _huber_sqrt_weight(residual: np.ndarray, huber_k: float) -> float:
    ...
    return float(np.sqrt(float(huber_k) / max(norm, 1e-12)))
```

Applied in the TC window (`tc_fgo.py`):

```620:624:python/gnss_gpu/tc_fgo.py
        scale = _huber_sqrt_weight(residual / sigma, huber_k)
        residuals.append(residual * scale)
```

Meanwhile the marginal prior enters with **σ = 0.2 m** on position (`tc_fgo.py` `naive_marginalization_prior`, lines 1052–1070) — orders of magnitude tighter than saturated Huber-scaled DD rows at km misclosure. The LM therefore accepts IMU+marginal-locked states even when raw DDPR RMS is thousands of metres.

**Secondary onset mechanism:** DD factor **count** drops in the onset window (31.8 → 7.9 mean) because `_build_dd_measurements` elevation/SNR gating and `DDPseudorangeComputer` min-common-sats reject rows when the seed/geometry degrades (`wp11_run_tc_fgo.py:163–184`), not because satellites disappear from the sky.

**WP12a mitigations wired:**

| mechanism | effect on probe AllRMS |
|---|---:|
| Raw (un-Huber) DDPR RMS for bad-streak detection | exposes true misclosure; enables recovery trigger |
| Recovery DD-only LS + RTK seed, `max_shift_m` 5 000, clear marginal on fire | 1 100 m → **331 m** (recovery alone) |
| `+anchor-fix` | **173 m**, `<50cm_full%` 6.8 % |
| Adaptive Huber off when `last_dd_pr_rms_m > 15 m` | lets DD factors pull at large misclosure |

Full probe scoring: `results/wp12a/ablation_4000_summary.json`.

## 3. Recovery bug (recovery=0 with bad_streak_max=1 511)

**Root cause:** the recovery trigger **did fire** (bad-streak counter reached 1 511 in the pre-fix `full` ablation), but every recovery attempt **rejected** because `dd_pr_position_update_from_epoch` refused shifts larger than **50 m** while the IMU-drifting position was already km wrong:

```527:531:python/gnss_gpu/tc_fgo.py
    if shift <= float(max_shift_m) or rms_ok:
        stats["accepted"] = True
        return pos, stats
    return seed.copy(), stats
```

With the old default `max_shift_m=50`, WLS corrections of hundreds–thousands of metres were discarded; `recovery_events` stayed 0. The fix (`wp12_run_tc_fgo.py`):

1. Use **raw** DDPR RMS for bad-streak detection (Huber RMS was capped ~7–70 m even at km error).
2. Recovery DD-LS: RTK baseline seed when available, `max_shift_m=5000`, `prior_sigma_m=50`, up to 12 iterations.
3. On acceptance: **clear** `marginal_prior` / `marginal_sigmas` and reset bad-streak.
4. Skip marginal update on the recovery epoch.

Post-fix smoke: `recovery=5` on 500 epochs; full probe: **323–332** recovery events, `bad_streak_max=3`.

## 4. Ablation table (run1, first 4 000 rover epochs → 3 900 scored)

| config | AllRMS (m) | `<50cm_full%` | recovery events | notes |
|---|---:|---:|---:|---|
| WP11 baseline (pre-fix) | 1 100.6 | 0.16 | 0 | no stabilization flags |
| +bias | 1 098.4 | 0.06 | 0 | inert on AllRMS |
| +bias+anchor-fix | 1 096.0 | 6.84 | 0 | anchors help metric, not AllRMS |
| full stack (pre-fix) | 2 238.8 | 6.81 | **0 (bug)** | `quality_marginal` + broken recovery |
| **probe recovery** (post-fix) | **330.6** | 0.16 | 323 | recovery alone |
| **probe best** (`recovery+anchor-fix`) | **173.2** | **6.81** | 332 | best probe |

## 5. Full 3-run table (best config: `--recovery --anchor-fix`)

Scored with `experiments/score_vs_inuex35.py`. Comparison rows from campaign doc / WP11.

| method | run | n_scored | coverage% | AllRMS | `<50cm_full%` | ppc_official% |
|---|---|---:|---:|---:|---:|---:|
| inuex35 README | run1 | n/a | 100.0 | 47.40 | **56.7** | n/a |
| inuex35 README | run2 | n/a | 100.0 | 32.08 | **69.9** | n/a |
| inuex35 README | run3 | n/a | 100.0 | 34.52 | **67.9** | n/a |
| WP7/8 RTK baseline | run1 | 7 485 | 62.8 | 19.47 | **26.6** | 23.65 |
| WP7/8 RTK baseline | run2 | 6 466 | 70.7 | 9.33 | **43.2** | 48.05 |
| WP7/8 RTK baseline | run3 | 12 875 | 84.1 | 5.64 | **43.7** | 40.67 |
| WP11 TC-FGO float | run1 | 11 676 | 97.9 | 12 148 | **0.2** | 0.06 |
| WP11 TC-FGO float | run2 | 9 148 | 100.0 | 1 273 | **2.1** | 3.12 |
| WP11 TC-FGO float | run3 | 15 291 | 99.9 | 21 410 | **2.2** | 0.10 |
| **WP12a best** | run1 | 11 676 | **97.9** | **234.7** | **7.1** | 2.44 |
| **WP12a best** | run2 | 9 148 | **100.0** | **103.1** | **9.5** | 9.55 |
| **WP12a best** | run3 | 15 291 | **99.9** | **27.6** | **7.7** | 1.67 |

Artifacts: `results/wp12a/full_run{1,2,3}.pos`, `full_run{1,2,3}_score.json`, logs with recovery counts.

### 5.1 Run1 canyon (tow 188 990–189 070)

| segment | n_scored | coverage% | AllRMS | `<50cm%` |
|---|---:|---:|---:|---:|
| WP12a best | 401 | 3.36 | **120.9** | 0.0 |
| WP11 float (ref) | 401 | 3.36 | 15 664 | 0.0 |
| WP7 baseline (ref) | — | — | ~125.6 | — |

Recovery bounds canyon blow-up versus WP11, but anchor source is still WP10 RTK float (~100 m biases in NLOS), so canyon AllRMS remains triple-digit.

## 6. Execution

```powershell
$env:PYTHONPATH="python"
# 4000-epoch probe (diagnostic)
python experiments/wp12_run_tc_fgo.py --run tokyo/run1 --max-epochs 4000 `
  --export-pos results/wp12a/probe_best_4000.pos `
  --recovery --anchor-fix `
  --telemetry-csv results/wp12a/probe_best_4000_telemetry.csv

# Full 3-run (parallel)
python experiments/wp12_run_tc_fgo.py --run tokyo/run1 --export-pos results/wp12a/full_run1.pos --recovery --anchor-fix
python experiments/wp12_run_tc_fgo.py --run tokyo/run2 --export-pos results/wp12a/full_run2.pos `
  --baseline-pos results/wp10/sweep/run2/b0_baseline_no_wp10.pos --recovery --anchor-fix
python experiments/wp12_run_tc_fgo.py --run tokyo/run3 --export-pos results/wp12a/full_run3.pos `
  --baseline-pos results/wp10/sweep/run3/b0_baseline_no_wp10.pos --recovery --anchor-fix
```

| run | epochs | recovery events | wall (s) | log |
|---|---:|---:|---:|---|
| run1 probe 4k | 3 900 | 332 | 370 | `probe_best_4000.log` |
| run1 full | 11 676 | 1 688 | 1 058 | `full_run1.log` |
| run2 full | 9 148 | 542 | 815 | `full_run2.log` |
| run3 full | 15 291 | 1 353 | 1 330 | `full_run3.log` |

## 7. Honest gaps / blockers

1. **No DD carrier / AR** — float DDPR-only graph cannot observably fix the bias accumulation WP11 documented; recovery re-acquires code-level position but not cm float.
2. **Anchor quality** — `anchor-fix` pulls toward WP10 RTK, which is itself canyon-broken (WP8); helps `<50cm%` where RTK is good, cannot fix NLOS-dominated epochs.
3. **`quality_marginal` + `imu_gnss_quality_scale` regress** when combined with broken recovery (pre-fix full stack 2 239 m); even post-fix, scaling marginal σ from Huber-capped RMS understates misclosure — must use raw RMS (fixed for detection; quality_marginal still defaults to pre-fix behaviour if enabled).
4. **DD epoch builder is seed-static** — `build_dd_pr_epochs` uses initial RTK seed for gating (`wp11_run_tc_fgo.py:147–186`); after km drift the usable DD row count drops independent of sky view.
5. **Schur marginalization** still placeholder — diagonal 0.2 m prior is the wrong cross-epoch memory model vs inuex35 ISAM2 + DDPR-sanity FSM.

## 8. Recommended WP12b shape

Port AR stack in dependency order (from WP11 §7), but **keep recovery FSM as the float safety net**:

1. DD carrier-phase factors + ambiguity states in-window.
2. LAMBDA + ratio + subset-AR + DDPR cross-validation (reuse `local_fgo` paths).
3. Fix-and-hold with post-AR cost gate.
4. Promote IMU biases in-graph (WP12a `--optimize-imu-biases` already scaffolded).
5. Rebuild DD rows from current float position each epoch (or relinearize measurements) so canyon epochs retain factors.

Success criterion unchanged: run1 full-length **AllRMS < 20 m** at ~100 % coverage, then chase `<50cm_full%` vs inuex35.

## 9. Deliverables

- `results/wp12a/WP12A_REPORT.md` (this file)
- `results/wp12a/ablation_4000_summary.json` — ablation + probe table
- `results/wp12a/probe_*_telemetry.csv` + `*_onset_summary.json` — diagnostic artifacts
- `results/wp12a/full_run{1,2,3}.pos` + `*_score.json`
- `results/wp12a/full_run1_canyon_score.json`
- Code: `python/gnss_gpu/tc_fgo.py`, `experiments/wp12_run_tc_fgo.py`, `experiments/wp12a_diagnose_probe.py`, `experiments/wp12a_score_ablations.py`
- Tests: `tests/test_tc_fgo.py` — **11 passed**
