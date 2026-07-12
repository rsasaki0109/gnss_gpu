# WP11 — Tightly-coupled GNSS+IMU float FGO skeleton (inuex35 port) — tokyo run1-3

Workspace: `C:\Users\rsasa\Workspace\old\gnss_gpu`. Skeleton built in work items 1–2 (`python/gnss_gpu/tc_fgo.py`, `experiments/wp11_run_tc_fgo.py`, `tests/test_tc_fgo.py`); this report covers work item 3 (full-length 3-run scoring + gate).

**Headline result: gate FAIL — skeleton is end-to-end and fast, but long-run float drift diverges catastrophically.** The smoke bar (first 2000 epochs of run1: AllRMS **8.73 m**, span coverage 100 %) passed, yet full-length runs blow up to **1.3–21 km** AllRMS after progressive IMU-dominated drift without recovery. No run meets the WP11 gate (run1 **AllRMS < 20 m AND coverage 100 %**). This is an honest negative at the campaign-metric level, not a plumbing failure: the pipeline runs, exports `.pos` for nearly every rover epoch, and scores — the estimator simply does not yet hold position over multi-km trajectories.

## 1. WP11 gate verdict

| run | gate criterion | result | pass? |
|---|---|---|:---:|
| run1 | AllRMS < 20 m **and** coverage 100 % | AllRMS **12 148 m**, coverage **97.9 %** (11 676 / 11 928) | **NO** |
| run2 | (same bar, run1 is headline) | AllRMS **1 273 m**, coverage **100.0 %** (9 148 / 9 151) | **NO** |
| run3 | (same bar) | AllRMS **21 410 m**, coverage **99.9 %** (15 291 / 15 301) | **NO** |

**Verdict: WP11 gate not met on any run.** Closest partial pass is run2 coverage (essentially 100 %); accuracy is orders of magnitude off everywhere.

## 2. Full 3-run metric table

Scored with `experiments/score_vs_inuex35.py` on exported `.pos` (status 5 = float, no FIX epochs). Comparison rows: inuex35 README targets (`internal_docs/inuex35_tc_fgo_benchmark.md`) and WP7/8 RTK baseline (`results/wp7/final/run{N}/wp7_baseline_score.json`, `--preset low-cost --max-pos-jump-rate 2.3`).

| method | run | n_scored | coverage% | AllRMS | FixRMS | fix% | `<50cm%` | `<50cm_full%` | ppc_official% |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inuex35 README (external) | run1 | n/a | 100.0 | 47.40 | 0.815 | 49.5 | 56.7 | **56.7** | n/a |
| inuex35 README (external) | run2 | n/a | 100.0 | 32.08 | 0.277 | 60.8 | 69.9 | **69.9** | n/a |
| inuex35 README (external) | run3 | n/a | 100.0 | 34.52 | 0.211 | 59.4 | 67.9 | **67.9** | n/a |
| WP7/8 RTK baseline | run1 | 7 485 | 62.8 | 19.47 | 0.313 | 15.1 | 42.5 | **26.6** | 23.65 |
| WP7/8 RTK baseline | run2 | 6 466 | 70.7 | 9.33 | 0.125 | 13.0 | 61.1 | **43.2** | 48.05 |
| WP7/8 RTK baseline | run3 | 12 875 | 84.1 | 5.64 | 0.081 | 5.2 | 51.9 | **43.7** | 40.67 |
| **WP11 TC-FGO float (this WP)** | run1 | 11 676 | **97.9** | **12 148** | n/a | 0.0 | 0.2 | **0.2** | 0.06 |
| **WP11 TC-FGO float (this WP)** | run2 | 9 148 | **100.0** | **1 273** | n/a | 0.0 | 2.1 | **2.1** | 3.12 |
| **WP11 TC-FGO float (this WP)** | run3 | 15 291 | **99.9** | **21 410** | n/a | 0.0 | 2.2 | **2.2** | 0.10 |

Score JSON artifacts: `results/wp11/full_run{1,2,3}_score.json`. Trajectories: `results/wp11/full_run{1,2,3}.pos`.

### 2.1 Run1 canyon segment (tow 188 990–189 070)

Same window and scorer path as `experiments/sweep_libgnss_rtk_wp7.py::score_segment` (401 epochs in range).

| segment | n_scored | coverage% | AllRMS | fix% | `<50cm%` |
|---|---:|---:|---:|---:|---:|
| WP11 TC-FGO float, canyon | 401 | 3.36 | **15 664** | 0.0 | 0.0 |
| WP7 baseline, full run1 (ref) | — | — | canyon ~125.6 m (WP9 §2) | 0.0 | — |

Canyon is scored, but the float skeleton has already diverged (~20 km error) before this window (first error > 100 m at tow **188 038**, epoch ~2 842), so canyon numbers reflect runaway drift, not canyon-specific behaviour.

## 3. Execution (work item 3)

Commands (parallel, `--max-epochs` omitted = full length; per-run RTK baseline for two-phase init):

```powershell
$env:PYTHONPATH="python"
python experiments/wp11_run_tc_fgo.py --run tokyo/run1 --export-pos results/wp11/full_run1.pos `
  --baseline-pos results/wp10/sweep/run1/a0_baseline_no_wp10.pos
python experiments/wp11_run_tc_fgo.py --run tokyo/run2 --export-pos results/wp11/full_run2.pos `
  --baseline-pos results/wp10/sweep/run2/b0_baseline_no_wp10.pos
python experiments/wp11_run_tc_fgo.py --run tokyo/run3 --export-pos results/wp11/full_run3.pos `
  --baseline-pos results/wp10/sweep/run3/b0_baseline_no_wp10.pos
```

The runner already exposes `--baseline-pos` (default is run1 WP10 path); no code change was required for run2/run3.

| run | epochs exported | phase-2 @ | wall time | s/epoch | log |
|---|---:|---|---:|---:|---|
| run1 | 11 676 | 75 | **1 263 s** (~21 min) | 0.11 | `results/wp11/full_run1.log` |
| run2 | 9 148 | 116 | **1 131 s** (~19 min) | 0.12 | `results/wp11/full_run2.log` |
| run3 | 15 291 | 187 | **1 775 s** (~30 min) | 0.12 | `results/wp11/full_run3.log` |

All three exited **0**. Runtime was **faster** than the ~0.15 s/epoch smoke estimate (no slowdown issue). Epoch counts are slightly below PPC denominators (run1 −252, run2 −3, run3 −10) because `load_ppc_window_geometry` drops epochs without usable sat geometry — a pre-existing loader behaviour, not a TC-FGO export bug.

**Failures encountered:** none at the process level. **Accuracy failure:** progressive divergence on all runs (see §4).

## 4. Divergence diagnosis

Per-epoch error vs reference (`score_vs_inuex35` alignment):

| run | first 1 000 ep RMS | first 2 000 ep RMS | first err > 100 m (tow / idx) | full AllRMS |
|---|---:|---:|---|---:|
| run1 | 1.9 m | **8.7 m** (matches smoke) | 188 038 / 2 842 | 12 148 m |
| run2 | 12.4 m | 12.0 m | 177 907 / 4 537 | 1 273 m |
| run3 | 3.7 m | 9.6 m | 180 656 / 5 981 | 21 410 m |

Error grows monotonically in 1 000-epoch blocks after ~epoch 2 500 on run1 (e.g. block 3 000–3 999: RMS **2 492 m**). Root cause is structural, not a single bad epoch:

1. **No absolute position anchor when DD count is weak** — DD pseudorange is the only GNSS pull; when geometry degrades (canyon / NLOS), the 5-epoch window cannot recover a ~km bias.
2. **Attitude and IMU biases frozen inside each LM solve** — propagated outside the window; small bias mis-estimation compounds over minutes of IMU-dominated motion.
3. **Naive marginalization** (diagonal σ = 0.2 m pos / 0.3 m/s vel on the front state) locks in a drifting solution with no Schur complement or GNSS-quality scaling (contrast inuex35's DDPR-residual IMU inflation).
4. **No recovery FSM** — unlike inuex35's DDPR-sanity anchor / ambiguity wipe / IMU-predicted fallback (WP8–J documented why cross-epoch memory matters in our RTK filter; TC-FGO has the IMU propagation path but no divergence detector).

Smoke passed because it stopped at epoch 2 000, **before** run1's divergence onset at ~2 842. The gate was therefore optimistic for a float-only skeleton without AR or robustness layers.

**Cheap fix assessment:** loosening marginal σ or adding a periodic RTK/DDPR anchor would be a band-aid without AR and bias states in-graph; not attempted in WP11 item 3 per "document honestly" when the gap is architectural.

## 5. Design decisions recap (work items 1–2)

| choice | decision | rationale |
|---|---|---|
| State in LM | antenna pos+vel (6D/epoch) | small window; attitude/bias via `INSEKF` outside solve |
| GNSS | DD pseudorange vs `base.obs` | reuses proven `local_fgo` path; no receiver clock states |
| IMU | collapsed preintegration segments + NHC/ZUPT | lever-arm (0.31, 0, 0.55) m; gravity in ENU |
| Window | 5 epochs (~1 s), numpy LM | matches inuex35 lag order-of-magnitude; Python-sequential |
| Marginalization | naive drop + diagonal prior | placeholder — identified as drift source in §4 |
| Init | two-phase static RTK FIX + moving heading | same speed threshold (1 m/s) as inuex35 |
| Export | status 5 (float), IMU fill for all rover epochs | 97–100 % coverage achieved; accuracy did not |

Unit tests: `pytest tests/test_tc_fgo.py` → **6 passed** (unchanged).

## 6. Honest gaps vs inuex35

| inuex35 capability | WP11 status |
|---|---|
| DD carrier phase + ambiguity states | **missing** |
| LAMBDA + subset-AR + fix-and-hold | **missing** |
| DDPR cross-validation at fixed position | **missing** |
| IMU bias in graph + bias random-walk | **missing** (biases propagated, not optimized) |
| GNSS-quality-scaled IMU covariance | **missing** |
| Doppler velocity priors | **missing** |
| PLATEAU NLOS weighting in DD factors | **missing** |
| Recovery FSM (DDPR anchor, ambiguity wipe) | **missing** |
| Schur / proper marginalization | **missing** |

The campaign insight from WP8–J still holds: **tight coupling's value is cross-epoch memory through bad GNSS**. WP11 wires IMU propagation for coverage but lacks the AR stack and robustness that make inuex35's memory trustworthy.

## 7. Recommended WP12 shape

Port the inuex35 AR and validation stack onto this graph, in dependency order:

1. **DD carrier phase factors** — extend `local_fgo` DD geometry to phase; per-sat ambiguity `Double` states in the window (or held integers via constant folding for perf).
2. **LAMBDA + ratio test** — reuse existing `local_fgo` LAMBDA path; wire to raw PPC DD streams (WP4 showed AR on a biased seed is useless without a good float — WP11 float must be stabilized first, likely via bias states + IMU inflation).
3. **Subset-AR** — drop worst residuals, keep best ratio (inuex35 `optimize/ar.py` pattern).
4. **DDPR cross-validation** — reject fixes that worsen code residuals at the fixed position.
5. **Fix-and-hold** — `ar_mode=3` style with post-AR cost gate.
6. **Parallel stabilization** (can land with or before AR): promote **accel/gyro biases** into LM state; **DDPR-residual IMU σ inflation**; optional **Doppler** body-velocity prior; **PLATEAU** weights on DD factors.

Success criterion for WP12: run1 full-length **AllRMS < 20 m** at 100 % coverage (WP11 gate), then chase `<50cm_full%` vs inuex35.

## 8. Deliverables

- `results/wp11/WP11_REPORT.md` (this file)
- `results/wp11/full_run{1,2,3}.pos` — full trajectories
- `results/wp11/full_run{1,2,3}_score.json` — scorer output
- `results/wp11/full_run{1,2,3}.log` — runner logs with wall times
- `results/wp11/WP11_PROGRESS.md` — build notes (updated with report pointer)
- `results/wp11/smoke_run1_2000.pos` — smoke artifact from work item 2
