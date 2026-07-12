# WP11 Progress — TC-FGO float skeleton

Date: 2026-07-06.

**Full report (work item 3):** [`WP11_REPORT.md`](WP11_REPORT.md) — 3-run scoring, gate verdict (**FAIL**), divergence diagnosis, WP12 recommendation.

## What was built

| Deliverable | Path | Status |
|---|---|---|
| TC-FGO core module | `python/gnss_gpu/tc_fgo.py` | done |
| CLI runner | `experiments/wp11_run_tc_fgo.py` | done |
| Unit tests | `tests/test_tc_fgo.py` (6 tests) | passing |
| Smoke trajectory | `results/wp11/smoke_run1_2000.pos` | done |

### `tc_fgo.py` capabilities

- Per-epoch sliding-window states: antenna **position + velocity in base-anchored ENU** (6 DOF / epoch in the LM solve).
- **Attitude quaternion + accel/gyro biases** propagated alongside via `INSEKF` mechanization; they enter IMU factors at the linearization point only (not LM states in WP11).
- IMU preintegration factors using `imu_preintegration_segment_with_bias_jacobians` collapsed segments, gravity in ENU, lever-arm `(0.31, 0, 0.55)` m body→antenna.
- **DD pseudorange** factors reusing `local_fgo` DD geometry (`DDPseudorangeEpoch` + `_dd_expected_and_jacobian_m`).
- NHC (lateral/vertical body velocity) and ZUPT factors, flag-guarded per epoch.
- Sliding fixed-lag window (default 5 epochs ≈ 1 s @ 5 Hz) with numpy Levenberg–Marquardt.
- Naive marginalization: diagonal prior on the new front state after dropping the oldest epoch.
- IMU-only propagation between GNSS epochs → **every rover epoch gets a `.pos` row**.

### Runner

- Two-phase init: static RTK FIX window + `INSEKF.feed_imu_for_alignment`; phase-2 yaw when consecutive RTK FIX speed exceeds 1 m/s.
- Seeds / geometry from PPC `tokyo/run1`, baseline `.pos` from WP10 `a0_baseline_no_wp10.pos`.
- Status column **5 = float** in exported `.pos`.

## Design decisions

### State parameterization

**Chosen:** optimize antenna `pos+vel` (6D) inside the window; carry `q, b_a, b_g` outside the solver and refresh via INS propagation between windows.

**Trade-off:** window stays small (`6n` vs `15n`), reuses verified `INSEKF` bias/attitude transients, but attitude/bias are frozen during each LM solve — acceptable for WP11 float skeleton; WP12+ can promote biases (or full error-state) into the graph.

### DD vs undifferenced pseudorange

**Chosen:** **DD pseudorange** against `base.obs` (same path as `solve_ppc_segment_multifamily_fgo.py`).

**Rationale:** eliminates receiver clock, proven in `local_fgo`, simpler wiring than undiff + per-epoch clock states. Undiff left for a future fallback if DD count collapses in canyon gaps.

### Marginalization

Naive drop + diagonal prior (`marginal_pos_sigma=0.2 m`, `marginal_vel_sigma=0.3 m/s`) from the previous window's second state. No Schur complement yet.

## Tests

```
pytest tests/test_tc_fgo.py  →  6 passed
```

Coverage: IMU factor FD Jacobian, NHC/ZUPT, lever-arm offset, marginalization prior, synthetic constant-velocity E2E (<0.1 m recovery).

## Smoke validation — `tokyo/run1`, first 2000 epochs

Command:

```powershell
$env:PYTHONPATH="python"
python experiments/wp11_run_tc_fgo.py --run tokyo/run1 --max-epochs 2000 `
  --export-pos results/wp11/smoke_run1_2000.pos
python experiments/score_vs_inuex35.py --traj results/wp11/smoke_run1_2000.pos `
  --city tokyo --run run1 --format pos
```

Runtime: ~312 s (DD prep + 2000 sequential window solves).

| Metric | WP11 smoke | Notes |
|---|---:|---|
| Processed epochs | 2000 | `--max-epochs 2000` |
| **Span coverage** | **100%** | 2000/2000 `.pos` rows |
| Scorer `coverage%` (vs full 11928) | 16.8% | expected for partial export |
| **AllRMS** (2000 scored epochs) | **8.73 m** | |
| WP3b backbone (full run 2D) | 85.82 m | campaign reference |
| fix% | 0.0% | float-only (status 5) |
| `<50cm%` | 0.9% | no AR yet |
| Phase-2 transition | epoch **75** | ~15 s; yaw from moving RTK FIX chain |

**Smoke bar:** pipeline end-to-end ✓, span coverage 100% ✓, AllRMS materially below WP3b 85.82 m ✓ (**8.73 m** on this span).

### Bug found and fixed during smoke

Initial smoke had `phase2@1999` because phase-2 velocity used only the first five **static** RTK fixes (always <1 m/s). Fixed to scan **consecutive RTK FIX pairs** along the full fix timeline. First broken run: AllRMS **256.8 m**; after fix: **8.73 m**.

## Known gaps (WP12+)

- No DD carrier / ambiguity / LAMBDA (float only).
- Attitude/bias not in LM state vector; no combined IMU bias random-walk factors between epochs.
- Marginalization is naive; no GNSS-quality-scaled IMU covariance inflation (inuex35 throttles IMU trust from DDPR residuals).
- No Doppler velocity priors, no PLATEAU NLOS weighting.
- Per-epoch solve is Python-sequential (~0.15 s/epoch here); full 12k-epoch runs need profiling / chunking.
- ENU origin anchored at base station position; could re-anchor at first static fix for numerical conditioning.
- Multi-run scoring completed in work item 3 — see [`WP11_REPORT.md`](WP11_REPORT.md). Full-length runs diverge (run1 AllRMS **12 148 m**); smoke-only bar was insufficient.

## Blockers

Long-run float drift (§4 of report) — WP12 AR stack + bias/robustness layers required before campaign metrics are meaningful.
