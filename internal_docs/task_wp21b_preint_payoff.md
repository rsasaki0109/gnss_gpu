# WP21b Task Spec — Make IMU Preintegration Pay (Phase B, Python-first)

Follow-up to WP21 (`internal_docs/task_wp21_imu_preint.md`, report `results/wp21/WP21_REPORT.md`).
WP21's measured outcome: preint default (97.4 m AllRMS) LOST to plain CV (76.2 m) on the
run2-3000ep window because `sigma_pos` was derived from accel/gyro white noise only
(floor-clamped to 0.3 m), ignoring heading-estimation error; a hand-tuned
`sigma_pos_floor=2.0` flipped it to 73.3 m (beats CV ~4%). Hand-tuned floors are not a
model. This task makes the preint arm win from *modeled* uncertainty.

## Work items (in priority order)

1. **Propagate heading uncertainty into the predict noise.**
   The dominant unmodeled error is heading. Track a heading variance (from `INSEKF`
   covariance if available, else a documented random-walk model on the
   `ComplementaryHeadingFilter`/gyro-integration error) and map it through the lever of
   per-epoch displacement: a heading error of sigma_theta over displacement d contributes
   ~ (d * sigma_theta)^2 of cross-track position variance. Fold this into the per-epoch
   process noise handed to predict. The 0.3 m hard floor must become a derived quantity
   (keep a small numerical floor like 0.05 m for stability, documented).
2. **Use the per-particle velocity-KF path, not just a scalar guide.**
   The device predict already supports covariance-aware propagation
   (`x_new ~ N(x + mu_v*dt, sigma_pos^2*I + dt^2*Sigma_v)`, see `include/gnss_gpu/pf_device.h`
   and how `pf_device_runtime.py` maintains `{mu_v, Sigma_v}`). Feed the preintegrated
   Delta_v (rotated into the nav frame) and its covariance into the per-particle velocity
   KF between GNSS epochs (Python-side state manipulation is fine; do NOT modify CUDA
   kernels — if the runtime API lacks a needed setter, add a minimal Python-side one).
3. **Fix the pre-existing gravity-sign bug in `IMUPredictor`** (`python/gnss_gpu/imu.py`):
   it assumes accel_z ~= -9.81 at rest while PPC logs +9.81, so it adds ~2g. Make the
   convention explicit (auto-detect from the first seconds at rest, or a documented
   parameter), add a regression test, and re-run arm (b) with the fix. This is now in
   scope because arm (b) is a baseline.
4. **Re-run the ablation** (`experiments/exp_wp21_imu_rbpf.py`) on the SAME run2-3000ep
   window, same scorer, arms: (a) CV, (b) heuristic-fixed, (c) preint-v2 (items 1+2).
   Report the old preint-v1 numbers alongside for continuity. Include ESS/resample-rate.
   Optional if time permits: enable the existing Doppler KF update
   (`pf_device_doppler_kf_update`) in all arms symmetrically and report a second table.

## Gates

- G1: WP21 unit tests still pass; new tests for heading-variance propagation, velocity-KF
  feeding, and the IMUPredictor fix.
- G2: preint-v2 beats CV on AllRMS on the same window **without any hand-tuned floor**
  (all noise terms derived from modeled uncertainties; small documented numerical floor OK).
- G3: updated `results/wp21/WP21_REPORT.md` (append a Phase B section) with the honest
  read; a measured negative with diagnosis is acceptable, an unmeasured claim is not.

## Constraints (unchanged from WP21)

- No FGO at runtime; no CUDA kernel edits; don't touch the PPC production selector.
- Continue on branch `agent/wp21-imu-preint`. Commit as rsasaki0109. No push, no PR.
- Append milestones to `results/wp21/PROGRESS.md` as you go.
