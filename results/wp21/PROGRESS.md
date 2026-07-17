# WP21 Progress Log

Live progress signal for WP21 (IMU Preintegration RBPF Core, Phase A: Python).
Spec: `internal_docs/task_wp21_imu_preint.md`. Branch: `agent/wp21-imu-preint`.

## 2026-07-17

- Read context: `internal_docs/pf_only_imu_roadmap_2026_07_17.md` (WP21
  section), `python/gnss_gpu/pf_device_runtime.py`, `include/gnss_gpu/pf_device.h`,
  `src/particle_filter/pf_device.cu` (`pfd_predict_kernel` -- confirmed the
  device predict already supports a velocity guide + isotropic
  `sigma_pos`/`velocity_process_noise`, so no CUDA kernel changes are
  needed), `python/gnss_gpu/tc_fgo.py`, `python/gnss_gpu/imu.py`,
  `python/gnss_gpu/ins_ekf.py`, `python/gnss_gpu/io/ppc.py`,
  `experiments/score_vs_inuex35.py`, `internal_docs/proper_rbpf_velocity_results.md`.
- **Finding**: the per-sample on-manifold preintegration recursion (Delta_p,
  Delta_v, Delta_R, bias Jacobians, F/G noise-covariance propagation) that
  `tc_fgo.py`'s sliding window actually consumes does **not** live in
  `tc_fgo.py` itself -- `tc_fgo.collapse_imu_preintegration_segment` only
  *sums* pre-built per-interval segments. The recursion that builds those
  segments lives in `experiments/gsdc2023_imu.py`
  (`preintegrate_processed_imu` with `sample_dt_mode="taroz"`, called via
  `imu_preintegration_segment_with_bias_jacobians`), which is what
  `experiments/wp11_run_tc_fgo.py` / `wp12_run_tc_fgo.py` /
  `ppc_imu_adapter.py` wire into `tc_fgo`. Treated this as the true "existing
  IMU preintegration math" referenced in the task spec and cross-checked
  against it for G1 (documented in the new module's docstring).
- Implemented `python/gnss_gpu/imu_preintegration.py`: standalone `PreintegratedIMU`
  class + `preintegrate_raw()` function, ported (re-derived, not imported)
  from the `gsdc2023_imu.py` recursion. Zero FGO/experiments dependency
  (only `numpy`). Gravity/frame convention matches `tc_fgo._G_ENU` /
  `ins_ekf._G_ENU` = `(0,0,-9.81)`, documented in the module docstring.
- Implemented `tests/test_imu_preintegration.py`: 12 tests -- SO(3) helper
  sanity, constant-acceleration/pure-rotation analytic exact-match,
  constant-speed-circle closed-form match, bias-correction Jacobian
  finite-difference checks (accel and gyro), and G1 cross-checks against
  `experiments/gsdc2023_imu.py` on both synthetic samples and a real PPC
  Tokyo run2 `imu.csv` slice (2000 samples, ~20s).
  **All 12 pass.** G1 result: max relative difference on real PPC data is
  ~8e-17 (delta_p), 0.0 (delta_v, delta_angle) -- i.e. floating-point-exact
  agreement with the engine `tc_fgo` actually uses. **G1: PASS.**
- Next: PF integration adapter (deliverable 3), then the 3-arm ablation
  experiment (deliverable 4) on PPC Tokyo run2.
