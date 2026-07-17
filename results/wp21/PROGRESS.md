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
- Implemented `python/gnss_gpu/pf_imu_preint_adapter.py` (deliverable 3, thin
  adapter, no changes to `pf_device_runtime.py` / CUDA): `ImuPreintPfGuide`
  closes a buffered `PreintegratedIMU` segment into a per-epoch ECEF
  velocity guide + `sigma_pos` (from the preintegration covariance's
  position block), feeding the *existing* `pf.predict(velocity=...,
  sigma_pos=..., rbpf_velocity_kf=False)` interface. Heading stays outside
  the particle state via the existing `imu.ComplementaryHeadingFilter`
  (roll/pitch=0 flat-vehicle approximation, documented, matches
  `experiments/ppc_imu_adapter.py`'s existing assumption for this same
  dataset). A single non-particle nominal-velocity accumulator bridges
  segments across epochs (needed to close `p_j = p_i + v_i*dt + 0.5*g*dt^2 +
  R_i@dp`), complementary-blended toward a GNSS-derived velocity each epoch
  to bound open-loop IMU drift over a multi-minute run.
- 7 unit tests in `tests/test_pf_imu_preint_adapter.py` (no GPU/CUDA
  required; a stub PF object). All pass, including a static-segment check
  that gravity is properly cancelled (guards against the failure mode where
  raw accel's ~9.81 m/s^2 gravity reading leaks into the displacement guide
  as a huge phantom climb).
- Implemented `experiments/exp_wp21_imu_rbpf.py` (deliverable 4): three-arm
  ablation (cv / heuristic-IMU / preint-IMU) on PPC Tokyo run2, scored with
  `experiments/score_vs_inuex35.py`. Debugged `robust_spp` needing a
  reasonable seed (`data["origin_ecef"]`, not the satellite-cloud centroid)
  to converge at all on PPC pseudoranges.
- Ran the full 3000-epoch / 100k-particle / GPS-only ablation (~3 min total).
  Results: cv AllRMS=76.164m, heuristic AllRMS=1,575,084m (diagnosed:
  `imu.IMUPredictor`'s gravity-compensation sign assumes accel_z~-9.81 at
  rest; PPC's accel_z~+9.81 at rest, so it *adds* a second g every sample --
  pre-existing dataset-convention bug in reused legacy code, not modified
  per non-goals), preint (default) AllRMS=97.429m. G1/G2 pass; G3 passes
  literally (preint beats the buggy heuristic by 4 orders of magnitude) but
  the more informative honest finding is preint-default underperforms plain
  CV -- diagnosed as `sigma_pos` floor (0.3m) too tight once heading
  uncertainty is accounted for (the 9x9 covariance only models accel/gyro
  white noise). Confirmed via a `sigma_pos_floor=2.0` sensitivity re-run:
  preint AllRMS drops to 73.349m, *beating* CV by ~4%. Full diagnosis +
  concrete Phase B recommendations written up in `WP21_REPORT.md`.
- Ran the existing PF test suite subset (`tests/test_*pf*`,
  `tests/test_*rbpf*`, 38 files, 188 tests): 184 passed, 3 skipped, 1
  pre-existing unrelated failure (Windows path-separator assertion in
  `test_eval_gsdc2023_ct_rbpf_fgo.py`, confirmed present without any WP21
  changes via `git stash`).
- Wrote `results/wp21/WP21_REPORT.md`. WP21 deliverables 1-5 complete.
