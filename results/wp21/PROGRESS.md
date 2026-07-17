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

## 2026-07-17 (WP21b, Phase B: make preint pay)

Spec: `internal_docs/task_wp21b_preint_payoff.md`. Continuing on the same
branch `agent/wp21-imu-preint`.

- Read `internal_docs/task_wp21b_preint_payoff.md`,
  `internal_docs/task_wp21_imu_preint.md`, `results/wp21/WP21_REPORT.md`,
  `python/gnss_gpu/imu_preintegration.py`,
  `python/gnss_gpu/pf_imu_preint_adapter.py`,
  `experiments/exp_wp21_imu_rbpf.py`, `python/gnss_gpu/pf_device_runtime.py`,
  `include/gnss_gpu/pf_device.h`, and the CUDA predict kernel
  (`src/particle_filter/pf_device.cu::pfd_predict_kernel`) to understand the
  exact `rbpf_velocity_kf` semantics: with `velocity_kf=true`, `mu_v` per
  particle blends toward the (single, global) `vel_guide` via
  `velocity_guide_alpha`, then position noise is drawn from
  `sigma_pos^2*I + dt^2*Sigma_v` (per-particle `Sigma_v`), and `Sigma_v`'s
  diagonal grows by `velocity_process_noise*dt` -- there is no existing
  mechanism to inject a specific (non-isotropic, freshly-modeled) `Sigma_v`
  each epoch other than through the generic scalar process-noise growth, or
  by writing `d_vcov` directly via the existing
  `get_particle_states`/`set_particle_states` (16-col: `[x,y,z,cb,
  mu_vx,mu_vy,mu_vz, Sigma_v(3x3 row-major)]`) round trip.
- **Item 3 (gravity-sign fix, done first since it only touches `imu.py` and
  is easy to verify in isolation)**: `IMUPredictor.__init__` now takes
  `gravity_convention: str | None = None` (`"positive_at_rest"` /
  `"negative_at_rest"`), auto-detected from the mean accel-Z of the first
  `align_samples` (default 100) IMU samples when `None`. PPC's `imu.csv`
  (+9.81 at rest) now correctly auto-selects `"positive_at_rest"` and
  subtracts gravity instead of adding a second `g`. Explicit override
  supported for datasets using the other convention (preserves the
  pre-existing behavior for any future caller relying on it). New tests in
  `tests/test_imu.py` (8 tests): autodetect both conventions, explicit
  override reproduces both the fixed and the original (buggy-on-PPC-data)
  behavior on demand, invalid-value rejection, plus
  `ComplementaryHeadingFilter` heading-variance tests (below). All pass.
- **Item 1 (heading-uncertainty -> predict noise)**: extended
  `gnss_gpu.imu.ComplementaryHeadingFilter` (backward-compatible additive
  change -- old call patterns produce byte-identical `.heading` output;
  `.heading_variance_rad2` is a new, ignorable attribute) with a heading
  variance state: grows via gyro-integration angular random walk
  (`sigma_gyro_radps_sqrthz^2 * dt` per `update_heading_gyro` sample, same
  noise-density convention as `ins_ekf.INSConfig`/`imu_preintegration`), and
  shrinks on `correct_heading_spp(spp_heading_rad, sigma_spp_heading_rad=...)`
  via the *exact* variance propagation of the fixed-gain complementary
  filter's own update formula (`Var_new = (1-alpha)^2*Var +
  alpha^2*sigma_meas^2`) -- an optional kwarg, so existing single-arg call
  sites are unaffected.
  `pf_imu_preint_adapter.ImuPreintPfGuide` gained `use_heading_uncertainty`
  (default False, preserves WP21 Phase A "preint-v1" numbers exactly) and
  `sigma_spp_pos_m` (documented default 30m, this repo's own characterized
  raw-SPP accuracy scale for this dataset/pipeline -- see WP21_REPORT.md
  Sec.7 -- not a hand-fit constant): converts consecutive-fix displacement
  into a per-epoch SPP-heading measurement sigma via
  `sqrt(2)*sigma_spp_pos_m/|displacement|` (capped at pi, i.e. fully
  uninformative below `min_heading_fix_disp_m`). When enabled,
  `close_segment` folds `sigma_pos_heading = |segment_displacement| *
  sqrt(heading_variance_rad2)` into `sigma_pos` in quadrature with the
  existing accel/gyro-covariance term, floored only by the small
  `sigma_pos_floor` (default 0.05m, numerical-stability floor, not a
  tuning knob).
- **Item 2 (per-particle velocity-KF feeding)**: added
  `ParticleFilterDeviceRuntime.set_velocity_covariance(cov_3x3)` --
  the "minimal Python-side setter" the spec calls for when the native API
  lacks one: round-trips `get/set_particle_states` (patching only the
  Sigma_v block, columns 7:16) and restores log-weights afterward (since
  `set_particle_states` resets them to uniform), broadcasting one
  covariance to every particle. No CUDA kernel edits.
  `ImuPreintPfGuide.close_segment` now also exposes
  `velocity_covariance_ecef` (the preintegration's delta_v covariance
  block, rotated body->ECEF) after each closed segment. New function
  `pf_imu_preint_adapter.imu_preint_predict_velocity_kf` calls
  `pf.set_velocity_covariance(guide.velocity_covariance_ecef)` then
  `pf.predict(..., rbpf_velocity_kf=True, velocity_guide_alpha=1.0)`, so
  `dt^2*Sigma_v` in the existing CUDA predict path carries this epoch's
  modeled accel/gyro velocity uncertainty (kept separate from item 1's
  heading term, which lives in the isotropic `sigma_pos` channel, to avoid
  double-counting).
- New/updated tests: `tests/test_imu.py` (new, 8 tests: gravity-sign fix +
  heading-variance tracking), `tests/test_pf_imu_preint_adapter.py` (+11
  tests: heading-uncertainty-disabled-by-default exact reproducibility,
  heading-uncertainty-enabled monotonicity + cross-track-lever-formula
  check, `_sigma_spp_heading_rad` edge cases, heading-variance shrink on a
  confident correction, `velocity_covariance_ecef` PSD/symmetry check,
  `imu_preint_predict_velocity_kf` covariance-injection + CV-fallback
  checks), `tests/test_pf_device_wrapper.py` (+4 GPU tests:
  `set_velocity_covariance` pre-init guard, non-finite rejection,
  broadcast + weight-preservation, and an end-to-end sanity check that the
  injected `Sigma_v` actually inflates the post-predict particle spread
  under `rbpf_velocity_kf=True`). All new tests confirmed to actually
  execute on this machine (not skipped) -- CUDA extension is available
  here.
- Ran all 52 new WP21/WP21b-specific tests (`test_imu.py`,
  `test_imu_preintegration.py`, `test_pf_imu_preint_adapter.py`,
  `test_pf_device_wrapper.py`): **all 52 pass.** Ran the broader
  `tests/test_*pf*.py tests/test_*rbpf*.py` subset (39 files, 201 tests):
  **197 passed, 3 skipped, 1 failed** -- the 1 failure is the same
  pre-existing, WP21-unrelated Windows-path-separator assertion documented
  in the Phase A section above, confirmed still present and still
  unrelated. **G1: PASS.**
- **Item 4 (ablation re-run)**: updated `experiments/exp_wp21_imu_rbpf.py`
  with a 4th arm (`preint_v1` = Phase A path unchanged, `preint_v2` = items
  1+2 combined); `spp_velocity_and_heading` now also returns the
  consecutive-fix displacement magnitude, threaded through to
  `ImuPreintPfGuide.close_segment(spp_displacement_m=...)`. Ran the full
  3000-epoch/100k-particle/GPS-only ablation twice (once as a stray
  backgrounded run that outlived its originating turn, once as a
  foreground re-run after the coordinator flagged the missing result) --
  **both produced numerically identical AllRMS/ESS/resample-rate figures**
  (only wall-clock differed from GPU contention between the two concurrent
  processes), confirming determinism. Results (`experiments/results/wp21/wp21b_full_tokyo_run2.csv`,
  gitignored per `experiments/results/*/` like all other experiment
  outputs in this repo):

  | arm | AllRMS [m] | mean ESS/N | resample rate | mean sigma_pos [m] | wall [s] |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | cv | 76.164 | 1.12e-05 | 1.000 | 2.0 (fixed) | 72.4 |
  | heuristic (gravity-fixed) | 11,728.284 | 1.00e-05 | 1.000 | 2.0 (fixed) | 74.6 |
  | preint_v1 (Phase A path, re-run) | 97.429 | 1.14e-05 | 1.000 | 0.300 (=floor) | 67.4 |
  | preint_v2 (items 1+2) | **75.332** | 1.35e-05 | 1.000 | 1.589 | 101.3 |

  `cv` and `preint_v1` reproduce Phase A's numbers exactly (76.164m,
  97.429m), confirming the re-run environment matches and that
  `use_heading_uncertainty=False` is a true no-op. `heuristic`'s AllRMS
  drops from Phase A's buggy 1,575,084m to 11,728m (99.26% reduction) after
  the gravity-sign fix, but is still far worse than cv (expected: open-loop
  accel double-integration with no bias estimate or wheel-speed correction
  on PPC). **preint_v2 AllRMS=75.332m beats cv's 76.164m by 1.09%**,
  achieved with only a small (0.05m) numerical-stability floor and no
  hand-tuned override -- **G2: PASS**. `mean_sigma_pos=1.589m` for
  preint_v2 (vs `0.300m` = the floor, for preint_v1) confirms the
  heading-uncertainty term is the actively dominant contributor, not the
  floor. Full honest diagnosis (why the margin is modest, not larger than
  Phase A's hand-tuned `floor=2.0` check's 3.85%, and concrete follow-up
  recommendations) written up in `results/wp21/WP21_REPORT.md` Phase B
  §B.6. **G3: PASS** (measured positive result with full diagnosis).
- Wrote the Phase B section of `results/wp21/WP21_REPORT.md` (§B.1-B.8):
  what was built, G1/G2/G3 results, the ablation table with Phase A
  numbers alongside, and deviations from the spec (Doppler-KF second table
  skipped as the spec's own "optional if time permits" item;
  `sigma_spp_pos_m` is a documented constant rather than a per-epoch
  DOP/residual estimate; heading uncertainty routed through `sigma_pos`
  rather than `Sigma_v` to avoid double-counting with item 2).
  **WP21b deliverables complete: G1/G2/G3 all PASS.**
