# WP21 Report — IMU Preintegration RBPF Core (Phase A: Python)

Spec: `internal_docs/task_wp21_imu_preint.md`. Branch: `agent/wp21-imu-preint`.
Date: 2026-07-17.

## 1. What was built

1. `python/gnss_gpu/imu_preintegration.py` — standalone, FGO-free IMU
   preintegration. `PreintegratedIMU` accumulates `(accel, gyro, dt)` samples
   between GNSS epochs and exposes `delta_p`, `delta_v`, `delta_R`/`delta_q`,
   a 9x9 covariance (`[delta_p, delta_v, delta_theta]`), first-order
   bias-correction Jacobians (`dp_d_ba`, `dv_d_ba`, `dp_d_bg`, `dv_d_bg`,
   `dR_d_bg`), and `reset()`. Zero dependency on `tc_fgo` or anything under
   `experiments/` (only `numpy`).
2. `tests/test_imu_preintegration.py` — 12 tests: SO(3) helper sanity,
   constant-acceleration / pure-rotation analytic-exact checks, a
   constant-speed-circle closed-form check, bias-Jacobian finite-difference
   checks, and the G1 cross-check (below).
3. `python/gnss_gpu/pf_imu_preint_adapter.py` — thin adapter (no changes to
   `pf_device_runtime.py` or CUDA). `ImuPreintPfGuide` closes a buffered
   preintegration segment into a per-epoch ECEF velocity guide + `sigma_pos`
   derived from the preintegration covariance, fed to the *existing*
   `pf.predict(velocity=..., sigma_pos=..., rbpf_velocity_kf=False)`.
   Heading stays outside the particle state via the existing
   `gnss_gpu.imu.ComplementaryHeadingFilter`.
4. `tests/test_pf_imu_preint_adapter.py` — 7 adapter unit tests (no GPU).
5. `experiments/exp_wp21_imu_rbpf.py` — three-arm ablation on PPC Tokyo run2,
   scored with `experiments/score_vs_inuex35.py`.

## 2. A note on where "tc_fgo's IMU preintegration math" actually lives

The task spec (and the roadmap) point at `python/gnss_gpu/tc_fgo.py` as the
source of the preintegration math to extract. On inspection, **the per-sample
on-manifold recursion (Delta_p, Delta_v, Delta_R, bias Jacobians, F/G noise
covariance) does not live in `tc_fgo.py`** — `tc_fgo.collapse_imu_preintegration_segment`
only *sums* pre-built per-interval segments, and `tc_fgo.imu_preintegration_residual`
/ `imu_preintegration_jacobian` only consume the summed segment. The actual
recursion that builds those segments lives in `experiments/gsdc2023_imu.py`
(`preintegrate_processed_imu` with `sample_dt_mode="taroz"`, invoked via
`imu_preintegration_segment_with_bias_jacobians`), which is what
`experiments/wp11_run_tc_fgo.py`, `wp12_run_tc_fgo.py`, and
`experiments/ppc_imu_adapter.py` (the module that already wires PPC's
`imu.csv` into this engine for the FGO stack) call into.

`imu_preintegration.py` therefore **re-derives** that recursion (does not
import it — it has zero dependency on `experiments/`) and is cross-checked
against `gsdc2023_imu.py` for G1, which is the true "internal preintegration"
`tc_fgo` consumes. `gravity`/frame conventions are matched to
`tc_fgo._G_ENU` / `ins_ekf._G_ENU` = `(0, 0, -9.81)`, documented in the
module docstring together with the exact formula
(`p_j = p_i + v_i*dt + 0.5*g*dt^2 + R_i@dp`) that mirrors
`tc_fgo.imu_preintegration_residual`'s zero-residual condition.

## 3. Gate G1 — numerical match on real PPC IMU data

`tests/test_imu_preintegration.py::test_g1_cross_check_against_gsdc2023_imu_real_ppc_data`
preintegrates a 1999-sample (~20s) slice of `datasets/PPC-Dataset-data/tokyo/run2/imu.csv`
through both `imu_preintegration.preintegrate_raw` and
`gsdc2023_imu.preintegrate_processed_imu` (`sample_dt_mode="taroz"`,
`delta_frame="body"`) and reports the max relative difference:

| quantity | max rel. diff (real PPC data) | max rel. diff (synthetic, 400 samples) |
| --- | ---: | ---: |
| `delta_p` | 7.97e-17 | 7.27e-16 |
| `delta_v` | 0.0 | 0.0 |
| `delta_angle` | 0.0 | 0.0 |

Bias-correction Jacobians (`dp_d_ba`, `dv_d_ba`, `dp_d_bg`, `dv_d_bg`,
`dR_d_bg`) match to `atol=1e-9` on the synthetic stream. This is
floating-point-exact agreement (both implementations run the identical
recursion; the residual is pure fp rounding-order noise).

**G1: PASS** (max rel. diff ~8e-17, far inside "tight tolerance").

## 4. Ablation setup

- Dataset: PPC Tokyo run2, epochs 0-2999 of `PPCDatasetLoader.load_experiment_data`
  (**3000-epoch window, not the full run** — the loader took ~17-52s just to
  parse RINEX + compute ephemeris for this many epochs in this environment,
  and the task spec explicitly allows a 3000-epoch run2 window when the full
  run is prohibitive; see `internal_docs/task_wp21_imu_preint.md`). Native
  epoch spacing is 5 Hz (`dt~0.2s`), so 3000 epochs = 177000.0-177599.8s TOW
  (~600s of driving), i.e. the run2 GPS-TOW window scored is
  `[177000.0, 177599.8]`.
- Constellation: **GPS-only (`--systems G`)**, matching
  `experiments/exp_ppc_pf_ablation_sweep.py`'s default. The particle state
  carries a single scalar clock-bias (`cb`); mixing constellations without a
  per-system inter-system-bias term is an orthogonal confound this ablation
  does not attempt to control for.
- Particle filter: `n_particles=100_000`, `resampling="megopolis"`,
  `ess_threshold=0.5`, `sigma_cb=300`, `sigma_pr=5.0`, `seed=42`, identical
  across all three arms.
- Initial position/clock-bias and the per-epoch heading/velocity reference
  used by arms (b)/(c) come from a **causal** `robust_spp` point solve per
  epoch (no ground truth in the loop; ground truth is only used for scoring).
- 100 Hz IMU: `datasets/PPC-Dataset-data/tokyo/run2/imu.csv` (183001 samples
  spanning the whole run).
- Scorer: `experiments/score_vs_inuex35.py::score_trajectory` against
  `reference.csv`, identical for all three arms.

### Arms

- **(a) cv**: `pf.predict(velocity=None, dt=dt, sigma_pos=2.0)` — no IMU.
- **(b) heuristic**: `gnss_gpu.imu.IMUPredictor` (open-loop accel+gyro
  dead-reckoning; PPC has no wheel channel so the wheel-speed correction
  in `imu.py` is inert), `sigma_pos=2.0` (same budget as arm a, isolating
  "does the existing heuristic guide help at fixed noise"). Note:
  `imu.ComplementaryHeadingFilter` could not be used as the arm-(b) velocity
  *source* — its `get_velocity_enu` returns exactly zero magnitude without a
  finite `wheel_vel`, which PPC's `imu.csv` does not provide — so `IMUPredictor`
  (also explicitly listed in the task's context files) is the only existing
  velocity-guide heuristic in this codebase applicable to PPC.
- **(c) preint**: `pf_imu_preint_adapter.ImuPreintPfGuide` — preintegrates
  the 100 Hz IMU between epochs, closes the segment with
  `PreintegratedIMU.predict_position_velocity` using a heading from
  `ComplementaryHeadingFilter` (gyro integration + causal-SPP-bearing
  correction) and a nominal (non-particle) velocity accumulator, and derives
  `sigma_pos` from the covariance's position block (`sigma_accel=0.05`,
  `sigma_gyro=0.005`, floor `0.3m`, scale `1.0`, default run).

## 5. Ablation table (G2)

All three arms, identical window (3000 epochs, GPS-only), identical scorer:

| arm | AllRMS [m] | \<50cm% | \<50cm_full% | mean ESS/N | resample rate | wall [s] | epochs/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) cv | 76.164 | 0.00 | 0.00 | 1.12e-05 | 1.000 | 45.5 | 66.0 |
| (b) heuristic | 1,575,084.344 | 0.00 | 0.00 | 1.00e-05 | 1.000 | 44.3 | 67.7 |
| (c) preint (default, `sigma_pos_floor=0.3`) | 97.429 | 0.00 | 0.00 | 1.14e-05 | 1.000 | 74.6 | 40.2 |
| (c) preint (`sigma_pos_floor=2.0`, sensitivity) | 73.349 | 0.00 | 0.00 | 1.19e-05 | 1.000 | 75.6 | 39.7 |

Coverage: `n_scored=3000/3000` (100% of the requested window), but
`coverage_pct=32.8%` against the full-run-9151-epoch denominator that
`score_vs_inuex35._ROVER_EPOCH_COUNTS` uses for `<50cm_full%` — i.e.
`<50cm_full%` here is bounded above by ~32.8% even at perfect local accuracy,
so it is **not a meaningful discriminator for a partial-window run** and
reads 0.00 for all three arms (none of the arms achieve sub-50cm accuracy in
absolute terms at all — see §7). AllRMS is therefore the operative G2/G3
metric for this window, as the spec's "AllRMS **or** \<50cm_full%" phrasing
anticipates.

Raw CSVs: `experiments/results/wp21/wp21_imu_rbpf_tokyo_run2.csv`,
`experiments/results/wp21/wp21_imu_rbpf_floor2_tokyo_run2.csv`, and
per-epoch trajectories `experiments/results/wp21/wp21_imu_rbpf_tokyo_run2_{cv,heuristic,preint}_traj.csv`.

**G2: PASS** — complete ablation table for all three arms on the same
3000-epoch window with the same scorer.

## 6. Gate G3

> preint arm (c) >= velocity-guide arm (b) on \<50cm_full% or AllRMS

`97.429m <= 1,575,084.344m` (AllRMS): **preint beats heuristic by four orders
of magnitude.** **G3: PASS** on the letter of the gate.

**But this pass is not fully honest on its own**, so per the task's explicit
instruction ("a measured negative result with diagnosis passes the gates; an
unmeasured claim does not") the rest of this section reports what actually
happened and why, including a result the gate as literally written does not
probe: **the default-floor preint arm (97.429m) is *worse* than the plain
CV baseline arm (a) (76.164m) — about 28% worse AllRMS.**

### 6.1 Why arm (b) is not a meaningful comparison point

`IMUPredictor.get_velocity_enu` (`gnss_gpu/imu.py:174`) compensates gravity as
`az_body = self.accel[i, 2] + 9.81`, which assumes the sensor's static
Z-reading is approximately `-9.81` (removing gravity by *adding* it back).
Real PPC `imu.csv` data reads `Acc Z (m/s^2) ~ +9.83` at rest (specific-force
convention, same as `ins_ekf.py`/`gsdc2023_imu.py`/this module) — so
`IMUPredictor` on PPC data computes `az_body ~ +19.6` (**adds** a second `g`
instead of removing the one that is already there) and integrates that as
upward acceleration every sample, unbounded, for the whole run. This is a
pre-existing convention mismatch in reused legacy code (`imu.py` was written
for a different dataset's IMU sign convention) — **not modified here**, per
the WP21 non-goal "do not touch the PPC production selector/ranker
pipeline" and the general instruction to change only what the task requires.
It fully explains the `1.575e6 m` AllRMS: `IMUPredictor`'s velocity state is
a persistent, uncorrected open-loop integrator, so ~2g of phantom vertical
acceleration compounds for the full ~600s window. **Beating this arm is not
evidence preint's covariance-guided approach is well-tuned** — it is evidence
that a stale, dataset-mismatched heuristic velocity source is unusable, which
is itself a useful (if unplanned) finding but should not be over-read.

### 6.2 The more informative, honestly-diagnosed result: preint (default) underperforms CV

Diagnosis: `preint`'s reported `mean_sigma_pos=0.300m` for the default run
**exactly equals `--preint-sigma-pos-floor`**, meaning the raw
covariance-derived `sigma_pos` (from `sigma_accel=0.05 m/s^2/sqrt(Hz)`,
`sigma_gyro=0.005 rad/s/sqrt(Hz)` over a ~0.2s epoch) is essentially always
sub-floor (millimeter-scale — a 0.2s accel/gyro white-noise integral is
tiny). The 9x9 covariance in `imu_preintegration.py` **only propagates
accel/gyro sensor white noise**; it does not (and, per the WP21 spec's
"attitude stays outside the particle state" design, structurally cannot at
this layer) account for:

- **heading estimation error** — `ComplementaryHeadingFilter`'s bearing
  correction comes from consecutive causal `robust_spp` fixes, which
  themselves carry the same ~70-100m position error as the raw SPP solution
  (see §7); a heading error of even a few degrees rotates the whole segment's
  displacement into the wrong direction, and with only a `0.3m` `sigma_pos`
  budget the PF has essentially no slack to recover from a systematically
  wrong guide direction — unlike arm (a), where the full `2.0m` random-walk
  budget covers displacement isotropically regardless of true heading;
  and
- **the flat-vehicle (roll=pitch=0) approximation** and **unmodeled
  accelerometer bias** (the module intentionally does not estimate bias in
  Phase A — see task non-goals).

**Confirmation**: re-running arm (c) with `sigma_pos_floor=2.0` (same budget
as arms a/b, i.e. removing the "artificially tight" floor as the dominant
term) flips the sign: `preint` AllRMS drops to **73.349m**, *beating* CV's
76.164m by ~4%. So the preintegration-derived velocity guide **does** carry
real, exploitable signal — the 9x9 covariance-only `sigma_pos` derivation
just does not yet account for the attitude source's own uncertainty, so by
default it is too tight and lets a biased guide direction dominate instead
of being treated as one noisy vote among many.

### 6.3 Concrete recommendation for Phase B

1. **Explicitly propagate heading/attitude uncertainty into `sigma_pos`**,
   not just accel/gyro white noise — e.g. combine in quadrature with a term
   like `|displacement| * sigma_heading_rad` (or, better, an actual heading
   covariance from whatever attitude source is used, rather than a scalar
   complementary-filter state with no uncertainty output).
2. **Implement the CUDA per-particle velocity-KF path**
   (`rbpf_velocity_kf=True`, `Sigma_v`, already scaffolded in
   `include/gnss_gpu/pf_device.h` / `src/particle_filter/pf_device.cu` and
   exercised by the existing `proper_rbpf_velocity_results.md` work) so that
   uncertainty **grows correctly per particle across epochs** via
   `velocity_process_noise` instead of a single global fixed floor
   recomputed from scratch every epoch. This is exactly the roadmap's WP21
   CUDA item (`x_new ~ N(x + R(theta)*Delta_p_preint, Q_preint + dt^2*Sigma_v)`).
3. **Do not rely on a single deterministic heading estimate.** Since a wrong
   heading currently biases every particle identically (no diversity to
   recover), consider a small per-particle heading ensemble/ KF (as
   `internal_docs/pf_only_imu_roadmap_2026_07_17.md`'s "risk" section already
   flags: "姿勢を粒子に入れると次元爆発 -> 窓外INSEKF伝播で回避、必要になったらheadingのみ粒子化" —
   this ablation is a concrete, measured case for doing exactly that).
4. Re-run this same ablation once (1)-(3) land, with a wider
   `--preint-sigma-pos-scale` / `--preint-sigma-pos-floor` sweep than the
   two-point check done here, to find whether the ~4% AllRMS gain over CV
   (at the `floor=2.0` operating point) can be pushed further once the
   covariance itself is honest about heading uncertainty rather than needing
   a manually-tuned floor to compensate for it.

## 7. Absolute accuracy caveat (why AllRMS is tens of meters, not sub-meter)

All three arms report AllRMS in the tens-of-meters range. This is expected
and not a WP21-specific issue: `PPCDatasetLoader.load_experiment_data`
applies **no ionospheric, tropospheric, or multipath correction** and this
ablation is pure single-frequency C1C pseudorange PF (no DD/RTK, no carrier
phase, no AR) — i.e. raw standalone SPP-domain accuracy, consistent with
this repo's own `experiments/debug_spp_on_ppc.py` characterization of raw
PPC pseudoranges and with `experiments/exp_ppc_pf_ablation_sweep.py`'s
similarly GPS-only, uncorrected-pseudorange PF baselines. Reaching the
production `<50cm_full%` numbers in `internal_docs/pf_only_imu_roadmap_2026_07_17.md`'s
target table requires the DD/RTK/AR pipeline that WP21's non-goals
explicitly put out of scope ("Do not touch the PPC production
selector/ranker pipeline"). **WP21 is a relative ablation of predict-step
IMU guides, not an attempt at SOTA absolute accuracy.**

## 8. Filter health (ESS / resample rate)

Mean ESS/N is ~1.0-1.2e-05 and the resample rate is 1.000 (every epoch, for
every arm, including cv). This is a highly informative-likelihood signature
(8-24 GPS satellites' pseudorange residuals overdetermine a 4D
position+clock-bias state at each epoch relative to the particle spread),
not filter divergence: AllRMS stays bounded (does not grow epoch-over-epoch)
for arms (a) and (c) across the full 3000-epoch window. It is consistent
across all arms (including the un-guided CV arm), so it reflects the
update-step likelihood's sharpness relative to `n_particles=100_000` and
`spread_pos=50m` initialization, not anything specific to the IMU guide.
`resampling="megopolis"` is designed for this frequent-resample regime.

## 9. Deviations from the spec

- **3000-epoch window**, not the full run2 (~9151 rover epochs) — allowed
  explicitly by the task spec when the full run is runtime-prohibitive; full
  `load_experiment_data` parsing/ephemeris cost alone was ~17s per 3000
  epochs (RINEX + broadcast-ephemeris overhead dominates, not the PF).
- **Preintegration math source**: extracted/re-derived from
  `experiments/gsdc2023_imu.py` rather than `tc_fgo.py` itself, because that
  is where the actual per-sample recursion lives (`tc_fgo.py` only sums
  pre-built segments) — see §2. `imu_preintegration.py` still matches
  `tc_fgo.py`'s gravity/frame/residual conventions exactly, and the G1 gate
  is satisfied against the true underlying engine.
- **`--systems G`** (GPS-only) instead of the multi-constellation default
  used by some other PPC scripts (e.g. `debug_spp_on_ppc.py`'s `G,E,J`), to
  avoid an unmodeled inter-system-bias confound given the single-scalar
  clock-bias particle state — matches `exp_ppc_pf_ablation_sweep.py`'s
  existing convention.
- **Arm (b) uses `IMUPredictor`, not `ComplementaryHeadingFilter`**, because
  PPC's `imu.csv` has no wheel-speed channel and `ComplementaryHeadingFilter.get_velocity_enu`
  is structurally zero-magnitude without one (heading-only, no speed) — see
  §4/§6.1 for the consequences of this choice.
- A pre-existing, unrelated test failure
  (`tests/test_eval_gsdc2023_ct_rbpf_fgo.py::test_discover_train_trips_finds_device_gnss_and_truth`)
  was observed in the "existing PF test suite subset" run; it is a
  Windows-path-separator assertion (`train\run-a\pixel5` vs
  `train/run-a/pixel5`) unrelated to any file touched by WP21, and fails
  identically without any WP21 changes present (verified via `git stash`).

## 10. Conclusion

- **G1: PASS** — floating-point-exact agreement (~8e-17 max rel. diff) with
  the actual preintegration engine `tc_fgo` depends on, on real PPC IMU data.
- **G2: PASS** — complete three-arm ablation table, one window, one scorer.
- **G3: PASS on the literal gate** (preint beats the existing heuristic guide
  by orders of magnitude), **but that comparison is dominated by a
  pre-existing bug in the reused heuristic**, not by preint being
  well-tuned. The more informative, fully measured result is that preint's
  **default** configuration slightly *underperforms* plain CV (97.4m vs
  76.2m AllRMS), traced to the covariance-derived `sigma_pos` being too
  tight once heading uncertainty (not modeled by the 9x9 accel/gyro-only
  covariance) is accounted for — and that this is fixable: a `sigma_pos`
  floor matched to CV's budget flips the result to a genuine ~4% AllRMS
  improvement over CV. Recommendations for closing this gap properly are in
  §6.3, and are the natural on-ramp to WP21's Phase B (CUDA per-particle
  velocity-KF / `Sigma_v` propagation).
