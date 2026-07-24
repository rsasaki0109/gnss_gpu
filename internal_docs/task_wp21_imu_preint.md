# WP21 Task Spec — IMU Preintegration RBPF Core (Phase A: Python)

Part of the PF-only roadmap (`internal_docs/pf_only_imu_roadmap_2026_07_17.md`).
Goal of the roadmap: push the particle filter WITHOUT FGO; IMU enters as the
per-particle motion/propagation model. This task is the first implementation slice.

## Context (read these first)

- `internal_docs/pf_only_imu_roadmap_2026_07_17.md` — the roadmap this serves (WP21 section).
- `python/gnss_gpu/pf_device_runtime.py` — production device-resident RBPF runtime
  (particles = {x,y,z,cb}, per-particle velocity KF {mu_v, Sigma_v}).
- `include/gnss_gpu/pf_device.h` — device API; note the covariance-aware predict
  (`x_new ~ N(x + mu_v*dt, sigma_pos^2*I + dt^2*Sigma_v)`).
- `python/gnss_gpu/tc_fgo.py` — contains the existing IMU preintegration math
  (Delta_p, Delta_v, Delta_R, covariance propagation) buried inside the FGO stack.
- `python/gnss_gpu/imu.py` — `load_imu_csv`, `ComplementaryHeadingFilter`, `IMUPredictor`.
- `python/gnss_gpu/ins_ekf.py` — INS EKF used for out-of-window attitude.
- `python/gnss_gpu/io/ppc.py` — PPC dataset loader (Tokyo run1/2/3 include 100 Hz IMU
  at `datasets/PPC-Dataset-data/tokyo/run{1,2,3}/imu.csv`).
- `experiments/score_vs_inuex35.py` — the dual-metric scorer (AllRMS / <50cm_full% etc.).
- `internal_docs/proper_rbpf_velocity_results.md` — how the current velocity-KF RBPF
  was evaluated (reuse its evaluation style).

## Non-goals / hard constraints

- **No FGO at runtime.** The new preintegration module must NOT import `tc_fgo`
  (extract/port the math, do not wrap it). Tests MAY import tc_fgo to cross-check numbers.
- Do not modify CUDA kernels in this phase (Phase B will). Work through the existing
  device predict interface (velocity guide mean + covariance) from Python.
- Do not touch the PPC production selector/ranker pipeline.
- Windows environment. Discover how experiments are run (look at README/internal_docs;
  prefer the same python env other experiments use). Do not rebuild the CUDA extension
  unless something is broken.

## Deliverables

1. **`python/gnss_gpu/imu_preintegration.py`** — standalone, FGO-free preintegration:
   - class `PreintegratedIMU`: accumulate (accel, gyro, dt) samples between GNSS epochs;
     expose Delta_p, Delta_v, Delta_R (or quaternion), 9x9 (or 15x15 with bias) covariance,
     first-order bias-correction Jacobians, reset().
   - Gravity handling and frame conventions must match what tc_fgo uses (document them
     in the module docstring).
2. **Unit tests** `tests/test_imu_preintegration.py`:
   - synthetic trajectories (constant accel, pure rotation, circle) with analytic truth;
   - cross-check against tc_fgo's internal preintegration on the same sample stream
     (agreement to tight tolerance; state the tolerance you achieve);
   - bias-correction Jacobian check via finite differences.
3. **PF integration** — new IMU mode in `pf_device_runtime.py` (or a thin adapter module):
   `imu_mode="preint"`: between GNSS epochs, preintegrate the 100 Hz IMU and feed the
   device predict a per-epoch velocity/displacement guide with a process-noise covariance
   derived from the preintegration covariance (instead of the current heuristic guide).
   Heading/attitude: use the existing `INSEKF` or `ComplementaryHeadingFilter` outside
   the particle state (do NOT put attitude in particles).
4. **Ablation experiment** `experiments/exp_wp21_imu_rbpf.py`:
   - PPC Tokyo run2, at least a 3000-epoch window (full run if runtime permits);
   - three arms: (a) current CV predict (no IMU), (b) current velocity-guide IMU,
     (c) new preint mode;
   - score all arms with the `score_vs_inuex35.py` metrics (AllRMS, <50cm_full%, etc.)
     and report ESS / resample-rate statistics as filter-health indicators.
5. **`results/wp21/WP21_REPORT.md`** — setup, numbers table, honest conclusion
   (negative result acceptable but must be measured, not assumed). Append milestones
   to `results/wp21/PROGRESS.md` as you go (it is the live progress signal).

## Gates

- G1: preintegration matches tc_fgo numerically on real PPC IMU data (report max rel. diff).
- G2: ablation table complete for all three arms on the same window with the same scorer.
- G3: preint arm (c) >= velocity-guide arm (b) on <50cm_full% or AllRMS; if it is not,
  the report must contain a diagnosis (e.g. covariance too tight/loose, heading error
  dominates) and a concrete recommendation for Phase B.

## Git

- Branch: `agent/wp21-imu-preint` off `main`.
- Commit as **rsasaki0109** (never jim/jim-auto). Small, logical commits.
- Run the new tests plus the existing PF test suite subset you may affect
  (`tests/test_*pf*`, `tests/test_*rbpf*`) before the final commit.
- Do NOT push or open a PR — leave the branch local for review.
