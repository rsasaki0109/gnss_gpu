# WP22a Task Spec — IMU Preint on the DD-Domain RBPF (target-metric ablation)

Follow-up to WP21/WP21b (branch `agent/wp21-imu-preint`, report `results/wp21/WP21_REPORT.md`).
Part of the PF-only roadmap (`internal_docs/pf_only_imu_roadmap_2026_07_17.md`).

## Why this task

WP21b proved the preint plumbing works (heading-variance-derived sigma_pos +
per-particle Sigma_v feeding beats CV with no hand tuning), but only in a raw-SPP
harness whose AllRMS is ~75 m and whose filter is weight-degenerate
(ESS/N ~1e-5, resampling every epoch). The roadmap's target metric is
**`<50cm_full%` (3D, `experiments/score_vs_inuex35.py`)** on PPC Tokyo, where the
strongest standalone RBPF path (RBPF-velKF + DD + gate + hybrid) currently scores
**3.0 / 1.2 / 3.2** on run1/2/3 (see `internal_docs/inuex35_tc_fgo_benchmark.md`
around lines 300-312 for that configuration). This task measures whether IMU
preintegration moves the metric that matters.

## Work items (in order)

1. **Locate and reproduce the baseline.** Find the exact runner/config that produced
   the 3.0/1.2/3.2 standalone RBPF numbers (start from
   `internal_docs/inuex35_tc_fgo_benchmark.md:300-312`; the runner is likely under
   `experiments/`). Reproduce the **run2** number first and document the exact
   command + score. If exact reproduction is impossible (missing config), get as
   close as you can and document the delta before proceeding.
2. **Integrate the WP21b preint path into that pipeline's predict step**:
   `ImuPreintPfGuide` with `heading_variance_rad2` enabled (sigma_pos from modeled
   uncertainty) + `set_velocity_covariance()` Sigma_v feeding, using the PPC 100 Hz
   IMU. Keep it switchable (`--imu preint` vs `--imu off`).
3. **Ablation on run2** (same coverage/window as the reproduced baseline):
   IMU-off vs IMU-preint. Report `<50cm_full%`, AllRMS, coverage, and filter-health
   stats (ESS/N, resample rate) for both arms. If wall-clock permits, add run1 and
   run3.
4. **Degeneracy diagnosis**: if the DD path also resamples every epoch at ESS/N ~1e-5,
   include a short analysis of likelihood sharpness vs particle count (e.g. effective
   log-likelihood spread across particles) — this feeds WP22b's likelihood work
   (particle-wise NLOS + C/N0-driven GMM per the roadmap).

## Gates

- G1: baseline reproduced (or near-reproduced with documented delta) on run2.
- G2: complete IMU-off vs IMU-preint ablation table on `<50cm_full%` + AllRMS + health stats.
- G3: honest report at `results/wp22a/WP22A_REPORT.md` with a concrete, evidence-backed
  recommendation for WP22b. A measured negative passes; an unmeasured claim does not.

## Constraints

- No FGO at runtime. No CUDA kernel edits. Don't touch the PPC production selector.
- Branch: create `agent/wp22a-dd-imu` off `agent/wp21-imu-preint`. Commit as
  rsasaki0109. No push, no PR.
- Append milestones to `results/wp22a/PROGRESS.md` as you go.
- Long runs: launch via background execution and collect results in the same session;
  if a run exceeds your budget, reduce to a documented window rather than leaving
  the table incomplete.
