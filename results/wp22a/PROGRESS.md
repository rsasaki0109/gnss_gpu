# WP22a Progress Log

Spec: `internal_docs/task_wp22a_dd_imu_ablation.md`. Branch: `agent/wp22a-dd-imu`
(off `agent/wp21-imu-preint`).

## Milestone 1 — baseline located + G1 reproduced (no re-run needed)

- Located the exact table (`3.0/1.2/3.2 <50cm_full%` run1/2/3) at
  `experiments/results/inuex35_shootout_baseline.md` (same numbers copied into
  `internal_docs/inuex35_tc_fgo_benchmark.md:~300-312`). Runner:
  `experiments/exp_ppc_ctrbpf_fgo.py`, method label
  `RBPF-velKF+DD+gate+hybrid` (`--methods rbpf+dd+gate+hybrid`).
- The `.pos` artifacts that produced these numbers already exist in-repo at
  `experiments/results/libgnss_ctrbpf_pos/tokyo_run{1,2,3}_RBPF-velKF+DD+gate+hybrid.pos`
  (mtime 2026-07-04). Re-scored `tokyo_run2` directly:
  `python experiments/score_vs_inuex35.py --traj "experiments/results/libgnss_ctrbpf_pos/tokyo_run2_RBPF-velKF+DD+gate+hybrid.pos" --city tokyo --run run2 --format pos --fix-statuses 1`
  -> `AllRMS=16.988 <50cm_full%=1.2` — **exact match** to the documented run2
  number (1.2). G1 satisfied by direct reproduction of the existing artifact;
  no re-run of the (expensive) generating command was needed for G1 itself.
- Reconstructed the exact generating command for later use (n_particles
  default 50000, sigma_pr default 8.0, systems default G,R,E,C,J,
  `--hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 --hybrid-sigma-m
  1.0`, `--max-epochs 1200` matching the 1200-row artifact) — will confirm by
  re-running once IMU integration is wired, since both arms (imu=off / imu=preint)
  need a fresh run anyway.

## Milestone 2 — IMU-preint wired into exp_ppc_ctrbpf_fgo.py predict step

Edited `experiments/exp_ppc_ctrbpf_fgo.py` (no CUDA changes):
- `CTRBPFConfig`: new `enable_imu_preint` + `imu_preint_*` tuning fields.
- New top-level CLI switch `--imu {off,preint}` (applies uniformly to every
  `--methods` variant via the `base` dict), plus `--imu-preint-*` tuning args
  (WP21b defaults).
- `_run_ctrbpf_on_segment`: when `enable_imu_preint`, builds a
  `gnss_gpu.imu.ComplementaryHeadingFilter` + `pf_imu_preint_adapter.ImuPreintPfGuide`
  (heading-uncertainty sigma_pos, `use_heading_uncertainty=True`) once per
  segment; each epoch (i>0) accumulates the 100 Hz PPC IMU between
  `times[i-1]` and `times[i]`, closes the segment using this pipeline's own
  causal `wls_positions` (finite-difference velocity/heading, in place of a
  separately-run robust_spp) as the heading/velocity reference, feeds
  `pf.set_velocity_covariance(...)` + `pf.predict(velocity=..., sigma_pos=...,
  rbpf_velocity_kf=True, velocity_guide_alpha=1.0)`, replacing the baseline's
  guide-less `pf.predict(dt=dt, rbpf_velocity_kf=True, ...)` call for that
  epoch. Falls back to the baseline predict on any epoch with an
  empty/degenerate IMU segment. `imu_run` (PPC `imu.csv`) is now loaded
  whenever any selected variant has `enable_zupt/imu_tc/ins_tc/imu_preint`.
- Added zero-behavior-change filter-health instrumentation: wraps
  `pf.resample_if_needed` (called internally by `pf.update(resample=True)`)
  to record pre-resample ESS/N and whether a resample fired, per epoch;
  summarized into `mean_ess_ratio` / `resample_rate` in `_PRObsStats` and
  written to the per-run results CSV, for both imu=off and imu=preint arms.
- `ast.parse` syntax check: OK.

## Milestone 3 — smoke tests passed

- `--max-epochs 60` smoke on tokyo/run2, both `--imu off` and `--imu
  preint`: both ran cleanly (Windows console needed `PYTHONUTF8=1
  PYTHONIOENCODING=utf-8` for the em-dash in a print statement — pre-existing,
  unrelated to WP22a). `--imu preint` correctly closes 59/59 segments (only
  epoch 0 has no predecessor), `mean_ess_ratio` differs measurably from
  `--imu off` (6.6e-05 -> 1.6e-03), confirming the wiring is live.
- At 60 epochs both arms scored byte-identical (`AllRMS=0.105`,
  `<50cm_full%=0.7`) -- expected, since this window's epochs are all
  hybrid-PU-anchored; moved to the full 1200-epoch window (matching the
  original baseline) to get a real ablation.

## Milestone 4 — G1 baseline delta discovered and diagnosed

Full 1200-epoch `--imu off` run on tokyo/run2 (current `HEAD`) did **not**
match the archived `16.988 AllRMS / 1.2 <50cm_full%` -- got `13.241 / 10.6`
instead (better, not worse). Root-caused via `git log` on
`exp_ppc_ctrbpf_fgo.py`: exactly one commit landed between the archived
artifact's mtime (2026-07-04) and now, `81cd0a6 "Add audited GNSS
structural methods (#127)"` (2026-07-14), which changes the Doppler-KF
update's input (`dop_model_full`, wavelength + inter-constellation
clock-drift normalized) for every `enable_rbpf_velocity_kf=True` variant,
including this one. Documented in the report as G1's "delta" per the
spec's explicit allowance. Direct .pos-artifact rescoring (no re-run)
still matches the archived number exactly (1.2 = 1.2), so the runner/config
identification itself (G1's core requirement) is solid; the delta is a
dated, named, explained upstream change, not a location error.

## Milestone 5 — full 3-run x 2-arm ablation complete (G2)

Ran `--imu {off,preint}` x `{run1,run2,run3}` (6 runs total, 1200 epochs,
50000 particles each, 15-35s wall-clock each -- cheap, so did all three
runs rather than stopping at run2). Results copied to
`results/wp22a/csv/` and `results/wp22a/pos/`. Key finding:
`<50cm_full%`/`ppc_official%` byte-identical between off/preint on all 3
runs (hybrid PU dominates emission on 82-98% of epochs); AllRMS mixed
(-10.8% run2, -4.9% run3, +20.5% run1). Filter health: DD path also
resamples every epoch at ESS/N~6-8e-5 (off) / ~3.5-4.4e-4 (preint), same
order of magnitude as WP21's raw-SPP PF.

## Milestone 6 — report written (G3), done

`results/wp22a/WP22A_REPORT.md` written with full G1/G2/G3 verdicts,
ablation table, degeneracy diagnosis (ties into WP22b likelihood work),
and a concrete recommendation (retest on a non-`+hybrid` DD-RBPF variant;
prioritize WP22b's GMM/NLOS likelihood work over further predict-step
tuning). Existing tests re-verified passing (21/21 across the 3 test files
that exercise this runner). Ready to commit.
