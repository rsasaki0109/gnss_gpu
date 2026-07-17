# WP22b Progress Log

Spec: `internal_docs/task_wp22b_likelihood.md`. Branch: `agent/wp22b-likelihood`
(off `agent/wp22a-dd-imu`).

## Milestone 1 — read spec + prior art, planned implementation

- Read `internal_docs/task_wp22b_likelihood.md`, `results/wp22a/WP22A_REPORT.md`,
  `experiments/exp_ppc_ctrbpf_fgo.py` (11.7k lines), `include/gnss_gpu/pf_device.h`,
  `python/gnss_gpu/pf_device_runtime.py`,
  `python/gnss_gpu/validation/cn0_validation.py` (the C/N0 validation commit
  `ccaf92c`).
- Key findings that shaped the plan:
  1. `pf_device_get_log_weights`/`set_log_weights` already exist in the CUDA
     API and are wrapped in Python (`get_log_weights`/`set_log_weights`) --
     tempering needs **no CUDA edits**, exactly as the spec anticipated.
  2. The runner already has a generic bisection-based log-weight tempering
     primitive, `_apply_pr_ess_guard` (used today only to guard the
     undifferenced-PR sub-update via `--pr-ess-guard-min-ratio`, default
     off). It is not PR-specific in its logic (just its call site), so item
     2 reuses it verbatim at a new, epoch-level call site instead of writing
     a new bisection routine.
  3. `pf_device_weight_gmm` (item 3) only accepts one scalar `w_los` per
     kernel call. Per-satellite w_los therefore needs multiple kernel calls
     grouped by bucketed w_los (spec's preferred first approach) -- this is
     *exact* under the PF's independent-observation model (product of
     per-satellite mixture likelihoods), not an approximation, so no CUDA
     parameter extension is used or needed.
  4. `ParticleFilterDeviceRuntime` already has full `per_particle_nlos_gate`
     + threshold + Huber plumbing (item 4) wired to the native kernels
     (`pfd_weight_kernel`/`pfd_weight_dd_carrier_afv_kernel` in
     `pf_device.cu`, confirmed by reading the kernel: each particle computes
     its own residual from its own hypothesized state, so a shared scalar
     threshold already yields a per-particle-varying rejection set). It was
     simply never passed through this runner's `_build_pf` -- zero CUDA
     work, pure Python wiring.
  5. `rbpf+dd+gate` (no `+hybrid`) is the existing non-hybrid DD-RBPF method
     label item 1 asks for; WP22a's own report explicitly recommends it as
     the next ablation target.
  6. The runner's raw `weights[i]` array is documented
     ("`_pr_likelihood_weights`: Convert PPC C/N0-like values into PF
     likelihood multipliers") as the same measured-C/N0-like quantity the
     validated `cn0_validation.py` module operates on -- used directly for
     item 3, no new RINEX/SNR parsing needed.

## Milestone 2 — implemented items 2-4 in `exp_ppc_ctrbpf_fgo.py`

All changes confined to `experiments/exp_ppc_ctrbpf_fgo.py`. No CUDA edits,
no changes to `pf_device.cu`/`pf_device.h`, no changes to the PPC production
selector.

- **Item 2 (tempering)**: new `CTRBPFConfig.enable_epoch_tempering` +
  `epoch_tempering_target_ess_ratio` (default 0.10) +
  `epoch_tempering_max_iters`. `_resample_deferred` now also defers when
  tempering is enabled (single well-defined per-epoch log-weight delta to
  temper). Pre-epoch log-weights are snapshotted at the top of the epoch
  loop; `_apply_pr_ess_guard` is called again just before `pf.estimate()`
  is computed for emission (so tempering actually affects *this* epoch's
  output, not just the next epoch's resample) -- new `--enable-epoch-
  tempering` / `--epoch-tempering-target-ess-ratio` / `--epoch-tempering-
  max-iters` CLI flags, applied uniformly to every selected `--methods`
  variant (same pattern as `--imu {off,preint}`).
- **Item 3 (C/N0+elevation GMM)**: new `_cn0_elevation_w_los` (calibrated
  logistic on C/N0 deficit below an elevation-dependent clear-sky baseline;
  see its docstring for the exact formula and the validated-data
  calibration) and `_pf_update_gmm_cn0` (buckets satellites by computed
  w_los into `cn0_gmm_n_buckets` bins, one `pf.update_gmm` kernel call per
  non-empty bucket). Raw per-satellite C/N0 (`weights[i]`, pre-transform) is
  threaded through the same epoch filters as `sat_i`/`pr_i`/`w_i`
  (system/finite/elevation/prefit-gate masks) as a new `cn0_i` array so it
  stays aligned. New `--enable-cn0-gmm` (implies `enable_pr_gmm=True`) +
  `--cn0-gmm-*` tuning flags, applied uniformly.
- **Item 4 (particle-wise NLOS)**: `_build_pf` now passes
  `per_particle_nlos_gate`/threshold/Huber config through to
  `ParticleFilterDevice(...)`. New `CTRBPFConfig.enable_particle_nlos` +
  threshold/Huber fields, `--enable-particle-nlos` + tuning flags, applied
  uniformly.
- New `_PRObsStats` fields (`epoch_tempering_epochs/mean_alpha/
  mean_post_ratio`, `cn0_gmm_mean_w_los`) and result-CSV columns for all
  three items, plus per-epoch diagnostic columns
  (`epoch_tempering_alpha/pre_ratio/post_ratio`, `cn0_gmm_buckets_used`,
  `cn0_gmm_w_los_mean`) when `collect_internal_diagnostics` is on.
- `ast.parse` syntax check: OK.

## Milestone 3 — smoke tests + regression suite

- `ast.parse` on `exp_ppc_ctrbpf_fgo.py`: OK.
- `tests/test_ppc_particle_mode_emission.py` (9), `tests/test_ppc_pf_nlos_mask_args.py`,
  `tests/test_wp4_run_local_fgo_full.py` (12) — 21/21 passing (same suite WP22a used).
- 40-epoch smoke runs on tokyo/run2, each new flag independently: baseline,
  `--enable-epoch-tempering` (mean_ess_ratio 7.43e-05 -> 0.1000, target hit
  exactly), `--enable-cn0-gmm` (cn0_gmm_mean_w_los=0.948, pr_gmm_epochs=40/40),
  `--enable-particle-nlos` (mean_ess_ratio 7.43e-05 -> 1.17e-03), and the
  all-on combination. All ran cleanly, all new CSV columns populated as
  expected.

## Milestone 4 — item 1 (G1): non-hybrid baseline, full grid

Ran `--methods rbpf+dd+gate` (no `+hybrid`), `--imu {off,preint}`, all three
1200-epoch runs, `n_particles=50000` (WP22a's exact windows/seed). AllRMS
28-54m, `<50cm_full%`=0.0% on all 6 cells (expected: no RTK anchor, a much
weaker floor than WP22a's `+hybrid` table — do not compare directly).
ESS/N 7.4e-05 to 4.9e-04, resample-every-epoch on all 6 cells, reproducing
WP22a's degeneracy diagnosis on the non-hybrid variant. **G1: PASS.**

## Milestone 5 — items 2-4 + full ablation grid (G2, G3)

Ran all 5 configs (baseline/temper/gmm/nlos/allon) x 2 IMU arms x 3 runs =
30 cells, ~15-70s each. Scored every `.pos` output with
`experiments/score_vs_inuex35.py --fix-statuses 1` via a one-off aggregation
script (`results/wp22b/score_grid.py`), merged with health-stat CSVs into
`results/wp22b/csv/wp22b_grid_scored.csv`.

- **Tempering (item 2)**: run2 ESS/N 1510x (`off`) / 374x (`preint`), AllRMS
  improved -4.6% / -0.1%. **G2: PASS** (>=10x with no AllRMS degradation).
- **C/N0+elevation GMM (item 3)**: AllRMS +1% to +25% regression on every
  cell (mean +12.9% `off` / +8.3% `preint`), modest ESS/N gain (3-12x).
  Genuine negative result, diagnosed as a likelihood-shape (not just
  sharpness) change acting on a filter whose real problem is cloud
  diffuseness.
- **Particle-wise NLOS (item 4)**: AllRMS +95% to +870% regression — severe.
  Root-caused via two extra diagnostic re-runs
  (`wp22b_diag_nlos_{pronly,ddonly}`) to the undifferenced-PR per-particle
  gate's 30m default threshold specifically; the DD-carrier-AFV gate alone
  showed zero measurable effect.
- **All-on**: tempering partially mitigates (not fixes) the particle-NLOS
  harm (42.0m/40.2m mean AllRMS vs 233.7m/114.3m for particle-NLOS alone),
  but is still worse than plain baseline (+10.4%/+6.7%).
- Wrote `results/wp22b/WP22B_REPORT.md` with the full grid, G1/G2/G3
  verdicts, mechanism-level diagnoses for both negative results, and
  concrete recommendations for WP22c (BVH ray-traced LOS/NLOS priors should
  replace the C/N0 heuristic and the raw-residual NLOS threshold) and WP23
  (re-verify the DD-carrier-AFV gate's zero-effect finding on a converged
  filter). **G3: PASS.**

Raw per-config CSVs archived at `results/wp22b/csv/wp22b_{config}_{arm}_runs.csv`
(10 files) plus the 2 diagnostic CSVs and the merged `wp22b_grid_scored.csv`.
`.pos` trajectories (30 files, gitignored like WP22a's) at
`results/wp22b/pos/<config>_<arm>/tokyo_run{1,2,3}_RBPF-velKF+DD+gate.pos`.
