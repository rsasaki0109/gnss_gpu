# WP23a Progress Log

Spec: `internal_docs/task_wp23a_carrier_afv.md`. Branch: `agent/wp23a-carrier`
(off `agent/wp22b-likelihood`).

## Milestone 1 — read spec + prior art, key findings that shaped the plan

- Read `internal_docs/task_wp23a_carrier_afv.md`, `results/wp22b/WP22B_REPORT.md`,
  `experiments/exp_ppc_ctrbpf_fgo.py` (12k lines), `python/gnss_gpu/pf_device_runtime.py`,
  `include/gnss_gpu/pf_device.h`.
- **Critical finding #1 (contradicts the task's own framing):** the WP22b
  winner `rbpf+dd+gate` (`enable_dd_carrier_afv=True`) already calls
  `pf.update_dd_carrier_afv(...)` every epoch -- DD-carrier AFV (fractional
  cycle carrier phase) was *already* the sole "DD" signal in WP22b's
  baseline, not DD-pseudorange as the WP22b report's "DD-PR RBPF regime"
  language implies. `DDResult` (from `DDCarrierComputer`) has no
  pseudorange field at all -- there is a *separate* `DDPseudorangeComputer`
  /`DDResult`-alike class, but it was only ever wired to an anchor
  (`enable_dd_pr_ls_anchor` -> `pf.position_update`) or FGO post-process
  cache, **never** to a genuine per-epoch `pf.update_dd_pseudorange(...)`
  weight update in the non-hybrid path. So "wire DD carrier AFV into the
  non-hybrid path" (item 1) is reinterpreted as: (a) add the *missing*
  DD-pseudorange weight-update stage, and (b) restructure the *existing*
  single-shot DD-carrier-AFV call into a genuine Suzuki two-family
  multi-stage schedule (item 2) -- not "wire carrier AFV in" from zero.
  This is flagged as a deviation-worth-reporting, not silently corrected.
- **Finding #2 (prior art discovered, reused per "wire, don't reinvent"):**
  a separate, much more elaborate MUPF (Multiple Update Particle Filter)
  implementation already exists for a *different* (GSDC2023/Trimble, not
  PPC) PF-smoother track:
  `gnss_gpu.dd_carrier_epoch_update.apply_carrier_epoch_update` +
  `gnss_gpu.pf_smoother_config.MupfConfig`, with a **validated
  coarse-to-fine sigma sequence default `(2.0, 0.5, 0.05)` cycles**
  (resample-if-needed before each of 3 progressively sharper
  `update_carrier_afv` calls). This is exactly Suzuki's core idea. WP23a's
  new DD-CP stage reuses this validated sequence default rather than
  inventing a new one, but adds explicit ESS-target bisection tempering
  per step (the GSDC track does not), since the spec's item 2 explicitly
  requires "temper to ESS target" per stage.
- `update_dd_carrier_afv` / `update_dd_joint` / `update_dd_pseudorange`
  Python wrappers already exist in `pf_device_runtime.py` (confirmed no
  CUDA edits needed) -- wired directly, nothing new added to `pf_device.cu`
  / `pf_device.h`.
- `_apply_pr_ess_guard` (WP22b's bisection ESS-target tempering primitive)
  is reused verbatim at new per-stage call sites via a new thin wrapper
  `_mupf_stage_update` (update -> temper -> `resample_if_needed()`).
- No cycle-slip detector exists anywhere in the DD machinery for the PPC
  path (grepped `dd_carrier.py` and `exp_ppc_ctrbpf_fgo.py` for "slip":
  none). Implemented a minimal epoch-to-epoch DD-carrier-phase continuity
  proxy gate (`_cp_slip_gate`), per the spec's documented fallback.

## Milestone 2 — implementation

All changes confined to `experiments/exp_ppc_ctrbpf_fgo.py` + one new test
file. No CUDA edits, no PPC production-selector changes, no FGO wired into
the runtime loop.

- New `CTRBPFConfig.enable_cp_mupf` + `cp_mupf_dd_pr_sigma_m` (5.0, inherited
  from `fgo_dd_pr_sigma_m`'s tc_fgo/rbpf_fgo default) +
  `cp_mupf_pr_stage_target_ess_ratio` (0.10) +
  `cp_mupf_dd_cp_sigma_sequence_cycles` ((2.0, 0.5, 0.05), reused from the
  GSDC MUPF track) + `cp_mupf_cp_stage_target_ess_ratio` (0.10) +
  `cp_mupf_stage_max_iters` + `cp_mupf_min_pairs` + `cp_mupf_cp_n_groups`
  (satellite-group-split fallback, off by default) +
  `cp_mupf_slip_gate_enabled`/`cp_mupf_slip_max_delta_cycles`/
  `cp_mupf_slip_max_dt_s` (cycle-slip proxy).
- New helpers: `_mupf_stage_update` (one Suzuki stage: weighted update with
  `resample=False` -> `_apply_pr_ess_guard` -> `resample_if_needed()`),
  `_dd_result_slice` (index a DD result via `dataclasses.replace`),
  `_cp_slip_gate` (epoch-to-epoch continuity proxy),
  `_dd_result_groups` (round-robin satellite-group split).
- New method variant `rbpf+dd+cp+gate`: same base + `aaa_gate` as WP22b's
  `rbpf+dd+gate`, but `enable_dd_carrier_afv=False` +
  `enable_cp_mupf=True` -- the MUPF schedule *replaces* (mutually
  exclusive `elif`) the old single-shot DD-carrier-AFV call at the same
  epoch call site, rather than double-applying it.
- Schedule per epoch: (i) DD-pseudorange update (new -- previously never
  wired as a weight update in this path) -> temper -> resample-if-needed;
  cloud-spread-vs-lambda/2 diagnostic measured here; (ii) DD-carrier AFV
  applied as the 3-step coarse-to-fine sigma sequence, each step
  independently tempered + resample-if-needed, after an epoch-to-epoch
  cycle-slip continuity gate.
- Mid-epoch resamples inside the MUPF block invalidate WP22b's blanket
  `enable_epoch_tempering` pre/post log-weight correspondence (a resample
  permutes particles); the epoch-start snapshot is re-anchored
  immediately after the MUPF block when this happens, so a same-epoch
  blanket temper afterward (if any updates still follow) stays valid.
- **Bug found + fixed during smoke-testing:** the per-variant DD-computer
  passthrough gate (`need_dd_for_variant` / `dd_pr_for_variant`, used when
  building each variant's `run_pf` call) checked
  `enable_dd_carrier_afv`/`enable_dd_pr_ls_anchor`/`enable_fgo_lambda` but
  not the new `enable_cp_mupf` -- so `rbpf+dd+cp+gate` silently got
  `dd_pr_computer=None` even though the module-level per-run DD-PR
  computer was constructed, and MUPF stage (i) never fired (`mupf_pr=0/N`
  in a first smoke run). Fixed by adding `enable_cp_mupf` to both
  conditions; confirmed fixed (`mupf_pr=8/40` on the next smoke run,
  matching `mupf_cp`'s count).
- Extended `_DDStats` with MUPF diagnostics (epochs/pairs/resample-rate/
  mean-alpha/mean-post-ESS-ratio per stage, slip counts, cloud-spread-vs-
  half-lambda mean) -- reused the existing return-tuple slot rather than
  widening the 14-element `run_pf` return signature.
- New CSV columns (`mupf_pr_*`, `mupf_cp_*`, `mupf_cloud_spread_*`,
  `cp_mupf*` config echo) added to the per-run result dict.
- New CLI flags: `--cp-mupf-dd-pr-sigma-m`, `--cp-mupf-pr-stage-target-ess-ratio`,
  `--cp-mupf-dd-cp-sigma-sequence-cycles`, `--cp-mupf-cp-stage-target-ess-ratio`,
  `--cp-mupf-stage-max-iters`, `--cp-mupf-min-pairs`, `--cp-mupf-cp-n-groups`,
  `--disable-cp-mupf-slip-gate`, `--cp-mupf-slip-max-delta-cycles`,
  `--cp-mupf-slip-max-dt-s`.
- `ast.parse` syntax check: OK.

## Milestone 3 — G1 sanity test + regression suite

- New `tests/test_wp23a_dd_carrier_afv_sanity.py`: synthetic, noiseless
  DD-CP epoch (random Earth-scale satellite geometry, GPS-L1 wavelength),
  7 explicit particle hypotheses swept +/-0.02 m around the true rover
  position along the reference-satellite line of sight (tight enough that
  every DD pair's own fractional-cycle wrap boundary is respected -- see
  the test's docstring for why the first attempt at +/-0.09 m offsets hit
  a real (not spurious) local-wiggle multimodality artifact from a
  near-antipodal random satellite pair, which is itself a nice concrete
  illustration of the spec's own "lambda-spaced multimodality" hazard).
  Asserts the AFV log-likelihood peaks exactly at the true-position
  particle and decays monotonically on both sides. **PASSED.**
- Full regression suite (WP22a/WP22b's suite +
  `test_pf_device_wrapper.py` + `test_dd_carrier_epoch_update.py` + the
  new WP23a test): 41/41 passing.
- Smoke-tested `rbpf+dd+cp+gate` on tokyo/run2 (10-40 epochs, `--imu off`,
  `--enable-epoch-tempering`): ran cleanly after the per-variant gating
  fix above; `mupf_pr` and `mupf_cp` epoch counts now match; satellite
  group-split fallback path (`--cp-mupf-cp-n-groups 2`) also smoke-tested,
  ran cleanly.
- **Early cloud-spread diagnostic signal (40-epoch smoke, tokyo/run2,
  `--imu off`):** `spread/halfLambda` ratio measured at 204.6 (i.e. the
  post-DD-PR-stage cloud is ~205x wider than half a carrier wavelength) --
  this is the measured mechanism the spec's item 3/5 asks to check for,
  and strongly suggests the same "diffuse-cloud-collides-with-sharp-
  likelihood" failure mode WP22a/WP22b diagnosed for pseudorange will
  reproduce for carrier AFV too. Confirmed/quantified on the full grid
  below.

## Milestone 4 — full ablation grid (G2)

Ran the 4-cell x 3-run grid (`rbpf+dd+gate` vs `rbpf+dd+cp+gate`, both with
`--enable-epoch-tempering`, x `--imu {off,preint}`, tokyo/run{1,2,3},
1200 epochs, 50k particles) in the foreground -- ~15-40s per invocation.
Scored all 12 `.pos` outputs with `experiments/score_vs_inuex35.py` via
`results/wp23a/score_grid.py`, merged into
`results/wp23a/csv/wp23a_grid_scored.csv`.

- `<50cm_full%` = 0.0% on all 12 cells.
- AllRMS statistically unchanged between baseline and +CP-MUPF on every
  cell (<0.5% difference) -- per-arm means: baseline off 37.64m / preint
  37.42m; +CP-MUPF off 37.64m / preint 37.60m.
- Cloud-spread-vs-half-lambda diagnostic: 58x-126x too wide on every cell
  -- the mechanism the spec's item 3 anticipated, now measured.
- **New finding while inspecting the grid**: `mupf_pr_mean_alpha = 0.0` on
  all 12/12 cells -- the DD-PR stage is being unconditionally reverted by
  `_apply_pr_ess_guard`'s "pre_ratio already below target -> revert"
  branch, because the untempered PR+Doppler-KF prefix already pushes
  ESS/N to ~1e-4 before the DD-PR stage's own guard call ever sees it.
  **G2: PASS** (complete table + diagnostics; see WP23A_REPORT.md §3).

## Milestone 5 — targeted diagnostic + root-cause + report (G3)

- Added a diagnostic-only `--cp-mupf-resample-before-stage` flag
  (resample-if-needed *before*, not only after, each MUPF stage --
  mirrors the validated GSDC MUPF track's own ordering) to confirm and
  quantify the Milestone 4 finding without touching the main grid's
  spec-literal methodology.
- One extra targeted run (tokyo/run2, `--imu off`, beyond the main grid,
  mirroring WP22b's own practice of extra diagnostic runs to root-cause a
  negative result): AllRMS 29.262m -> 20.677m (-29%), `mupf_pr_mean_alpha`
  0.000 -> 0.008 (nonzero, confirming the mechanism), cloud-spread ratio
  124.9 -> 84.4 (improved but still ~840x too large). `<50cm_full%` stays
  0.0%.
- Regression suite re-run after the diagnostic-flag addition: 41/41
  passing.
- Wrote `results/wp23a/WP23A_REPORT.md`: framing correction (§1), what was
  built (§2), G1 sanity test (§2), full ablation table (§3, G2), two
  root-caused failure mechanisms + targeted diagnostic + concrete WP23b
  requirement list (§4, G3), deviations (§5).
- **G3: PASS.**
