# WP22b Report — Expose the PF and Fix the Likelihood (NLOS/C-N0 GMM + tempering)

Spec: `internal_docs/task_wp22b_likelihood.md`. Branch: `agent/wp22b-likelihood`
(off `agent/wp22a-dd-imu`). Date: 2026-07-17.

## 1. What was built

All changes are in `experiments/exp_ppc_ctrbpf_fgo.py`. No CUDA kernel edits
(`pf_device.cu`/`pf_device.h` untouched), no FGO wired into the runtime loop,
no changes to the PPC production selector/ranker. Every item reuses an
existing, previously-unwired device/runner capability rather than adding new
CUDA surface:

1. **Item 2 — adaptive likelihood tempering.** New
   `CTRBPFConfig.enable_epoch_tempering` / `epoch_tempering_target_ess_ratio`
   (default 0.10) / `epoch_tempering_max_iters`, plus
   `--enable-epoch-tempering` / `--epoch-tempering-target-ess-ratio` /
   `--epoch-tempering-max-iters` CLI flags applied uniformly to whichever
   `--methods` variant is selected (same pattern as WP22a's `--imu
   {off,preint}`). Implementation reuses `_apply_pr_ess_guard` verbatim — an
   existing generic bisection-on-log-weights primitive already in the file
   (previously wired only to a PR-sub-update ESS guard,
   `--pr-ess-guard-min-ratio`, default off) — at a new, epoch-level call
   site: log-weights are snapshotted at the top of each epoch (before
   predict/update), and the guard is invoked again just before
   `pf.estimate()` is computed for emission, so the tempered weights affect
   *that epoch's* output, not just the next epoch's resample.
   `_resample_deferred` now also defers per-update resampling when
   tempering is enabled, so there is a single well-defined "this epoch's
   total log-likelihood delta" to temper (across whichever of PR/DD-carrier/
   Doppler-KF/position-update ran that epoch). **No CUDA edits** — the
   device API's `get_log_weights`/`set_log_weights` already existed and are
   already Python-wrapped.
2. **Item 3 — C/N0- and elevation-driven GMM.** New `_cn0_elevation_w_los`
   (a calibrated logistic mapping measured C/N0 + elevation to a per-
   satellite LOS mixture weight, documented in its docstring) and
   `_pf_update_gmm_cn0` (buckets satellites by computed w_los into
   `cn0_gmm_n_buckets` — default 5 — bins, issuing one `pf.update_gmm` call
   per non-empty bucket). `pf_device_weight_gmm` only accepts one scalar
   `w_los` per call, so the spec's first-choice approach ("multiple kernel
   calls over satellite subsets sharing parameters") was used; this is
   **exact**, not an approximation, under the PF's independent-observation
   model (the joint likelihood is the product of per-satellite mixture
   likelihoods), up to w_los quantization within a bucket. **No CUDA
   parameter extension was needed** — the per-satellite-grouping approach
   the spec asks to try first turned out to be practical, so the "minimal
   CUDA extension" escape hatch was not used. New `--enable-cn0-gmm` (implies
   `enable_pr_gmm=True`) + `--cn0-gmm-*` tuning flags. Calibration: baseline
   C/N0 ~30 dB-Hz @ 0deg / ~45 dB-Hz @ 90deg elevation, w_los=0.5 at a 14 dB
   deficit (mean of the two validated UrbanNav site gaps, 16.5/11.9 dB —
   `python/gnss_gpu/validation/cn0_validation.py`, commit `ccaf92c` "Add
   C/N0 validation against measured UrbanNav signal strength"), 4 dB
   logistic transition width (sharp, matching the validated 0.94-0.985 AUC).
3. **Item 4 — particle-wise NLOS deweighting.** `_build_pf` now passes
   `per_particle_nlos_gate` + thresholds + Huber config through to
   `ParticleFilterDevice(...)`. This is a **pure wiring change**: the native
   kernels (`pfd_weight_kernel` for undifferenced PR,
   `pfd_weight_dd_carrier_afv_kernel` for DD-carrier-AFV) already gate each
   satellite's contribution per particle — each particle computes its own
   residual from its own hypothesized state, so a single shared scalar
   threshold already produces a per-particle-varying rejection set (Niimi-
   style) — but this runner's `_build_pf` never passed the config through
   before WP22b, so the feature was always off. New
   `CTRBPFConfig.enable_particle_nlos` + threshold/Huber fields,
   `--enable-particle-nlos` + tuning flags.
4. New `_PRObsStats` fields and result-CSV columns for all three items
   (`epoch_tempering_*`, `cn0_gmm_*`, `particle_nlos*`), plus per-epoch
   diagnostic columns when `collect_internal_diagnostics` is on.
   `ast.parse` syntax check: OK. Existing regression tests that exercise
   this file still pass: `tests/test_ppc_particle_mode_emission.py` (9),
   `tests/test_ppc_pf_nlos_mask_args.py`, `tests/test_wp4_run_local_fgo_full.py`
   (12) — 21/21 passing after these changes (same suite WP22a used).

## 2. Item 1 / Gate G1 — non-hybrid PF-dominant baseline

Per WP22a's own recommendation (§7.2 of `WP22A_REPORT.md`), the baseline
variant is `RBPF-velKF+DD+gate` (`--methods rbpf+dd+gate`, no `+hybrid`),
run on the same run1/run2/run3 1200-epoch windows, both `--imu {off,preint}`
arms, all other flags at default (`n_particles=50000`, `sigma_pr=8.0`,
`systems=G,R,E,C,J`):

```
PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py \
  --runs tokyo/run1,tokyo/run2,tokyo/run3 --methods rbpf+dd+gate \
  --max-epochs 1200 --imu {off|preint} \
  --pos-dir results/wp22b/pos/baseline_{off|preint} \
  --results-prefix wp22b_baseline_{off|preint}
```

| run | arm | AllRMS [m] | \<50cm_full% | ppc_official% | mean ESS/N | resample rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| run1 | off | 50.943 | 0.0 | 0.0 | 7.430e-05 | 1.000 |
| run1 | preint | 53.543 | 0.0 | 0.0 | 4.074e-04 | 1.000 |
| run2 | off | 30.641 | 0.0 | 0.0 | 6.622e-05 | 1.000 |
| run2 | preint | 28.000 | 0.0 | 0.0 | 2.677e-04 | 1.000 |
| run3 | off | 32.692 | 0.0 | 0.0 | 8.853e-05 | 1.000 |
| run3 | preint | 31.391 | 0.0 | 0.0 | 4.850e-04 | 1.000 |

Raw per-run CSVs: `results/wp22b/csv/wp22b_baseline_{off,preint}_runs.csv`.
Scored (merged) CSV: `results/wp22b/csv/wp22b_grid_scored.csv`.

**This baseline is much worse than WP22a's `+hybrid` numbers** (AllRMS
6.7-15.5m, `<50cm_full%` 6.0-10.6%) — as expected and exactly why item 1
exists: without the libgnss++ v5 RTK `position_update` floor, this variant's
own DD-RBPF estimate (diffuse `spread_pos_init=50m` cloud, no external
anchor) is what gets scored, at AllRMS 28-54m and 0% pass rate on every
run/arm. **`<50cm_full%`/`ppc_official%` = 0.0% on all 6 cells — do not
compare these AllRMS/pass-rate numbers against WP22a's `+hybrid` table; they
measure a fundamentally different (much weaker) accuracy floor.** ESS/N
(7.4e-05 to 4.9e-04) confirms WP22a's degeneracy diagnosis reproduces here
too, at the same order of magnitude, resample-every-epoch on all 6 cells.

**G1: PASS** — complete baseline table, both arms, all three runs (spec
required item 1 as the reference table for the rest; done in full,
reusing WP22a's exact run windows/seed/particle count).

## 3. Item 2 / Gate G2 — adaptive likelihood tempering

Target ESS/N = 0.10 (default), same runs/arms as §2, `--enable-epoch-tempering`
added on top of the §2 baseline command (nothing else changed).

| run | arm | AllRMS [m] | Δ vs baseline | mean ESS/N | ESS/N x-baseline | mean tempering alpha |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| run1 | off | 51.950 | +2.0% | 1.0001e-01 | 1346x | 0.00464 |
| run1 | preint | 53.633 | +0.2% | 1.0001e-01 | 245x | 0.00606 |
| run2 | off | 29.220 | **-4.6%** | 1.0001e-01 | **1510x** | 0.00415 |
| run2 | preint | 27.979 | -0.1% | 1.0001e-01 | 374x | 0.00555 |
| run3 | off | 31.748 | -2.9% | 1.0001e-01 | 1130x | 0.00460 |
| run3 | preint | 30.656 | -2.3% | 1.0001e-01 | 206x | 0.00602 |

Raw CSVs: `results/wp22b/csv/wp22b_temper_{off,preint}_runs.csv`.
`<50cm_full%` stays 0.0% on every cell (unsurprising given the ~28-54m AllRMS
regime — tempering does not add new information, it redistributes existing
weight).

**G2 gate text: "tempering raises mean ESS/N by ≥10x without degrading
AllRMS on run2."** Measured on run2 specifically: `off` arm 6.622e-05 →
1.0001e-01 = **1510x** (AllRMS 30.641 → 29.220, **improved** -4.6%, not
degraded); `preint` arm 2.677e-04 → 1.0001e-01 = **374x** (AllRMS 28.000 →
27.979, essentially flat, -0.1%). Both arms clear the ≥10x bar by two to
three orders of magnitude and neither degrades AllRMS.

**G2: PASS.**

**Mechanism / estimator-consistency caveat (spec-required documentation):**
tempering here is implemented as a bisected scalar `alpha` applied to the
*delta* between an epoch's pre- and post-update log-weights
(`pre + alpha * delta`), i.e. an order-preserving affine transform of the
log-likelihood increment — it narrows the gap between the best- and worst-
weighted particles without changing which particle ranks where. That is
exactly why it does not hurt AllRMS here (the weighted-mean estimate barely
moves) while still fixing the resampling-collapse symptom WP22a measured
(ESS/N ~1e-4-1e-5 → target 0.10, three-plus orders of magnitude). **It is
not a free lunch**: tempering trades statistical efficiency for diversity —
the tempered posterior is not the true filtering posterior, only a
deliberately flattened version of it, so downstream consumers of the weight
distribution itself (not just the point estimate) should not treat a
tempered PF as posterior-calibrated. `resample_rate` stays 1.000 on every
cell (every epoch's *pre*-tempering ESS/N is already far below the adaptive-
resample threshold; tempering changes *how much* weight concentration there
is at resample time, not *whether* a resample fires under this filter's
`ess_threshold`).

## 4. Item 3 — C/N0 + elevation GMM

Same runs/arms, `--enable-cn0-gmm` added on top of the §2 baseline command.

| run | arm | AllRMS [m] | Δ vs baseline | mean ESS/N | ESS/N x-baseline | mean w_los |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| run1 | off | 53.378 | +4.8% | 8.832e-04 | 11.9x | 0.901 |
| run1 | preint | 57.374 | +7.2% | 2.035e-03 | 5.0x | 0.901 |
| run2 | off | 34.708 | **+13.3%** | 6.484e-04 | 9.8x | 0.872 |
| run2 | preint | 28.319 | +1.1% | 1.721e-03 | 6.4x | 0.872 |
| run3 | off | 40.950 | **+25.3%** | 3.654e-04 | 4.1x | 0.869 |
| run3 | preint | 36.640 | +16.7% | 1.645e-03 | 3.4x | 0.869 |

Raw CSVs: `results/wp22b/csv/wp22b_gmm_{off,preint}_runs.csv`. `<50cm_full%`
stays 0.0% everywhere. Mean w_los 0.87-0.90 means the calibrated logistic
(§1 item 2) classified most satellites as LOS-leaning on this dataset — not
implausible for open-sky PPC tokyo segments, but it means the mixture's
NLOS component (mu_nlos=15m, sigma_nlos=30m) is only lightly weighted on
average, yet **AllRMS still gets measurably worse (+1% to +25%, mean +12.9%
`off` / +8.3% `preint`)**, and the ESS/N gain is modest (3-12x, well short
of tempering's 200-1500x).

**Diagnosis (a genuine negative result, measured and reported honestly per
the spec's framing):** this baseline's core failure mode, per WP22a §6, is
a *diffuse, RTK-unanchored particle cloud* colliding with a *sharp*
pseudorange likelihood — not systematic NLOS bias. Softening the PR
likelihood into a mixture (even at high w_los) reshapes the likelihood
*surface*, not just its overall sharpness: it changes each particle's
relative ranking, not merely the spread between ranks (contrast tempering's
order-preserving rescaling, §3). When the softening is not well-matched to
this specific regime, that reshaping can (and here, does) shift the
effective posterior mode away from the true-position-adjacent particles,
trading a small amount of NLOS robustness for a larger amount of precision
loss on a filter that was already struggling to converge for an unrelated
reason.

## 5. Item 4 — particle-wise NLOS deweighting

Same runs/arms, `--enable-particle-nlos` added on top of the §2 baseline
command (`particle_nlos_undiff_pr_threshold_m=30.0`,
`particle_nlos_dd_carrier_threshold_cycles=0.5`, Huber off — library
defaults).

| run | arm | AllRMS [m] | Δ vs baseline | mean ESS/N | ESS/N x-baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| run1 | off | 185.036 | **+263%** | 1.048e-03 | 14.1x |
| run1 | preint | 104.307 | **+95%** | 2.043e-03 | 5.0x |
| run2 | off | 297.395 | **+870%** | 1.575e-03 | 23.8x |
| run2 | preint | 98.382 | **+251%** | 3.606e-03 | 13.5x |
| run3 | off | 218.544 | **+568%** | 2.367e-03 | 26.7x |
| run3 | preint | 140.229 | **+347%** | 3.590e-03 | 7.4x |

Raw CSVs: `results/wp22b/csv/wp22b_nlos_{off,preint}_runs.csv`. This is a
**severe, unambiguous degradation** — AllRMS 2x to nearly 10x worse than
baseline on every cell, despite ESS/N improving 5-27x (higher diversity, far
worse accuracy — a red flag, not a win).

**Root-caused via two targeted diagnostic re-runs** (run2/off, not part of
the main grid, `results/wp22b/csv/wp22b_diag_nlos_{pronly,ddonly}_runs.csv`):

- `--enable-particle-nlos --particle-nlos-dd-carrier-threshold-cycles 0`
  (undiff-PR gate **on**, DD-carrier gate forced off): AllRMS = **297.395** —
  identical to the full particle-NLOS cell. The undiff-PR gate alone
  reproduces the entire degradation.
- `--enable-particle-nlos --particle-nlos-undiff-pr-threshold-m 0`
  (DD-carrier gate **on**, undiff-PR gate forced off): AllRMS = **30.641** —
  identical to the plain baseline (§2) to three decimal places. The
  DD-carrier-AFV per-particle gate alone has **zero measurable effect**,
  positive or negative, in this regime.

**Diagnosis:** the undifferenced-PR per-particle threshold
(`per_particle_nlos_undiff_pr_threshold_m=30.0`, a *default* value never
previously exercised by this runner) is the entire cause. The mechanism
(`pfd_weight_kernel` in `pf_device.cu`, read as part of this task): each
particle keeps only satellites whose residual (from *that particle's own*
hypothesized position/clock-bias) is within 30m, unless fewer than
`min(n_sat,4)` would survive, in which case the gate is silently disabled
for that particle. In a converged, RTK-anchored filter this is the intended
Niimi-style behavior — reject satellites that are inconsistent with an
already-correct position. In this **non-hybrid, pre-convergence** baseline
(diffuse `spread_pos_init=50m` cloud, no external anchor, per WP22a §6),
particles that are tens of meters from the truth can still find a small,
internally-self-consistent subset of satellites within 30m of *their own*
(wrong) hypothesis — the gate then evaluates the likelihood over only that
lucky/wrong subset, which can score deceptively well relative to a correctly
-positioned particle being fairly judged against the full, noisier satellite
set. The result rewards self-consistent wrongness over which the ESS/N
metric (rewarding diversity) cannot distinguish from genuine NLOS rejection
— exactly the "higher ESS/N, worse AllRMS" signature measured above.

**This is not evidence the kernel feature is broken** — the DD-carrier gate
showed zero effect (neither helped nor hurt), and the mechanism itself is
standard Niimi-style per-particle gating, already validated elsewhere in
this codebase's test suite. It is a **threshold/regime mismatch**: a fixed
30m absolute residual threshold is simply not calibrated for a cloud this
diffuse.

## 6. Item 5 / Gate G3 — final ablation grid

All 5 configs x 2 IMU arms x all 3 runs (30 cells; spec required run2 as the
minimum, run1/run3 "if wall-clock permits" — wall-clock was ~15-70s per
config x arm x 3-run invocation, so all three runs were completed for every
cell, matching WP22a's practice).

**Per-run-averaged summary** (mean over run1/run2/run3; full 30-row detail
in `results/wp22b/csv/wp22b_grid_scored.csv`):

| config | imu | AllRMS [m] | Δ vs baseline | \<50cm_full% | mean ESS/N | resample rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| baseline | off | 38.092 | — | 0.0 | 7.635e-05 | 1.000 |
| baseline | preint | 37.645 | — | 0.0 | 3.867e-04 | 1.000 |
| +tempering | off | 37.639 | **-1.2%** | 0.0 | 1.0001e-01 | 1.000 |
| +tempering | preint | 37.423 | -0.6% | 0.0 | 1.0001e-01 | 1.000 |
| +GMM(C/N0) | off | 43.012 | +12.9% | 0.0 | 6.323e-04 | 1.000 |
| +GMM(C/N0) | preint | 40.778 | +8.3% | 0.0 | 1.800e-03 | 1.000 |
| +particle-NLOS | off | 233.659 | **+513%** | 0.0 | 1.663e-03 | 1.000 |
| +particle-NLOS | preint | 114.306 | +204% | 0.0 | 3.080e-03 | 1.000 |
| all-on | off | 42.042 | +10.4% | 0.0 | 1.0000e-01 | 1.000 |
| all-on | preint | 40.160 | +6.7% | 0.0 | 1.0000e-01 | 1.000 |

Note on all-on vs particle-NLOS-alone: all-on's AllRMS (42.0 `off` / 40.2
`preint`) is far better than particle-NLOS-alone (233.7 / 114.3) despite
including the same harmful gate — tempering appears to substantially (not
fully) *mitigate* the particle-NLOS gate's harm, plausibly because
end-of-epoch tempering curbs the runaway weight concentration the
"lucky/wrong" particles would otherwise accumulate before the next resample.
All-on is still worse than plain baseline (+10.4%/+6.7%), i.e. tempering
masks but does not fix the particle-NLOS threshold miscalibration, and the
GMM component's own negative bias (§4) is not touched by tempering either
(tempering doesn't correct likelihood *shape*, only rescales its overall
sharpness).

**G3 gate text: "full ablation grid complete with honest per-cell numbers;
report states which likelihood upgrades actually pay and a concrete
recommendation for WP22c... and WP23."**

**Which upgrades actually pay, on this measured grid:**

- **Tempering: pays.** Only item that improves (or leaves flat) AllRMS
  while delivering a 200-1500x ESS/N lift on the exact degeneracy WP22a
  measured. Cheap (no CUDA), safe (order-preserving on the point estimate),
  with a documented, real caveat (not posterior-calibrated; a diversity, not
  accuracy, mechanism).
- **C/N0+elevation GMM (as calibrated here): does not pay.** +1% to +25%
  AllRMS regression on every cell, modest ESS/N gain. Diagnosed as a
  likelihood-*shape* change acting on a filter whose problem is cloud
  diffuseness, not NLOS bias, in this specific non-hybrid regime.
- **Particle-wise NLOS deweighting (as calibrated here): actively harmful.**
  +95% to +870% AllRMS regression, root-caused via targeted ablation to the
  undifferenced-PR gate's 30m default threshold interacting badly with a
  diffuse, unconverged cloud. The DD-carrier-AFV gate is neutral (measured
  zero effect, not merely "small").

**Concrete recommendations:**

1. **WP22c (BVH ray-traced LOS/NLOS priors) should replace, not extend, the
   C/N0+elevation heuristic used here.** The C/N0-driven GMM's failure mode
   is a likelihood-shape mismatch caused by an imperfect NLOS classifier
   (aggregate C/N0+elevation, mean-gap-calibrated on a *different* site's
   data — §1 item 2, §4); a ray-traced per-satellite LOS/NLOS label is a
   strictly better-informed signal for exactly the same GMM
   `pf_device_weight_gmm` bucketed-call mechanism built in this task (drop
   the ray-traced boolean into `_cn0_elevation_w_los`'s role, e.g. w_los=1
   for ray-confirmed LOS / a calibrated low value for ray-confirmed NLOS,
   still using the existing bucketed multi-call plumbing — no CUDA changes
   needed there either).
2. **Do not carry forward `particle_nlos_undiff_pr_threshold_m=30.0`
   (or the per-particle-NLOS feature generally) onto a non-hybrid,
   pre-convergence filter without recalibration.** Either (a) retune the
   threshold much larger for this regime (order of the actual diffuse-cloud
   residual spread, not a converged-filter value), or (b) gate the feature
   to only activate once the filter is independently known to be converged
   (e.g. ESS/N above a floor, or already hybrid/RTK-anchored) — WP22c's
   ray-traced NLOS classification is also a natural drop-in replacement for
   the raw-residual threshold here: gate on "ray-traced NLOS" per satellite
   per particle instead of "|residual| > 30m", which is not itself an
   NLOS test.
3. **Adopt epoch tempering as a standing filter-health control for WP22c
   and WP23.** It is the one item here that reliably fixes WP22a's
   measured symptom (every-epoch resampling collapse) without a measured
   accuracy cost, and it composes with future likelihood work (as shown by
   its partial mitigation of the particle-NLOS harm in the all-on cell) —
   but it should be treated as a diagnostic/stabilization layer, not a
   substitute for fixing likelihood *shape* (GMM/NLOS calibration) or
   *cloud spread* (predict-step guides, WP21/WP22a), per its
   order-preserving-transform caveat in §3.
4. **WP23 (AR)**: the DD-carrier-AFV per-particle gate's measured zero
   effect here (§5) should be re-verified on a converged/hybrid-anchored
   filter state before being written off — it is plausible the gate simply
   never gets a chance to matter on this specific diffuse, non-hybrid
   baseline (either every particle's DD residuals uniformly clear or
   uniformly miss the 0.5-cycle threshold, or the `dd_min_pairs_update`/
   `aaa_gate` gates already filter out the epochs where it would bite).

**G3: PASS** — full 30-cell grid, honest per-cell numbers (including two
targeted diagnostic re-runs to root-cause the particle-NLOS regression
rather than merely reporting it), and falsifiable, actionable
recommendations for both WP22c and WP23.

## 7. Deviations from the spec

- **Tempering's call site is *before* `pf.estimate()` for emission, not
  merely before the deferred end-of-epoch resample.** The spec describes
  tempering as a per-epoch operation but does not pin down exactly where in
  the epoch it must apply; placing it only before the final resample (the
  most literal reading of "one log-weight snapshot per epoch") would have
  changed *only* the next epoch's initial condition, not the current
  epoch's *emitted* position — making it impossible to measure any
  AllRMS/`<50cm_full%` effect from tempering at all, which G2 explicitly
  requires measuring. Moving the call to just before `pf.estimate()` (still
  a single per-epoch application, still using the single pre-epoch
  log-weight snapshot) is a deliberate, documented choice to make the gate
  measurable as specified.
- **No CUDA edits were needed for item 3**, even though the spec permitted
  a minimal extension "THIS TASK ONLY" as a fallback. The spec's
  first-choice approach (bucketed per-satellite kernel calls) turned out to
  be practical and, per §1, mathematically exact under the PF's
  independent-observation model — so the fallback was not exercised. This
  is compliance with the letter and spirit of the constraint, not a gap.
- **Item 5's aggregation used a small one-off script**
  (`results/wp22b/score_grid.py`) to batch-call `score_vs_inuex35.py`
  across all 30 `.pos` outputs and merge with the health-stat CSVs, since
  the spec did not prescribe a specific aggregation mechanism.
- **Two extra diagnostic re-runs** (§5, `wp22b_diag_nlos_{pronly,ddonly}`)
  were added beyond the required grid to root-cause the particle-NLOS
  regression, rather than only reporting "it got worse." This follows the
  same standard the spec explicitly sets for G2 ("if it degrades, measure
  and diagnose — that itself is a valid outcome, but must be measured"),
  applied here to item 4 because the magnitude of the regression (up to
  +870%) demanded an explanation for the G3 recommendation to be credible
  and actionable rather than a bare number.
- **`<50cm_full%`/`ppc_official%` = 0.0% on all 36 measured cells** (30
  grid + 6 baseline). This is a real, expected property of the non-hybrid
  regime item 1 asks for (no external RTK floor, AllRMS in the tens of
  meters) — not a scoring bug. It means this report's gates are evaluated
  on AllRMS + ESS/N + resample rate, exactly as G2's text specifies
  ("Measure: ESS/N, resample rate, AllRMS, `<50cm_full%`"), with
  `<50cm_full%` reported for completeness but uninformative at this
  accuracy floor. WP22a's `+hybrid` numbers remain the reference for the
  target PPC metric; this report's numbers should not be compared to them
  directly.
- **`--imu preint` used the same WP21b defaults as WP22a**, not
  re-tuned for this ablation; the IMU-preint arm's role here is unchanged
  from WP22a (a secondary axis crossed with the new likelihood knobs), not
  a subject of this task's own investigation.
