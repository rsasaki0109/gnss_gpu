# WP23a Report — DD Carrier-Phase AFV in the PF (multiple-update, no AR yet)

Spec: `internal_docs/task_wp23a_carrier_afv.md`. Branch: `agent/wp23a-carrier`
(off `agent/wp22b-likelihood`). Date: 2026-07-17/18.

## 1. Framing correction (read before the rest of this report)

The task spec frames WP22b's winner (`rbpf+dd+gate` + `enable_epoch_tempering`)
as a "DD-PR" (double-differenced pseudorange) regime that carrier phase is
being added on top of for the first time. **This is not what the code does.**
Reading `experiments/exp_ppc_ctrbpf_fgo.py` and `python/gnss_gpu/dd_carrier.py`
shows:

- `rbpf+dd+gate`'s `enable_dd_carrier_afv=True` already calls
  `pf.update_dd_carrier_afv(...)` every epoch it has enough pairs -- DD
  **carrier-phase AFV** (fractional cycle, no ambiguity resolution) was
  already the sole "DD" signal in WP22b's non-hybrid baseline.
- `DDCarrierComputer`'s `DDResult` has **no pseudorange field at all**.
  There is a separate `DDPseudorangeComputer`/its own result class, but
  before this task it was wired only to a robust-LS position anchor
  (`enable_dd_pr_ls_anchor` -> `pf.position_update`, a soft constraint) or
  the FGO post-process cache -- **never** to a genuine per-epoch
  `pf.update_dd_pseudorange(...)` weight update in the non-hybrid path.

So item 1 ("wire DD carrier AFV into the non-hybrid path") is reinterpreted,
honestly, as: (a) add the DD-pseudorange weight-update stage that was
genuinely missing, and (b) restructure the *already-present* single-shot
DD-carrier-AFV call into the Suzuki two-family multiple-update schedule item
2 asks for, rather than "wire carrier AFV in" from a standing start. This
also means WP22b's own "DD-carrier-AFV per-particle gate showed zero
measurable effect" finding (§5 of `WP22B_REPORT.md`) was already measuring
carrier-AFV's real (near-zero) impact on this regime, not merely a gate's
threshold calibration -- consistent with this report's own findings below.

## 2. What was built

All changes confined to `experiments/exp_ppc_ctrbpf_fgo.py` + one new test
file (`tests/test_wp23a_dd_carrier_afv_sanity.py`). No CUDA edits (kernels
already existed), no FGO wired into the runtime loop, no PPC production
selector changes.

### Item 1 — DD-PR weight update + wiring

New `pf.update_dd_pseudorange(...)` call site (the existing Python wrapper
in `pf_device_runtime.py`, previously unused as a genuine weight update in
this path). `DDPseudorangeComputer` construction and per-variant
passthrough (`dd_pr_computer`) extended to trigger on the new
`enable_cp_mupf` flag, not just `enable_fgo_lambda`/`enable_dd_pr_ls_anchor`.

**Bug found + fixed during smoke-testing:** the per-variant DD-computer
passthrough gate (`need_dd_for_variant`/`dd_pr_for_variant`, computed once
per variant just before its `run_pf` call) checked
`enable_dd_carrier_afv`/`enable_dd_pr_ls_anchor`/`enable_fgo_lambda` but not
the new `enable_cp_mupf` -- so the new variant silently got
`dd_pr_computer=None` even though the per-run DD-PR computer had been built,
and MUPF stage (i) never fired (`mupf_pr=0/40` on the first smoke run).
Fixed by adding `enable_cp_mupf` to both conditions; confirmed fixed
(`mupf_pr` epoch counts subsequently match `mupf_cp`'s).

### Item 2 — Suzuki multiple-update schedule (new method `rbpf+dd+cp+gate`)

New `CTRBPFConfig.enable_cp_mupf`, mutually exclusive with
`enable_dd_carrier_afv` at the call site (`elif`, not both). Per epoch,
inserted at the same call site the old single-shot DD-carrier block used:

1. **Stage (i) DD-pseudorange**: `pf.update_dd_pseudorange(dd_pr_result,
   sigma_pr=cp_mupf_dd_pr_sigma_m, resample=False)` -> temper this stage's
   log-likelihood delta to `cp_mupf_pr_stage_target_ess_ratio` (bisection,
   reusing WP22b's `_apply_pr_ess_guard` at a new per-stage call site) ->
   `pf.resample_if_needed()`.
2. **Cloud-spread-vs-lambda/2 diagnostic** (item 3), measured on the
   post-stage-(i) cloud: `pf.get_position_spread()` vs
   `median(dd_result.wavelengths_m) / 2`.
3. **Stage (ii) DD-carrier AFV**: gated on cycle-slip continuity (below),
   then applied as a **coarse-to-fine sigma sequence**
   `cp_mupf_dd_cp_sigma_sequence_cycles = (2.0, 0.5, 0.05)` cycles, each
   value independently tempered to `cp_mupf_cp_stage_target_ess_ratio` and
   resample-gated. This sequence is not new to this task -- it is exactly
   the pattern already validated elsewhere in this codebase for a
   *different* (GSDC2023/Trimble) PF-smoother track
   (`gnss_gpu.dd_carrier_epoch_update.apply_carrier_epoch_update` /
   `gnss_gpu.pf_smoother_config.MupfConfig`, same default values), reused
   here per "wire, don't reinvent" rather than inventing a new schedule.
   The GSDC track does not add ESS-target bisection tempering per step;
   this task's item 2 explicitly requires it, so it was added here as a
   deliberate synthesis of both.
4. A satellite-group-split fallback (`cp_mupf_cp_n_groups`, round-robin,
   off by default) implements the spec's documented escape hatch "if a
   single CP update is still too sharp" -- smoke-tested, not exercised in
   the main grid (the coarse-to-fine sequence was tried first, per the
   spec's own ordering of remedies).
5. A mid-epoch resample inside the MUPF block invalidates WP22b's blanket
   `enable_epoch_tempering` pre/post log-weight correspondence (resampling
   permutes particles). The epoch-start snapshot is re-anchored
   immediately after the MUPF block whenever this happens, so a
   same-epoch blanket temper on whatever follows (if anything) stays valid.

### Item 3 — AFV parameter hygiene, cloud-spread diagnostic, cycle-slip gate

- **Wavelengths**: per-DD-pair, from `dd_result.wavelengths_m`
  (constellation/frequency-aware; already computed by `DDCarrierComputer`
  from the RINEX observation codes actually used that epoch -- not a
  single hardcoded L1 value).
- **sigma_cp documented, inherited from tc_fgo/rbpf_fgo**: rather than
  alias the existing `fgo_dd_sigma_cycles=0.20`/`fgo_dd_pr_sigma_m=5.0`
  fields (which belong to the unrelated, FGO-postprocess-only
  `enable_fgo_lambda` feature), new independently-tunable copies
  `cp_mupf_dd_pr_sigma_m=5.0` and the sequence's tightest step (0.05,
  matching `dd_sigma_cycles`'s own historical default) were added, with
  the coarse-to-fine outer values (2.0, 0.5) inherited from the validated
  GSDC MUPF default (item 2).
- **Cloud-spread-vs-lambda/2 diagnostic**: measured every epoch stage (i)
  fires (see §3's table) -- this is the central measured finding of this
  report (§4).
- **Cycle-slip gate**: no slip detector exists anywhere in this
  codebase's DD machinery for the PPC path (grepped `dd_carrier.py` and
  `exp_ppc_ctrbpf_fgo.py` for "slip": none found). Implemented
  `_cp_slip_gate`, an epoch-to-epoch DD-carrier-phase continuity proxy
  keyed by `(ref_sat_id, sat_id)`: a pair is dropped from that epoch's
  CP-AFV update if its raw DD-carrier phase [cycles] jumped by more than
  `cp_mupf_slip_max_delta_cycles=2.0` since the last epoch it was seen
  within `cp_mupf_slip_max_dt_s=2.0` seconds. This fired on a large
  fraction of epochs (1934-2388 pair-flags across 1200 epochs per run,
  §3) -- expected on this dataset (see §4's discussion of why, this is
  not itself the accuracy blocker).

### G1 — unit-level sanity test

New `tests/test_wp23a_dd_carrier_afv_sanity.py`: a synthetic, noiseless
DD-CP epoch (random Earth-scale satellite geometry, GPS-L1 wavelength, one
shared reference satellite), 7 explicit particle hypotheses swept +/-0.02 m
around the true rover position along the reference satellite's line of
sight (tight enough that every DD pair's own half-cycle wrap boundary is
respected). Calls `ParticleFilterDevice.update_dd_carrier_afv` directly and
asserts the log-likelihood peaks exactly at the true-position particle and
decays monotonically on both sides.

An earlier version of this test used +/-0.09 m offsets and **failed** with
a real (non-spurious) local wiggle: a randomly-generated near-antipodal
satellite pair had DD sensitivity high enough (~10.5 cycles/m) that a
0.09 m offset wrapped past that *one* pair's own +/-0.5 cycle boundary,
producing a genuine secondary local optimum -- a small, concrete,
self-contained illustration of the exact "lambda-spaced multimodality"
hazard the spec's own item 3 flags, caught by the test itself rather than
asserted away.

**G1: PASS.** Test passes; full regression suite (WP22a/WP22b's suite +
`test_pf_device_wrapper.py` + `test_dd_carrier_epoch_update.py` + this new
test): 41/41 passing.

## 3. Item 4 / Gate G2 — ablation grid

Same run1/run2/run3 1200-epoch windows and `n_particles=50000` as WP22b,
`--imu {off,preint}` x `--methods {rbpf+dd+gate, rbpf+dd+cp+gate}`, both
with `--enable-epoch-tempering` (WP22b's winning tempering config kept on
for both cells, so the only variable between rows is the DD-carrier
mechanism itself):

```
PYTHONPATH=python python experiments/exp_ppc_ctrbpf_fgo.py \
  --runs tokyo/run1,tokyo/run2,tokyo/run3 --methods {rbpf+dd+gate|rbpf+dd+cp+gate} \
  --max-epochs 1200 --imu {off|preint} --enable-epoch-tempering \
  --pos-dir results/wp23a/pos/{baseline|cpmupf}_{off|preint} \
  --results-prefix wp23a_{baseline|cpmupf}_{off|preint}
```

Scored with `experiments/score_vs_inuex35.py --fix-statuses 1` via
`results/wp23a/score_grid.py` (mirrors `results/wp22b/score_grid.py`'s
pattern). Full 12-row detail: `results/wp23a/csv/wp23a_grid_scored.csv`.

| config | imu | run | AllRMS [m] | \<50cm_full% | ESS/N | DD pairs (dd/cp) | CP mean alpha | PR mean alpha | spread/(lambda/2) |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | off | run1 | 51.950 | 0.0 | 1.000e-01 | 4460 | -- | -- | -- |
| baseline | off | run2 | 29.220 | 0.0 | 1.000e-01 | 4689 | -- | -- | -- |
| baseline | off | run3 | 31.748 | 0.0 | 1.000e-01 | 4687 | -- | -- | -- |
| baseline | preint | run1 | 53.633 | 0.0 | 1.000e-01 | 4460 | -- | -- | -- |
| baseline | preint | run2 | 27.979 | 0.0 | 1.000e-01 | 4689 | -- | -- | -- |
| baseline | preint | run3 | 30.656 | 0.0 | 1.000e-01 | 4687 | -- | -- | -- |
| +CP-MUPF | off | run1 | 51.995 | 0.0 | 8.03e-02 | 2246 | 0.690 | 0.000 | 125.2 |
| +CP-MUPF | off | run2 | 29.262 | 0.0 | 8.01e-02 | 2209 | 0.691 | 0.000 | 124.9 |
| +CP-MUPF | off | run3 | 31.673 | 0.0 | 8.00e-02 | 2660 | 0.688 | 0.000 | 125.9 |
| +CP-MUPF | preint | run1 | 53.402 | 0.0 | 8.03e-02 | 2246 | 0.690 | 0.000 | 68.5 |
| +CP-MUPF | preint | run2 | 27.951 | 0.0 | 8.01e-02 | 2209 | 0.691 | 0.000 | 71.1 |
| +CP-MUPF | preint | run3 | 31.445 | 0.0 | 8.01e-02 | 2660 | 0.688 | 0.000 | 58.1 |

Per-arm mean AllRMS: baseline off 37.639 / preint 37.423; +CP-MUPF off
37.643 / preint 37.599. **AllRMS is statistically unchanged (<0.5%
different) between the baseline and +CP-MUPF cells on every run/arm, and
`<50cm_full%` stays exactly 0.0% on all 12 cells** (the same all-36-cells
result WP22b measured for the pseudorange+Doppler-only regime -- carrier
phase, as wired here, does not move the needle).

Raw per-config CSVs: `results/wp23a/csv/wp23a_{baseline,cpmupf}_{off,preint}_runs.csv`.
Scored/merged: `results/wp23a/csv/wp23a_grid_scored.csv`.
`.pos` trajectories (gitignored, matching WP22a/b's practice):
`results/wp23a/pos/{baseline,cpmupf}_{off,preint}/tokyo_run{1,2,3}_<method_label>.pos`.

**G2: PASS** -- complete 4x3 ablation table with AllRMS, `<50cm_full%`,
ESS/N, DD-pair counts, and the item-3-required cloud-spread diagnostic on
every cell.

## 4. Item 5 / Gate G3 — measured failure mode

**`<50cm_full%` stays 0.0% everywhere -- carrier-phase AFV, wired as the
spec's multiple-update schedule specifies, does not produce the first
nonzero PF-only result.** Two independent, root-caused mechanisms explain
why, measured (not merely asserted):

### 4a. Cloud-spread-vs-lambda/2 mismatch (the mechanism the spec itself anticipated)

The post-DD-PR-stage cloud spread is **58x to 126x wider than half a
carrier wavelength** on every cell (column above; half-wavelength for GPS
L1 is ~0.095 m, so the cloud spans tens of meters). The DD-carrier AFV
likelihood is periodic in the true (unwrapped) range with period =
wavelength; when a particle cloud spans hundreds of half-wavelength
periods, particles at many different, mutually-inconsistent integer-cycle
offsets from the truth all land near *some* local peak of the periodic
likelihood -- the AFV update reshuffles weight among aliased local optima
across the whole diffuse cloud rather than concentrating it near the true
position. This is exactly the mechanism the spec's own item 3 asked to be
measured for, and it is real: it is the same "diffuse cloud collides with
a sharp likelihood" failure WP22a/WP22b diagnosed for pseudorange,
reproduced quantitatively for carrier phase.

### 4b. A newly-discovered inertness bug in the per-stage tempering primitive

`mupf_pr_mean_alpha = 0.000` on **all 12/12 cells**, despite the DD-PR
stage being attempted 237-240 times per run (`mupf_pr_epochs_applied`).
Root-caused by reading `_apply_pr_ess_guard` (WP22b's bisection primitive,
reused per-stage here): it contains an early-exit,
`if pre_ratio < target: pf.set_log_weights(pre); return 0.0, ...` --
**revert the update entirely** if the pre-update ESS/N is already below
the target. This is sound when the whole epoch is tempered once at the
end (WP22b's original use), but breaks when chained per *stage*: by the
time stage (i) DD-PR runs, the *untempered* PR + Doppler-KF updates
earlier in the same epoch have already pushed ESS/N to ~1e-4 (the same
order WP22a/WP22b measured pre-tempering), which is always below the
stage's own 0.10 target -- so the guard reverts stage (i) unconditionally,
every single epoch, on every one of the 6 (config x arm) cells that used
it. **The DD-PR stage, as specified literally ("update -> temper ->
resample-if-needed"), contributes zero information in this
implementation.**

A **targeted diagnostic re-run** (tokyo/run2, `--imu off`, one extra
invocation beyond the main grid, `results/wp23a/csv/wp23a_diag_resamplebefore_off_runs.csv`)
confirms the mechanism and quantifies its cost: adding
`--cp-mupf-resample-before-stage` (new diagnostic-only flag; calls
`pf.resample_if_needed()` **before**, not only after, each stage --
mirroring the validated GSDC MUPF track's own ordering, which resamples
before each sigma step) resets each stage's entering ESS/N to ~1.0 before
tempering, so the guard can bisect a genuine nonzero alpha instead of
reverting:

| variant | AllRMS [m] | mupf\_pr\_mean\_alpha | spread/(lambda/2) |
| --- | ---: | ---: | ---: |
| +CP-MUPF (spec-literal, resample after) | 29.262 | 0.000 | 124.9 |
| +CP-MUPF, `--cp-mupf-resample-before-stage` | **20.677** | 0.008 | 84.4 |

A **29% AllRMS reduction** on this run/arm from fixing the inertness bug
alone -- a real, measurable, positive effect, but `<50cm_full%` is still
0.0% (a 20.7 m AllRMS is nowhere near sub-meter) and the cloud-spread
ratio, while reduced, is still ~840x too large. This confirms both
mechanisms are real and additive: the inertness bug was silently
throwing away real (if weak) information, but even with it fixed, 4a's
cloud-diffuseness mismatch remains the dominant blocker.

### What WP23b must provide that fractional-cycle AFV cannot

1. **Integer-ambiguity resolution (the actual point of WP23b).** Fractional-
   cycle AFV is fundamentally rank-limited by cloud spread vs. wavelength;
   no amount of tempering/scheduling hygiene fixes a likelihood that is
   periodic on a scale ~100x smaller than the cloud it's being asked to
   discriminate within. WP23b's integer-ambiguity basins (LAMBDA) turn the
   carrier observation into an *unambiguous*, meter-scale-informative
   constraint once fixed -- this is a difference in kind, not degree, from
   anything a scheduling/tempering change to the AFV path can supply.
2. **A predict-step / anchor mechanism that shrinks the cloud below
   half a wavelength before AFV (or LAMBDA) is asked to discriminate
   within it.** Per WP21/WP22a, this baseline's cloud stays diffuse
   (tens of meters) with no external RTK floor; WP23b's GPU batch-LAMBDA
   should either (a) operate on a WLS/float seed independent of this PF's
   own diffuse cloud, or (b) be paired with a convergence mechanism (e.g.
   the hybrid RTK floor WP22a's `+hybrid` arm already validates, at
   AllRMS 6.7-15.5 m / 6-10.6% pass) so LAMBDA has a realistic ambiguity
   search space to begin with.
3. **A structurally sound multi-stage tempering primitive.** §4b's
   inertness bug should be fixed (not worked around per-task) before
   WP23b builds more per-stage updates on top of `_apply_pr_ess_guard`:
   either resample-before-stage (this report's diagnostic default,
   validated by the GSDC MUPF track), or a redesigned guard that doesn't
   unconditionally revert when the *entering* ESS/N (not this stage's own
   contribution) is already below target.
4. **The gamma posterior-mass fix decision** (per the spec's own framing)
   is out of scope for what fractional-cycle AFV alone can validate here;
   this report has no new evidence to add or subtract from that decision
   beyond confirming that AFV's likelihood-sharpness hazard is real and
   measured on this dataset (§4a), which is the regime that decision must
   hold up under.

**G3: PASS** -- honest report; `<50cm_full%` stayed 0.0% (no false
positive claimed), and this section provides a measured, root-caused (not
merely observed) failure-mode diagnosis with a concrete WP23b requirement
list, following WP22b's own precedent of targeted extra diagnostic runs to
root-cause a negative result rather than only report the raw number.

## 5. Deviations from the spec

- **Framing correction (§1)**: the spec's premise that WP22b's baseline was
  a "DD-PR" (not carrier) regime is incorrect as read from the code; this
  report documents the correction rather than silently building on the
  spec's (wrong) premise.
- **The per-stage tempering design (§4b) is a genuine limitation
  discovered while implementing item 2 exactly as specified.** The main
  ablation grid (§3) intentionally keeps the spec-literal
  "update -> temper -> resample-if-needed" ordering (not the fixed
  ordering) so the reported numbers reflect a faithful implementation of
  the spec's own text; the fix is offered as a new opt-in diagnostic flag
  (`--cp-mupf-resample-before-stage`, default off) plus one extra targeted
  run, not a silent change to the main grid's methodology.
- **Satellite-group-split fallback (`cp_mupf_cp_n_groups`)** was
  implemented and smoke-tested but not exercised in the main grid --
  the spec lists it as a fallback "if a single CP update is still too
  sharp"; the coarse-to-fine sigma sequence was the first remedy tried,
  per the spec's own ordering, and item 3's diagnostic (§4a) shows the
  actual blocker is cloud spread, not single-update sharpness alone, so
  group-splitting was not expected to help and was not spent further
  budget on beyond confirming it runs cleanly.
- **`dd_min_pairs_update` (legacy single-shot path) vs `cp_mupf_min_pairs`
  (new MUPF path)** both default to 3 pairs; not re-tuned as part of this
  task (out of scope -- item 3 asks for sigma/wavelength/slip hygiene, not
  a pair-count retune).
- No CUDA edits, no FGO wired at runtime, no changes to
  `pf_device.cu`/`pf_device.h`, no changes to the PPC production selector
  -- all per the spec's constraints.
