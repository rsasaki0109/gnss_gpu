# WP23a Task Spec — DD Carrier-Phase AFV in the PF (multiple-update, no AR yet)

Follow-up to WP22b (`results/wp22b/WP22B_REPORT.md`). Part of the PF-only roadmap
(`internal_docs/pf_only_imu_roadmap_2026_07_17.md`, WP23 section).

## Why this task

WP22b established the non-hybrid DD-PR RBPF regime: tempering fixes weight degeneracy
(ESS/N 0.10, the only upgrade that paid), but `<50cm_full%` = 0.0% across all 36 cells —
pseudorange+Doppler alone never reaches sub-meter. The accuracy source must be carrier
phase. The device API already has fractional-cycle DD carrier likelihoods that need no
integer AR: `pf_device_weight_dd_carrier_afv` and the fused `pf_device_weight_dd_joint`
(see `include/gnss_gpu/pf_device.h`). The known hazard is likelihood sharpness and
lambda-spaced multimodality: a naive CP update collapses or mislocks the cloud.
Suzuki (ICRA 2024, "Multiple Update Particle Filter", arXiv:2403.03394) fixes this by
applying measurement families sequentially per epoch, ordered from least to most sharp,
with diversity protection between.

## Work items (in order)

1. **Wire DD carrier AFV into the non-hybrid path.** Extend
   `experiments/exp_ppc_ctrbpf_fgo.py` with a method variant (e.g. `rbpf+dd+cp+gate`)
   that adds a DD-carrier AFV update on top of the WP22b winner
   (`rbpf+dd+gate` + `enable_epoch_tempering`). PPC loader already provides
   `carrier_phase`; check how existing DD machinery selects the reference satellite and
   reuse it. Check first whether an existing runtime wrapper for
   `pf_device_weight_dd_carrier_afv` / `update_dd_joint` exists in
   `python/gnss_gpu/pf_device_runtime.py` — wire, don't reinvent.
2. **Multiple-update schedule (Suzuki ICRA 2024).** Per epoch: (i) DD-PR update →
   temper to ESS target → resample-if-needed; (ii) DD-CP AFV update → temper →
   resample-if-needed. Make the schedule and per-stage ESS targets configurable.
   If a single CP update is still too sharp, split satellites into groups and update
   sequentially (document the grouping).
3. **AFV parameter hygiene.** Wavelength(s) per constellation/frequency from the data,
   `sigma_cp` documented (start from the values tc_fgo/rbpf_fgo used), and an explicit
   note on the AFV's lambda-spaced multimodality: measure whether the post-PR cloud is
   tight enough (report the cloud's positional spread vs lambda/2 per epoch as a
   diagnostic). Cycle-slip handling: reuse whatever slip detection the DD machinery
   already has; if none is wired, gate CP usage on epoch-to-epoch phase continuity and
   document it.
4. **Ablation** on the same run1/2/3 1200-epoch windows as WP22b, 50k particles:
   {WP22b-best (DD-PR+temper), +CP-AFV multiple-update} x {imu off, preint}.
   Metrics: `<50cm_full%`, AllRMS, ESS/N, resample rate, plus the cloud-spread
   diagnostic. Score with `experiments/score_vs_inuex35.py`.
5. **Report** `results/wp23a/WP23A_REPORT.md`: does CP-AFV produce the first nonzero
   `<50cm_full%` from a PF-only pipeline? If not, the report must show the measured
   failure mode (collapse? mislock? spread-vs-lambda mismatch?) and state what WP23b
   (integer-ambiguity basins + GPU batch-LAMBDA + gamma posterior-mass fix decision)
   must provide that fractional-cycle AFV cannot.

## Gates

- G1: CP-AFV update runs in the non-hybrid path with the multiple-update schedule;
  unit-level sanity test that the AFV likelihood peaks at the true position on a
  synthetic DD-CP epoch.
- G2: ablation table complete (4 cells x 3 runs) with all metrics + diagnostics.
- G3: honest report; nonzero `<50cm_full%` OR a measured failure-mode diagnosis with a
  concrete WP23b requirement list.

## Constraints

- No FGO at runtime. No CUDA kernel edits (the kernels you need already exist).
  Don't touch the PPC production selector.
- Branch: `agent/wp23a-carrier` off `agent/wp22b-likelihood`. Commit as rsasaki0109.
  No push, no PR.
- Append milestones to `results/wp23a/PROGRESS.md` as you go.
- Run everything in the FOREGROUND (WP22a/b runs were seconds-to-minutes each). Do not
  end your turn with anything uncollected.
