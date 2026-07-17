# WP22b Task Spec — Expose the PF and Fix the Likelihood (NLOS/C-N0 GMM + tempering)

Follow-up to WP22a (`results/wp22a/WP22A_REPORT.md`). Part of the PF-only roadmap
(`internal_docs/pf_only_imu_roadmap_2026_07_17.md`, WP22 section).

## What WP22a established

1. In the `rbpf+dd+gate+hybrid` config, the hybrid RTK position-update dominates
   emission on 82-98% of epochs — the PF's predict/weight steps are second-order,
   so `<50cm_full%` was byte-identical between IMU arms. **To improve the PF we must
   measure the PF.**
2. The DD-RBPF is weight-degenerate everywhere: ESS/N ~1e-4..1e-5, resampling every
   epoch, 50k particles → effectively a handful carry all weight. Diagnosis:
   sharp DD/PR likelihood over ~20+ satellites vs a meters-wide particle cloud.
   IMU-preint raises ESS/N ~5x but cannot fix a likelihood-side mismatch.

## Work items (in order)

1. **Non-hybrid PF-dominant baseline.** Using `experiments/exp_ppc_ctrbpf_fgo.py`,
   run the strongest non-`+hybrid` DD-RBPF variant (e.g. `rbpf+dd+gate`) on the same
   run1/2/3 1200-epoch windows as WP22a, arms `--imu off` and `--imu preint`.
   This table is the reference for everything below.
2. **Adaptive likelihood tempering.** Add a per-epoch temperature beta on the
   log-likelihood chosen adaptively to target a configurable ESS/N (default 0.10):
   after computing raw per-particle log-weights, solve for beta in (0,1] such that
   ESS(beta)/N ≈ target (bisection on the host over the log-weight vector is fine —
   pull log-weights once per epoch; no CUDA edits needed if log-weights are readable,
   else compute the update host-side for this experiment). Document the estimator
   consistency caveat (tempering trades statistical efficiency for diversity) in the
   report. Measure: ESS/N, resample rate, AllRMS, `<50cm_full%` vs item 1.
3. **C/N0- and elevation-driven GMM likelihood.** The device API has a LOS/NLOS
   mixture kernel (`pf_device_weight_gmm`, see `include/gnss_gpu/pf_device.h`) and the
   repo has a validated C/N0 predictor (see `results/validation/` and the commit
   "Add C/N0 validation against measured UrbanNav signal strength"). Drive the
   mixture weight w_los per satellite from measured C/N0 + elevation (simple logistic
   or lookup calibrated on the validation data; document the mapping). If the kernel
   only accepts scalar GMM parameters, first try per-satellite grouping (multiple
   kernel calls over satellite subsets sharing parameters); only if that is
   impractical, a minimal CUDA parameter extension is permitted THIS TASK ONLY —
   keep it to adding a per-satellite parameter array to the existing GMM kernel,
   rebuild, and run the existing PF test suite to confirm no regression.
4. **Particle-wise NLOS deweighting (Niimi-style).** The undifferenced weight kernel
   already supports a per-particle NLOS threshold + Huber. Enable/configure it (or
   its DD analogue) so the rejection set varies per particle. Measure its effect
   separately from item 3.
5. **Final ablation grid** on run2 (minimum; run1/3 if wall-clock permits):
   {baseline, +tempering, +GMM(C/N0), +particle-NLOS, all-on} x {imu off, preint}.
   Score with `experiments/score_vs_inuex35.py` + health stats.

## Gates

- G1: non-hybrid baseline table complete (item 1).
- G2: tempering raises mean ESS/N by ≥10x without degrading AllRMS on run2 (if it
  degrades, measure and diagnose — that itself is a valid outcome, but must be measured).
- G3: full ablation grid complete with honest per-cell numbers; report at
  `results/wp22b/WP22B_REPORT.md` states which likelihood upgrades actually pay and
  a concrete recommendation for WP22c (BVH ray-traced LOS/NLOS priors) and WP23 (AR).

## Constraints

- No FGO at runtime. CUDA edits only as narrowly permitted in item 3. Don't touch the
  PPC production selector.
- Branch: `agent/wp22b-likelihood` off `agent/wp22a-dd-imu`. Commit as rsasaki0109.
  No push, no PR.
- Append milestones to `results/wp22b/PROGRESS.md`. Do not end your turn with a
  background run uncollected.
