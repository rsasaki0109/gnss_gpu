# WP23b — PF-only integer-ambiguity basin AR

Date: 2026-07-18. Branch: `agent/wp23b-basin-ar`.

## Objective

Produce the first trustworthy RTK FIX from the PPC Tokyo particle-filter path
without any runtime FGO. The estimator is a Rao-Blackwellized particle filter
over integer-ambiguity basins. Each basin carries a discrete ambiguity lineage,
a cumulative marginal-likelihood weight, and a conditional Kalman filter.

The first measured target is the Tokyo run2 1200-epoch window:

- FIX is declared only when MAP-basin posterior mass `gamma > 0.99`;
- PF-only `<50cm_full%` is greater than zero; and
- false-fix rate is at most 1% on declared fixes.

Hybrid and truth trajectories may be used only as diagnostic/oracle arms. They
must not feed the production PF-only arm.

## Non-negotiable constraints

1. No sliding-window LM, GTSAM, or other FGO in the runtime positioning loop.
2. Preserve WP22b epoch tempering as a reproducibility baseline. New staged DD
   and AR updates use a separate annealed-SMC primitive which consumes the full
   observation likelihood (`sum(delta_beta) == 1`).
3. Do not seed LAMBDA from the diffuse position-particle cloud. Use an
   independent DD float KF with a joint position/ambiguity covariance.
4. Use `gnss_gpu.lambda_batch` for top-K generation once calls are genuinely
   batched. A CPU top-K implementation is allowed for the first basin prototype
   because the measured GPU batch-of-one path is slower.
5. Do not push or open a PR without the user's explicit approval.

## Work items and gates

### G1 — statistically valid staged tempering

Replace `_mupf_stage_update`'s one-shot ESS guard with annealed SMC:

- advance likelihood power from beta=0 to beta=1 using ESS-targeted bisection;
- resample and re-evaluate the observation likelihood between increments;
- never silently revert or discard a stage;
- accumulate the SMC log-normalizing-constant estimate for later basin weights;
- fail loudly if beta=1 cannot be reached within the configured step limit.

Tests must prove full beta consumption, more than one increment for a sharp
likelihood, finite evidence, and unchanged exact weights when no resampling is
needed. Re-run the WP23a run2/off diagnostic and compare with 20.677 m AllRMS.

### G2 — independent float seed

Implement a FGO-free local-coordinate KF with dynamic DD ambiguity tracks:

- position/velocity propagation from Doppler and optional IMU preintegration;
- DD pseudorange position update;
- DD carrier float-ambiguity update;
- joint position/ambiguity covariance suitable for LAMBDA;
- `(carrier - pseudorange) / wavelength`-style ambiguity initialization to
  cancel the common receiver-clock component;
- slip/outage generation changes and covariance inflation/reinitialization.

On run2/1200 report covariance SPD failures, normalized innovation statistics,
float position error, top-12/top-16 candidate oracle coverage, and the fraction
of DD-usable epochs for which at least one candidate conditions to <0.5 m.

### G3 — basin RBPF core

Implement 64-128 live basins with moves `{keep, adopt top-K, release, respawn}`.
Deduplicate identical active integer assignments, preserve lineage through
resampling, and accumulate conditional marginal likelihood across epochs.

Synthetic tests must cover wrong candidates, slips, outages/re-entry, basin
deduplication, and posterior-mass aggregation. The correct basin must survive
and gain posterior mass under coherent clean observations.

### G4 — PPC integration and first PF-only FIX

Add an opt-in method such as `rbpf+dd+ar+gate` to
`experiments/exp_ppc_ctrbpf_fgo.py`. Compare on the same run2/1200 window:

1. WP23a non-hybrid baseline;
2. basin AR with hybrid seed (diagnostic upper bound only);
3. basin AR with the independent float KF (production PF-only arm).

The hybrid-seed arm must first demonstrate that AR plumbing can produce a
correct FIX. The production arm passes when `<50cm_full% > 0`, declared-FIX
false rate is <=1%, and all FIX decisions use basin mass rather than a ratio
test. Report gamma calibration bins and retain ratio only as a diagnostic.

### G5 — purity and scale-up

Port the measured RB-FGO-PF lessons without porting FGO:

- trusted-DDPR commit gate;
- float/DDPR/fixed three-way output vote;
- minimum supported-DD count ablation (including `n_dd >= 9`);
- cluster-specific relinearization for coherent shifted basins;
- DDPR-based respawn after outage, with generation-aware slip handling.

Then run the 3-run 1200-epoch grid and full runs. Compare against the hybrid
floor (6-10.6%) and ultimately inuex35 (56.7/69.9/67.9%). A measured negative
with a root cause is acceptable; an unmeasured accuracy claim is not.

## Required artifacts

- implementation and focused tests;
- `results/wp23b/PROGRESS.md` maintained during development;
- scored CSVs and reproducible commands under `results/wp23b/`;
- `results/wp23b/WP23B_REPORT.md` with honest G1-G5 verdicts.
