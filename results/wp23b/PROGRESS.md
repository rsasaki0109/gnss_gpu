# WP23b Progress — PF-only basin AR

## 2026-07-18 — start

- Created branch `agent/wp23b-basin-ar` from `bc631a3`.
- Fixed the staged architecture and gates in
  `internal_docs/task_wp23b_basin_ar.md`.
- First implementation target: replace WP23a's stage-local one-shot ESS guard
  with annealed SMC that consumes beta=1 and records marginal likelihood.
- Existing untracked workspace files were left untouched.

## Live gate status

| Gate | Status | Evidence |
| --- | --- | --- |
| G1 annealed staged tempering | **pass** | 52 tests; run2/1200 AllRMS 10.742 m |
| G2 independent float seed | **pass, limited** | SPD 0 failures; partial top-16 oracle coverage 25.94% |
| G3 basin RBPF core | **pass** | 5 focused basin tests; versioned slips, dedup, cumulative evidence |
| G4 first PF-only FIX | **pass (run2/1200)** | gamma 0.995996; 14/14 correct declared FIX epochs |
| G5 purity and scale-up | **partial pass** | Tokyo 3-run/1200: 24/24 correct FIX; full runs pending |

## 2026-07-18 — G1 annealed SMC

Implemented `python/gnss_gpu/annealed_smc.py` and changed the WP23a MUPF
stage primitive to consume every staged observation likelihood through
beta=1. The existing WP22b whole-epoch flattening primitive remains unchanged
for baseline reproducibility.

Validation:

- `python -m pytest -q tests/test_annealed_smc.py ...`: **52 passed** across
  the new tests and the WP21-WP23a regression subset.
- A 40-epoch real-data smoke consumed beta=1 for every applied stage. DD-PR
  required 3.125 tempering increments on average; DD-CP required 1.625.
- Tokyo run2, first 1200 epochs, 50k particles, IMU off,
  `rbpf+dd+cp+gate`, epoch tempering enabled:

| metric | WP23a resample-before diagnostic | WP23b annealed SMC |
| --- | ---: | ---: |
| AllRMS [m] | 20.677 | **10.742** |
| `<50cm_full%` | 0.000 | **0.066** (5/1200 local passes) |
| DD-PR mean consumed beta | partial (`alpha=0.008`) | **1.000** |
| DD-PR mean tempering steps | not recorded | 3.795 |
| DD-CP mean consumed beta | partial | **1.000** |
| DD-CP mean tempering steps/call | not recorded | 1.517 |
| cloud spread / half wavelength | 84.4 | **10.08** |

Command:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONIOENCODING='utf-8'
python experiments/exp_ppc_ctrbpf_fgo.py --runs tokyo/run2 `
  --methods rbpf+dd+cp+gate --max-epochs 1200 --imu off `
  --enable-epoch-tempering `
  --pos-dir results/wp23b/pos/g1_annealed_off `
  --results-prefix wp23b_g1_annealed_off
```

## 2026-07-18 — G3/G4 integer basins and first FIX

Implemented `ambiguity_basin_pf.py` with versioned integer assignments and a
six-state ECEF position/velocity KF conditional per basin. Candidate weights
accumulate KF marginal likelihood across epochs; identical assignments are
deduplicated by Gaussian moment matching and stale slip generations are
discarded. Five synthetic tests cover clean-basin concentration, fixed-carrier
likelihood, deduplication, release, respawn, and population capping.

The PPC production arm uses the independent DD float KF, lowest-variance eight
ambiguities, CPU top-12 LAMBDA, 1% candidate birth mass, and at most 128 live
basins. FIX requires the same assignment to have `gamma > 0.99` for three rover
epochs and its conditional position to agree with the independent float KF
within 0.5 m. Reference truth is joined only after the decision.

Tokyo run2/1200 result: max gamma **0.995996**, 26 gamma-qualified epochs,
12 rejected by the independent consistency gate, 14 declared FIX epochs,
14 correct, 0 false. Scoring: FixRMS **0.181 m**, `<50cm_full%=1.7`, and
AllRMS 6.262 m. A gamma-only ablation produced 12/26 false fixes, proving the
consistency gate is necessary rather than cosmetic.

The main PPC runner now exposes this arm as the sole opt-in method
`--methods rbpf+dd+ar+gate`; it dispatches to the compact FGO-free harness.
G5 multi-run scale-up remains open.

Scored artifact: `results/wp23b/csv/g1_annealed_run2_score.csv`.

Important interpretation: the runner writes Status=1 for the whole PF
trajectory, so the scorer's `fix%=100` is not an AR claim. The five sub-50 cm
epochs prove that full likelihood consumption improved the float position-PF;
they do **not** satisfy G4. A real FIX remains gated on basin mass gamma>0.99.

## 2026-07-18 — G2 independent DD float KF

Implemented `python/gnss_gpu/dd_float_kf.py`:

- ECEF position/velocity constant-velocity KF with acceleration process noise;
- dynamic `(reference satellite, satellite, wavelength)` DD ambiguity tracks;
- DD-PR and DD-CP measurement updates with joint position/ambiguity covariance;
- same-epoch `CP_cycles - PR_m / wavelength` ambiguity roots;
- generation reset for carrier innovations above the slip threshold;
- stale/released track removal;
- `ahat`, `Qahat`, and position/ambiguity cross-covariance export;
- Gaussian position conditioning on an integer candidate.

Five focused synthetic tests cover convergence, SPD covariance, integer
conditioning, prediction, release/outage, and slip generation reset.

`experiments/exp_wp23b_float_seed.py` is the real PPC audit harness. It runs
the filter without truth input, accumulates variable-dimension LAMBDA problems,
then evaluates them in one genuine GPU batch. Truth is joined only afterward
to score candidate coverage.

Run2/1200 findings:

| metric | measured |
| --- | ---: |
| covariance SPD failures | **0 / 1200** |
| float position mean / median / p90 [m] | 5.006 / 4.117 / 10.985 |
| DD-PR NIS median / p90 | 0.249 / 0.882 |
| DD-CP NIS median / p90 | 0.563 / 20.455 |
| ambiguity generation resets | 422 |
| DD/LAMBDA epochs | 239 |
| full 29-ish ambiguity top-16 sub-50cm coverage | 3 / 239 = 1.26% |
| lowest-variance 6 ambiguity top-16 coverage | 57 / 239 = 23.85% |
| lowest-variance 8 ambiguity top-16 coverage | **62 / 239 = 25.94%** |

The measured negative is specific: a full-dimensional fix is not viable yet,
but partial 6-8 ambiguity basins have enough correct-candidate supply to build
and test cumulative basin weighting. This selects the G3 MVP design; it is not
an RTK FIX claim by itself.

Artifacts:

- `results/wp23b/csv/float_seed_partial_run2_epochs.csv`
- `results/wp23b/csv/float_seed_partial_run2_summary.json`

Command:

```powershell
$env:PYTHONPATH='python'
$env:PYTHONIOENCODING='utf-8'
python experiments/exp_wp23b_float_seed.py --max-epochs 1200 `
  --out-epochs results/wp23b/csv/float_seed_partial_run2_epochs.csv `
  --out-summary results/wp23b/csv/float_seed_partial_run2_summary.json
```

## 2026-07-18 — G5 trusted-DDPR commit and Tokyo 3-run grid

The original Float/basin agreement gate did not generalize safely. On the
unmodified Tokyo 3-run/1200 grid, run1 declared 14 fixes with 4 false
(28.57%), run2 remained 14/14 correct, and run3 declared none. The run1 false
fixes occurred while Float and the integer basin coherently drifted together.

Added a third, carrier-independent navigation arm: a DDPR+Doppler-only KF.
Production FIX now additionally requires MAP-to-DDPR separation <=1.75 m, at
least 9 DDPR pairs in the most recent trusted update, and DDPR age <=4 rover
epochs.

The final same-window grid declared **24/24 correct fixes, 0 false**: run1
10/10, run2 14/14, run3 0/0. Run1 FixRMS improved from 0.496 m to 0.211 m
because the four drifting fixes were rejected. The `n_dd` ablation from 0 to
18 did not change decisions in these windows; the active discriminator was
DDPR-only position consistency. `n_dd=24` reduced run1 to 4 correct fixes, so
9 remains a conservative support floor rather than a tuned accuracy lever.

Artifacts: `csv/g5_tokyo_3run_summary.{csv,json}`,
`csv/g5_min_dd_ablation.{csv,json}`, per-run `g5_trusted_*` diagnostics/scores,
and `pos/g5_trusted/` trajectories. Full-run scaling and cluster-specific
relinearization remain pending.

### G5 scale-up performance preparation

Two exact optimizations reduced the Tokyo 3-run/1200 wall time from 334.0 s to
195.2 s (**41.6% reduction, 1.71x speedup**):

1. DD batches with more than six rows now use the determinant lemma and
   Woodbury/information-form update, factoring only the six-state precision
   instead of a 20-30 row innovation covariance.
2. DD carrier and DD pseudorange computers share an explicit parsed-RINEX
   cache, avoiding duplicate base/rover observation parsing.

Across all 3600 epochs, the optimized arm had zero FIX or gamma-FIX decision
differences, maximum position delta 2.23e-8 m, and maximum gamma delta 5.01e-7.
The related regression subset passed with **75 passed, 2 skipped**. Evidence:
`csv/g5_optimization_benchmark.json`.
