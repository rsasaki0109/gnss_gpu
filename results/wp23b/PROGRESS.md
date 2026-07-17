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
| G3 basin RBPF core | pending | — |
| G4 first PF-only FIX | pending | — |
| G5 purity and scale-up | pending | — |

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
