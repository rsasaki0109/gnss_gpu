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
| G2 independent float seed | pending | — |
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
