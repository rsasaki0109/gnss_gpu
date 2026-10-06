# RB-FGO-PF runtime (frozen copy)

This directory holds the runtime behind the RB-FGO-PF Tokyo PPC results in the
top-level README and `internal_docs/inuex35_tc_fgo_benchmark.md`. The files
are copied verbatim from the private research workspace `repro_tc_fgo` at
commit `fa57b7f` (2026-10-06). They have not been refactored, so the code
still carries its campaign history (WP numbers, environment-variable
switches). `wp13m_lambda_diag.py` was never committed there; this is its first
archived copy.

| file | role |
|---|---|
| `wp16_run_rbpf.py` | runner: decodes one run, drives the filter, writes `.pos` / `.npz` |
| `rbpf_fgo.py` | Rao-Blackwellized PF over integer-ambiguity basins |
| `gtsam_rtk_standalone.py` | fixed-lag GNSS/IMU factor-graph backbone (shared float filter) |
| `wp13a_run_standalone_rtk.py` | data loading, reference and `.pos` helpers |
| `wp13m_lambda_diag.py` | finite-guarded MLAMBDA used by the backbone |
| `relabel_nb9_pos.py` | report floor: relabels fixes with fewer than `nb` resolved ambiguities as float |

## Dependencies

All pinned; `setup_deps.sh` fetches them.

- Python 3.12 with the packages in `requirements.txt`, including the
  `inuex35/cssrlib-numba` fork at `dd90eb0`.
- `inuex35/tightly-coupled-gnss-imu-fgo` at `5cbfec4`, checked out as `tc/`
  (BSD-3-Clause). The runtime imports `gnss_fgo` from `tc/src`.
- `inuex35/gtsam` at `3c2f54c`, built with its Python wrapper. The PyPI
  `gtsam` wheel lacks the `DoubleDifference*Factor[Arm]` factors.

```bash
bash setup_deps.sh                    # venv + pip + tc + GTSAM build
bash setup_deps.sh --python PY --skip-pip --skip-gtsam   # reuse an existing environment
```

The tc stage was exercised on 2026-10-06. The GTSAM stage replays the CMake
options of the original build (Visual Studio 17 2022, Release, Python wrapper
on, TBB/Boost features off) but has not yet been run on a clean machine. The original build used plain
MSVC with the build tree on a data drive. It did not use vcpkg; a vcpkg/MKL
attempt filled the system drive.

## Running

The data root defaults to the original workstation path; set `PPC_DATA_ROOT`
to the `tokyo` directory of the PPC dataset (`run{1,2,3}/` with `rover.obs`,
`base.obs`, `base.nav`, `imu.csv`, `reference.csv`).

```bash
PPC_DATA_ROOT=.../PPC-Dataset-data/tokyo bash run_ship.sh ship_r3 3 16000 --checkpoint-every 1200
python relabel_nb9_pos.py results/ship_r3/run3.pos results/ship_r3/run3.npz ship_r3_nb12.pos 12
python ../score_vs_inuex35.py --traj ship_r3_nb12.pos --city tokyo --run run3 --format pos \
  --data-root .../PPC-Dataset-data
```

A full run takes 30–70 min (single process, about 5–8 epochs/s). The filter
is stochastic: the seed is 20260710 unless `--rb-seed` is given, and results
move by up to about 3 pp between seeds (see the WP39 section of the benchmark
record). Compare variants over several seeds.

Optional map-aided setting (WP39, not the default): `--env NLOS_ENABLE=1
--env NLOS_AR_EXCLUDE=1 --env NLOS_MASK_DIR=<dir>`. The directory holds
`tokyo_run{N}_per_epoch_nlos.csv` masks from
`experiments/build_per_epoch_nlos_csv.py`. Build them with
`--receiver-pos-file` pointing at a first-pass output. Without that flag the
builder ray-traces at the ground truth, which is an oracle.

## Checking the copy

`check_frozen_copy.py` compares the shipped mode, resolved-ambiguity count
and position over the common prefix of two runs. On 2026-10-06, 400-epoch
prefixes of run1 and run3 from this directory were bit-identical to the
archived WP18 full runs:

```bash
bash run_ship.sh smoke_r3 3 400 --checkpoint-every 0
python check_frozen_copy.py results/smoke_r3/run3.npz <archived full_r3/run3.npz>
```
