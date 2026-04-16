# Real-Ephemeris E2E Summary

Date: 2026-04-15

This note records the real-data end-to-end checks after:

- `Ephemeris.compute_batch()` was fixed to reselect broadcast ephemeris per epoch.
- `UrbanSignalSimulator.compute_epoch()` started applying satellite clock correction via `sat_clk`.
- Windows editable installs were fixed so local `.pyd` extensions load correctly.
- Acquisition peak picking was upgraded to sub-sample parabolic interpolation.

## Data used

Downloaded locally with:

```powershell
python experiments\fetch_urbannav_subset.py --run Odaiba --output-dir experiments\data\urbannav
python experiments\fetch_urbannav_subset.py --run Shinjuku --output-dir experiments\data\urbannav
python experiments\fetch_plateau_subset.py --run-dir experiments\data\urbannav\Odaiba --preset tokyo23 --output-dir experiments\data\plateau_odaiba --mesh-radius 1
python experiments\fetch_plateau_subset.py --run-dir experiments\data\urbannav\Shinjuku --preset tokyo23 --output-dir experiments\data\plateau_shinjuku --mesh-radius 1
```

These datasets stay untracked because `experiments/data/` is ignored.

## Commands run

```powershell
$env:PYTHONPATH='python'; python experiments\verify_real_ephemeris.py
$env:PYTHONPATH='python'; python experiments\exp_e2e_trajectory.py
$env:PYTHONPATH='python'; python experiments\exp_e2e_positioning.py
```

## Real-ephemeris LOS/NLOS check

Script: `experiments/verify_real_ephemeris.py`

- Odaiba PLATEAU subset: `296,296` triangles, `211,555` BVH nodes.
- UrbanNav Odaiba trajectory: `25` sampled epochs available, `15` frames rendered.
- Each rendered epoch used `32` GPS satellites from broadcast ephemeris.
- Observed per-frame counts:
  - LOS: `5` to `9`
  - NLOS: `0` to `3`
  - Multipath: `1` to `5`
  - runtime: about `1.02` to `1.08 s/epoch`

Output:

- `experiments/results/los_nlos_verification/los_nlos_real_ephemeris.gif`

## E2E positioning results

Script: `experiments/exp_e2e_trajectory.py`

| Scenario | RMS [m] | P50 [m] | P95 [m] | Avg NLOS | Avg Acq |
| --- | ---: | ---: | ---: | ---: | ---: |
| Open Sky | 22.5 | 20.9 | 37.1 | 0.0 | 8 |
| Odaiba | 38.7 | 32.6 | 58.2 | 0.9 | 6 |
| Shinjuku | 91.5 | 56.0 | 148.9 | 2.0 | 6 |

Output:

- `experiments/results/e2e_positioning/e2e_trajectory_cdf.png`
- `experiments/results/e2e_positioning/e2e_positioning.png`

Interpretation:

- The trajectory experiment now reconstructs pseudoranges from acquisition code phase instead of injecting geometric truth directly.
- Open-sky error is no longer `0 m`; it now reflects acquisition quantization at `2.6 MHz` plus ambiguity-resolution noise.
- Sub-sample interpolation reduced the open-sky floor substantially relative to the integer-sample acquisition output.
- Urban error still grows with NLOS count, and Shinjuku remains harder than Odaiba.
- A `2 km` code-lock gate is used before WLS so gross false acquisitions do not dominate the summary.

Single-epoch check from `experiments/exp_e2e_positioning.py`:

- Open Sky (8/8 acquired): `9.2 m`
- Urban Odaiba (5/8 acquired, 1 NLOS visible): `32.9 m`

## Direct checks for the recent fixes

Using `experiments/data/urbannav/Odaiba/base.nav`:

- At GPS TOW `273375.10`, the satellite clock correction magnitude `c * sat_clk` had:
  - median absolute value: `42,065.7 m`
  - max absolute value: `227,544.5 m`
- For `G01`, `compute_batch([267400.0, 273800.0])` matched `compute(273800.0)` exactly:
  - position delta: `0.0 m`
  - clock delta: `0.0 s`

That confirms both changes are active on real navigation data rather than only in unit tests.
