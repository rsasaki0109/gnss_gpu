# UrbanNav Tokyo: current RTK baselines vs the GPU PF smoother (2026-10-03)

The README compared the PF smoother with "RTKLIB demo5 2.67 m / 13.08 m", a
number frozen in April 2026 that no tool in this repo regenerates. This record
re-measures the baselines on today's public data.

## Setup

- Data: public UrbanNav Tokyo subset (`experiments/fetch_urbannav_subset.py`),
  Odaiba (12,410 reference epochs) and Shinjuku (20,950), `rover_trimble`,
  `base_trimble`, `base.nav`.
- RTKLIB demo5: rtklibexplorer/RTKLIB `81943c1` (2026-09-28), `rnx2rtkp -p 2`,
  built in WSL. Options: gnssplusplus `configs/reproduce/rtklib_demo5_ppc.conf`
  (kinematic, L1+L2, forward, elmask 15, navsys 61, continuous AR).
- libgnss++ RTK: `gnss_solve` from the gnss_gpu pin `62bd0b73`, presets
  `low-cost` and `odaiba`.
- PF: preset `odaiba_stop_detect`, current main, seed 42 (the Odaiba P50/RMS
  match the 5-seed mean within noise).
- Metrics: horizontal error vs `reference.csv`. Full-denominator rates count
  every reference epoch, and an epoch with no output is a failure.

## Results

| Odaiba | cover | <1 m | <3 m | <5 m | P50 | RMS |
|---|---:|---:|---:|---:|---:|---:|
| RTKLIB demo5 | 97.3% | 53.2% | 70.0% | 86.8% | 0.79 | 40.87 |
| libgnss++ `low-cost` | 80.8% | 66.4% | 73.9% | 77.0% | 0.72 | 2.15 |
| libgnss++ `odaiba` | 87.5% | 63.5% | 74.5% | 82.8% | 0.71 | 2.31 |
| PF smoother (position update from SPP, default) | 98.2% | 13.5% | 63.3% | 81.5% | 2.25 | 13.56 |
| PF smoother, position update from libgnss++ `low-cost` | 98.2% | 16.1% | 70.3% | 80.9% | 1.95 | 11.32 |
| libgnss++ `low-cost` + PF (default) gap fill | 98.2% | 67.1% | 78.0% | 85.9% | 0.73 | 13.26 |
| libgnss++ `low-cost` + PF (RTK update) gap fill | 98.2% | 67.4% | 78.5% | 85.2% | 0.73 | 10.97 |

| Shinjuku | cover | <1 m | <3 m | <5 m | P50 | RMS |
|---|---:|---:|---:|---:|---:|---:|
| RTKLIB demo5 | 95.3% | 43.7% | 62.5% | 71.1% | 1.20 | 11.81 |
| libgnss++ `low-cost` | 78.3% | 59.6% | 76.0% | 76.9% | 0.77 | 3.20 |
| libgnss++ `odaiba` | 84.5% | 52.4% | 76.0% | 77.2% | 0.87 | 3.67 |
| PF smoother (position update from SPP, default) | 95.7% | 2.2% | 17.5% | 38.3% | 5.97 | 13.02 |
| PF smoother, position update from libgnss++ `low-cost` | 95.7% | 18.4% | 69.7% | 81.1% | 1.90 | 9.99 |
| libgnss++ `low-cost` + PF (default) gap fill | 95.8% | 59.8% | 77.9% | 81.1% | 0.87 | 9.87 |
| libgnss++ `low-cost` + PF (RTK update) gap fill | 95.8% | 60.6% | 81.4% | 84.7% | 0.87 | 9.68 |

## Findings

- **The PF smoother is not a better standalone engine than modern RTK.** Its
  median is about 3× worse on Odaiba and about 5× worse on Shinjuku, where the
  Odaiba-tuned preset generalises poorly.
- **Its value is coverage.** RTK leaves 3–22% of epochs without output, or
  (RTKLIB) with km-scale outliers. Filling RTK gaps with the PF raises the
  within-5 m share by about 8 points on both routes.
- **The PF's own position update is the weak link.** It is fed SPP (P50
  1.9 m, RMS 61 m). Feeding it the libgnss++ RTK solution instead moves
  Shinjuku from 17.5% to 69.7% within 3 m. The 1.9 m update sigma, tuned for
  SPP, still blurs RTK precision (<1 m stays at 16–18%).
- **Base-station coordinate offset.** RTK FIX epochs show a constant offset
  of E −0.70 / −0.74 m and N −0.01 / −0.05 m on both routes, which use the
  same base `CREF0001`. SPP errors show no such shift. So the RINEX-header
  base coordinate is about 0.72 m west of the reference frame. This biases
  every base-relative method (RTK and the PF's DD terms) and caps FIX
  accuracy at about 0.7 m. UrbanNav documents no Tokyo base coordinate.
  Calibrating this constant once from the reference (one offset, applied
  identically to all methods) was approved on 2026-10-03.

## Next

1. Apply the base calibration, then re-measure everything.
2. Make RTK + PF fusion a first-class path: status-aware position-update sigma
   from RTK FIX/FLOAT, plus output-level gap fill. Evaluate it on both routes
   with multiple seeds and lock it in a reproducible benchmark.
