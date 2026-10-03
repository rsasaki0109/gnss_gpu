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
- libgnss++ RTK: `gnss_solve` from gnssplusplus `62bd0b73` (the gnss_gpu
  pin at the time), presets
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

## Calibrated base (2026-10-04)

The base coordinate is calibrated once in
`configs/urbannav/tokyo_base_cref0001.json`. The removed offset is ENU
(−0.719, −0.024, −0.052) m: the pooled median of libgnss++ FIX errors, with
per-route medians agreeing within 0.035 m. The calibrated coordinate is then
applied identically to every method:

- RTKLIB: `ant2-postype=xyz`;
- libgnss++: `gnss_solve --base-ecef`;
- the PF: new `exp_pf_smoother_eval.py --base-ecef X Y Z`, which reaches the
  DD pseudorange, widelane and DD carrier computers.

**Windows caveat:** the libgnss++ `--base-ecef` parser reads its three values
in one unsequenced expression. MSVC evaluates right to left, so the values
must be passed as `Z Y X` on Windows builds; the fix belongs in
gnssplusplus-library. A wrong order shows up as `Warning: --base-ecef differs
from RINEX header by 1.08e7 m` and a 0% fix rate. This applies to `62bd0b73`;
gnssplusplus develop reads the values in order in `gnss_solve`, `gnss_live` and
`gnss_replay` (#556), so pass `X Y Z` there.

| Odaiba (calibrated base) | cover | <0.5 m | <1 m | <3 m | <5 m | P50 | RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTKLIB demo5 | 97.3% | 50.5% | 56.8% | 69.0% | 86.6% | 0.34 | 40.89 |
| libgnss++ `low-cost` | 78.5% | 61.8% | 66.9% | 72.9% | 74.5% | 0.07 | 1.97 |
| PF (SPP update) | 98.2% | 3.5% | 13.5% | 63.3% | 81.5% | 2.25 | 13.57 |
| PF (calibrated-RTK update, σ 1.9 m) | 98.2% | 4.9% | 19.6% | 70.1% | 80.7% | 1.87 | 11.60 |
| libgnss++ + PF (SPP update) gap fill | 98.2% | 62.0% | 67.6% | 77.5% | 84.6% | 0.20 | 13.24 |
| libgnss++ + PF (RTK update) gap fill | 98.2% | 62.1% | 67.9% | 78.1% | 83.7% | 0.20 | 11.25 |

| Shinjuku (calibrated base) | cover | <0.5 m | <1 m | <3 m | <5 m | P50 | RMS |
|---|---:|---:|---:|---:|---:|---:|---:|
| RTKLIB demo5 | 95.3% | 41.0% | 44.4% | 63.4% | 73.5% | 1.32 | 11.75 |
| libgnss++ `low-cost` | 77.7% | 59.4% | 67.6% | 76.0% | 76.4% | 0.17 | 3.06 |
| PF (SPP update) | 95.7% | 0.6% | 2.2% | 17.5% | 38.3% | 5.97 | 13.02 |
| PF (calibrated-RTK update, σ 0.5 m) | 95.7% | 1.6% | 6.4% | 40.8% | 67.3% | 3.43 | 11.04 |
| libgnss++ + PF (SPP update) gap fill | 95.8% | 59.5% | 67.8% | 77.7% | 80.3% | 0.28 | 10.18 |
| libgnss++ + PF (RTK update) gap fill | 95.8% | 59.5% | 68.0% | 78.9% | 82.0% | 0.28 | 10.64 |

- Calibration brings libgnss++ FIX to centimetre level (median 0.72 → 0.07 m
  on Odaiba) and RTKLIB to 0.34 m. The PF output is unchanged because its DD
  terms never ran: libgnsspp rows had no `prn`, so no satellite could be paired
  with the base. See
  [urbannav_pf_dd_satellite_ids_2026_10_04.md](urbannav_pf_dd_satellite_ids_2026_10_04.md);
  every PF row in this record predates that fix.
- **libgnss++ RTK + PF gap fill beats RTKLIB demo5 on every threshold** on
  Shinjuku, and on <0.5/1/3 m on Odaiba. Odaiba <5 m is the exception: 83.7–84.6%
  vs RTKLIB's 86.6%, because RTKLIB covers 97% of epochs.
- The PF does not retain RTK precision even with a 0.5 m update sigma: at most
  about 5% of epochs are within 0.5 m. Its only contribution is coverage in
  RTK gaps.
- Single seed (42); the PF arms carry seed noise of about ±0.3 m in P50.

## Next

1. Done (gnssplusplus-library #556): fix the `--base-ecef` argument-order bug.
2. Make the PF hold RTK precision while RTK FIX is available. Anchor it
   tightly to FIX (status-aware update sigma, re-centring), so it enters each
   RTK gap from a centimetre-level state. Measure accuracy inside RTK gaps
   specifically, on both routes with multiple seeds.
