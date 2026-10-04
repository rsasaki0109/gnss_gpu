# PPC as held-out data for the RTK-anchored PF (2026-10-04)

The RTK anchor, the RTK-fix heading and σ_pos 0.1 were developed on UrbanNav
Odaiba and Shinjuku ([record](urbannav_pf_rtk_anchor_2026_10_04.md)). This
record runs the same PF on the six PPC-Dataset routes (Tokyo and Nagoya,
2024, different receiver and IMU) without retuning.

## What PPC needed

- **Layout.** `experiments/materialize_ppc_urbannav_layout.py` copies each run
  to the UrbanNav file names and rewrites `imu.csv`. Against the reference
  heading rate, the UrbanNav gyro z gives +0.965 per rad/s (z-down); PPC gives
  −0.0169 per unit, i.e. deg/s with z-up. The script rotates 180° about x and
  converts to rad/s (after: +0.968).
- **Speed without an odometer.** PPC has no wheel speed, which the IMU guide
  multiplies by the heading. `--imu-speed-source doppler` uses the horizontal
  Doppler least-squares speed from GPS/Galileo/QZSS rows (L1/E1, 1575.42 MHz),
  held through epochs without a solution and zeroed below 0.15 m/s. Against
  the reference: |error| P50 0.023 / P90 0.15 m/s on Odaiba, 0.09 / 0.76 m/s on
  PPC Tokyo run1.
- **Doppler and satellite velocity in libgnsspp.** Rows carried neither, so
  gnssplusplus #558 adds `snr`, `carrier_phase`, `doppler`,
  `satellite_velocity` and `satellite_clock_drift` to `CorrectedMeasurement`.
  The gnss_gpu pin moves to `9d89a58a`. This also feeds gnss_gpu's TDCP,
  Doppler update and carrier-anchor paths, which read the same attributes and
  had been silently empty; on UrbanNav that changes `odaiba_stop_detect`
  slightly (seed 42 smoothed P50/RMS: Odaiba 1.63/4.10 → 1.47/3.93 m,
  Shinjuku 1.66/5.93 → 1.72/5.96 m) and the anchored preset by less
  (Odaiba 1.32 → 1.23 m RMS).
- **RTK anchor.** libgnss++ `gnss_solve --preset low-cost` (develop), base
  coordinate from the RINEX header (no calibration on held-out data).

## Arms (seed 42, all with Doppler speed)

- B: `odaiba_stop_detect` (no anchor)
- C: B + `--rtk-anchor-pos` (anchor only)
- A: `urbannav_rtk_anchored` (anchor + RTK heading + σ_pos 0.1)
- D: A + `--imu-gyro-bias-zupt` (preset `rtk_anchored_doppler`)

## Results (share of all reference epochs; PF smoothed output)

| route | RTK <1 / <5 m | B <1 / <5 m | C <1 / <5 m | A <1 / <5 m | D <1 / <5 m | D RMS |
|---|---:|---:|---:|---:|---:|---:|
| tokyo run1 | 76.2 / 83.6 | 25.0 / 80.7 | 72.4 / 91.2 | 79.2 / **93.9** | 79.2 / 90.8 | 2.68 |
| tokyo run2 | 85.2 / 90.9 | 47.1 / 94.8 | 83.5 / 96.5 | 87.2 / **98.4** | 86.4 / 98.2 | 1.12 |
| tokyo run3 | 82.0 / 86.4 | 22.6 / 81.7 | 77.2 / 87.9 | 86.9 / **95.9** | 88.6 / 95.7 | 1.71 |
| nagoya run1 | 81.9 / 87.0 | 11.3 / 86.0 | 68.2 / **87.5** | 70.6 / 83.2 | 76.4 / 85.8 | 6.86 |
| nagoya run2 | 57.5 / 67.6 | 36.1 / 74.0 | 56.2 / 75.1 | 55.1 / 67.6 | 56.8 / **79.2** | 4.86 |
| nagoya run3 | 50.0 / 55.9 | 20.8 / 63.9 | 47.3 / 68.8 | 44.9 / 72.2 | 47.2 / **80.4** | 4.76 |
| **mean** | 72.1 / 78.6 | 27.2 / 80.2 | 67.5 / 84.5 | 70.7 / 85.2 | **72.4 / 88.4** | |

RTK = libgnss++ `low-cost` alone (63–92% coverage). A's RMS on Nagoya was
10.0 / 26.9 / 6.8 m.

## Findings

- **Tokyo transfers.** On all three Tokyo routes A beats the anchor alone by
  2–8 points at <5 m and lowers the RMS (run2 2.37 → 1.14 m, run3 4.09 →
  1.69 m; run1 unchanged at 2.7 m). The RTK heading and σ_pos 0.1 were not
  tuned on these data.
- **Nagoya exposed a gyro bias.** In gaps longer than 15 s, A's error grew to
  10–20 m. The IMU-guide heading error on nagoya run2 grew from 4° (median,
  first 5 s after FIX) to 22° at 30–60 s and 37° beyond. The stationary
  yaw-rate mean is +0.15–0.16 deg/s on all Nagoya routes, against
  +0.02 deg/s on Tokyo and −0.01 deg/s on UrbanNav. σ_pos 0.1 trusts dead
  reckoning enough that this bias, uncorrected, dominates.
- **`--imu-gyro-bias-zupt`** holds the heading at standstill and learns the
  yaw-rate bias there (exponential gain 0.05 per epoch). D recovers Nagoya
  (run2 67.6 → 79.2% <5 m, RMS 26.9 → 4.9 m) and keeps Tokyo within 0.2
  points except run1 (−3.1). Over the six routes D is the best arm on average
  (<5 m 88.4% vs 84.5% for the anchor alone and 78.6% for RTK alone).
- **ZUPT is not free on low-bias IMUs.** On UrbanNav (wheel speed, bias
  ≤ 0.01 deg/s) it raises <1 m (Odaiba 70.2 → 76.6%) but lowers <5 m
  (98.1 → 94.5%), with the loss in gaps longer than 60 s. The cause is not yet
  understood (stop detection there contains no turning). So it is a separate
  preset, `rtk_anchored_doppler`, not part of `urbannav_rtk_anchored`.
- Nagoya FIX positions carry a ~0.1 m median offset (Tokyo 0.01–0.02 m),
  consistent with a header-coordinate offset of the Nagoya base.

## Next

1. Explain the ZUPT loss on UrbanNav long gaps, then decide whether one preset
   can serve both IMUs.
2. Multi-seed PPC runs (single seed here).
3. UrbanNav Hong Kong (`experiments/fetch_urbannav_hk_subset.py`) as a third
   held-out set.
