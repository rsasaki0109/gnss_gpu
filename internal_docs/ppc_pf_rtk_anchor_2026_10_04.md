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
- D: A + `--imu-gyro-bias-zupt` (the first `rtk_anchored_doppler`; the final
  presets also hold standstill, see the last section)

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
- **The ZUPT loss on UrbanNav was standstill drift, not heading.** With the
  ZUPT, Odaiba <1 m rose (70.2 → 76.6%) but <5 m fell (98.1 → 94.5%, seed
  42). The heading actually improved (median error beyond 60 s of RTK gap
  1.00° → 0.48°). The loss sat in one 48 s stretch where the vehicle stood
  still for ~100 s without a FIX: both runs drifted to 6–9 m (forward) while
  stationary, D's smoothed error crossed 5 m and A's did not. The preset's
  stop random walk (`--imu-stop-sigma-pos 0.1`, i.e. ~2 m per 50 s) lets
  biased pseudoranges move a parked car. Over five seeds the ZUPT-only Odaiba
  RMS was 1.09–1.60 m, so seed 42 was also on the bad side.
- Nagoya FIX positions carry a ~0.1 m median offset (Tokyo 0.01–0.02 m),
  consistent with a header-coordinate offset of the Nagoya base.

## One preset for both IMUs (stop σ 0.01)

Holding position at standstill (`--imu-stop-sigma-pos 0.01`) together with
the ZUPT fixes both datasets. `urbannav_rtk_anchored` now includes both, and
`rtk_anchored_doppler` is that preset plus Doppler speed.

Share of all reference epochs <1 / <3 / <5 m and RMS (PF smoothed; UrbanNav
uses wheel speed and the calibrated base, PPC Doppler speed and the header
base):

| route | A (old preset) | A + ZUPT | A + ZUPT + stop σ 0.01 (new preset) |
|---|---:|---:|---:|
| Odaiba (5 seeds for A and new) | 69.1 / 93.0 / 98.1, 1.26 m | 76.6 / 89.3 / 94.5, 1.60 m | **80.7 / 98.0 / 98.1, 0.83 m** |
| Shinjuku (5 seeds for A and new) | 72.3 / 87.8 / 91.8, 2.42 m | 71.6 / 87.7 / 91.9, 2.58 m | 72.2 / 88.0 / 92.1, 2.57 m |
| tokyo run1 | 79.2 / 87.6 / 93.9, 2.65 m | 79.2 / 87.7 / 90.8, 2.68 m | 80.3 / 87.6 / 93.8, 2.60 m |
| tokyo run2 | 87.2 / 94.2 / 98.4, 1.14 m | 86.4 / 96.0 / 98.2, 1.12 m | 86.1 / 93.9 / 98.2, 1.20 m |
| tokyo run3 | 86.9 / 94.8 / 95.9, 1.69 m | 88.6 / 94.8 / 95.7, 1.71 m | 89.5 / 94.9 / 95.7, 1.73 m |
| nagoya run1 | 70.6 / 74.2 / 83.2, 10.00 m | 76.4 / 79.5 / 85.8, 6.86 m | 74.8 / 79.8 / 86.5, 6.85 m |
| nagoya run2 | 55.1 / 64.5 / 67.6, 26.89 m | 56.8 / 74.3 / 79.2, 4.86 m | 59.2 / 74.9 / 79.1, 5.19 m |
| nagoya run3 | 44.9 / 63.0 / 72.2, 6.80 m | 47.2 / 71.5 / 80.4, 4.76 m | 48.1 / 71.1 / 79.9, 4.67 m |
| **PPC mean** | 70.7 / 79.7 / 85.2, 8.20 m | 72.5 / 84.0 / 88.4, 3.67 m | **73.0 / 83.7 / 88.9, 3.71 m** |

PPC rows are seed 42. The new preset matches or beats the old one on every
route at <5 m except the three Tokyo runs (−0.1 to −0.2 points), and its PPC mean is
3.7 points higher with less than half the RMS. Odaiba gains 11.6 points at
<1 m and 5.0 at <3 m. Shinjuku is unchanged within 0.3 points, RMS +0.15 m.

## Five seeds (2026-10-05)

Seeds 42 and 101–104 for the final `rtk_anchored_doppler` and for the anchor
alone (`odaiba_stop_detect` + Doppler speed + `--rtk-anchor-pos`). Share of
all reference epochs, mean ± sd over seeds:

| route | arm | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---|---:|---:|---:|---:|---:|
| tokyo run1 | anchor | 67.5 ± 0.1 | 72.8 ± 0.3 | 85.6 ± 0.1 | 90.8 ± 0.4 | 2.7 |
| | **preset** | 76.0 ± 0.7 | 79.9 ± 0.3 | 87.6 ± 0.1 | 93.7 ± 0.4 | 2.6 |
| tokyo run2 | anchor | 79.6 ± 0.1 | 83.6 ± 0.4 | 92.9 ± 0.8 | 96.4 ± 0.2 | 2.4 |
| | **preset** | 83.1 ± 0.2 | 86.3 ± 0.2 | 94.5 ± 1.0 | 98.3 ± 0.1 | 1.2 |
| tokyo run3 | anchor | 72.6 ± 0.3 | 77.3 ± 0.2 | 85.1 ± 0.2 | 87.9 ± 0.2 | 4.1 |
| | **preset** | 83.7 ± 0.7 | 89.5 ± 0.2 | 94.8 ± 0.0 | 95.7 ± 0.0 | 1.8 |
| nagoya run1 | anchor | 64.2 ± 0.6 | 68.8 ± 0.9 | 82.4 ± 0.2 | 87.5 ± 0.1 | 3.9 |
| | **preset** | 67.4 ± 0.3 | 76.4 ± 0.9 | 80.5 ± 1.6 | 86.3 ± 0.3 | 6.8 |
| nagoya run2 | anchor | 52.3 ± 0.2 | 56.4 ± 0.2 | 67.2 ± 0.6 | 75.7 ± 0.4 | 6.5 |
| | **preset** | 53.8 ± 0.3 | 56.7 ± 1.5 | 74.7 ± 0.9 | 79.3 ± 0.2 | 5.0 |
| nagoya run3 | anchor | 42.3 ± 1.3 | 45.7 ± 1.7 | 58.2 ± 0.4 | 69.6 ± 0.8 | 5.5 |
| | **preset** | 43.6 ± 0.4 | 48.2 ± 1.4 | 71.4 ± 0.6 | 80.0 ± 0.5 | 4.7 |
| **mean** | libgnss++ RTK alone | 68.1 | 72.1 | 77.4 | 78.6 | |
| | anchor | 63.1 ± 0.2 | 67.5 ± 0.2 | 78.6 ± 0.2 | 84.6 ± 0.1 | |
| | **preset** | **67.9 ± 0.2** | **72.8 ± 0.3** | **83.9 ± 0.4** | **88.9 ± 0.1** | |

- Seed spread is small (route-mean sd ≤ 0.4 points), so the preset's gain
  over the anchor alone, 4.3–5.3 points at every threshold, is well outside
  it. The seed-42 table above holds.
- Against libgnss++ RTK alone the preset ties at <0.5 m and adds 6.5 points
  at <3 m and 10.3 at <5 m.
- nagoya run1 is the exception: the preset is 1.2 points worse at <5 m and
  its RMS is 6.8 m vs 3.9 m. It still has a long-gap failure the bias ZUPT did
  not remove.

## Robust Doppler speed (2026-10-05)

On nagoya run1 the remaining failures were 30–66 m forward errors 15–45 s
into RTK gaps. The heading was fine there (P90 < 1.3°); the Doppler speed was
not (e.g. 31.7 m/s against a true 11.2 m/s): a single multipath Doppler moved
the least-squares fit. `doppler_ground_speed` now drops the row with the
largest range-rate residual and refits while that residual exceeds 0.5 m/s
and more than five rows remain. Against the reference speed (|error| P90):
nagoya run1 0.89 → 0.44 m/s, nagoya run2 0.85 → 0.50 m/s, Odaiba 0.149 →
0.147 m/s. An acceleration gate and a post-fit RMS gate were also tried and
did not help (the gate locked onto outliers).

`rtk_anchored_doppler` with the robust speed, seed 42 (<1 / <3 / <5 m, RMS):

| route | before | robust |
|---|---:|---:|
| tokyo run1 | 80.3 / 87.6 / 93.8, 2.60 m | 80.1 / 88.1 / 94.6, 2.44 m |
| tokyo run2 | 86.1 / 93.9 / 98.2, 1.20 m | 88.1 / 94.2 / 98.9, 1.06 m |
| tokyo run3 | 89.5 / 94.9 / 95.7, 1.73 m | 91.3 / 95.2 / 98.1, 1.51 m |
| nagoya run1 | 74.8 / 79.8 / 86.5, 6.85 m | 77.7 / 81.3 / 86.2, 6.35 m |
| nagoya run2 | 59.2 / 74.9 / 79.1, 5.19 m | 57.1 / 71.7 / 77.1, 4.85 m |
| nagoya run3 | 48.1 / 71.1 / 79.9, 4.67 m | 50.2 / 77.1 / 87.9, 4.11 m |
| **mean** (<0.5 / <1 / <3 / <5 m) | 67.8 / 73.0 / 83.7 / 88.9%, 3.71 m | **70.0 / 74.1 / 84.6 / 90.4%, 3.39 m** |

The mean gain (+1.5 points at <5 m) is well above the five-seed spread
measured before (≤ 0.4 points). nagoya run2 loses 2 points; nagoya run1 is
still below the anchor alone at <5 m (86.2 vs 87.5%).

## GNSS outages (2026-10-05)

nagoya run1's largest remaining failure was not a bad speed or heading: the
rover output no epochs for 15 s (550594–550608), and the PF resumed 73 m off.
Two things went wrong across the outage:

1. One predict spanned the whole 15 s with the speed times the *final*
   heading, a straight line through a curve.
2. `sigma_pos` is applied once per predict regardless of `dt`, so the cloud
   was still 0.1 m wide after 15 s of dead reckoning and could not
   re-acquire (69 m off 10 s later).

Epoch gaps longer than 1 s occur on seven of the eight routes (5 s or more:
Shinjuku 6, tokyo run1 1, nagoya run1 1, nagoya run2 1).

- **Path-averaged direction (always on).** For intervals longer than 0.5 s the
  IMU guide uses the time-weighted mean of the gyro-integrated heading
  direction (the chord of the path) instead of the final heading. Ordinary
  epochs are unchanged.
- **`--predict-gap-velocity-sigma` / `--predict-gap-min-s` (off by
  default).** After a gap longer than the threshold, the predict spread becomes
  √(σ_pos² + (σ_v·dt)²), and the backward pass replays it.

Seed 42, current presets (<1 / <3 / <5 m, RMS):

| route | before | path-averaged | + gap σ_v 1.0 m/s (gaps > 1 s) | + gap σ_v 1.0 m/s (gaps > 5 s) |
|---|---:|---:|---:|---:|
| Odaiba | 83.8 / 98.0 / 98.1, 0.79 m | 84.4 / 98.0 / 98.1, 0.75 m | 73.8 / 97.1 / 97.8, 1.01 m | (no gap > 5 s) |
| Shinjuku | 73.3 / 87.7 / 91.8, 2.58 m | 71.9 / 89.1 / 93.0, 1.69 m | 71.2 / 89.6 / 91.1, 2.38 m | 71.4 / 89.0 / 90.9, 2.64 m |
| tokyo run1 | 80.1 / 88.1 / 94.6, 2.44 m | 80.1 / 88.0 / 94.5, 2.39 m | 80.2 / 88.1 / 94.8, 2.23 m | 80.3 / 88.2 / 94.9, 2.21 m |
| nagoya run1 | 77.7 / 81.3 / 86.2, 6.35 m | 77.7 / 81.4 / 86.2, 5.23 m | 77.7 / 82.0 / 89.1, 3.04 m | 77.7 / 82.3 / 88.5, 3.52 m |
| nagoya run2 | 57.1 / 71.7 / 77.1, 4.85 m | 55.1 / 72.6 / 77.4, 4.65 m | 55.3 / 73.1 / 77.1, 4.69 m | 55.1 / 72.6 / 77.4, 4.65 m |
| **UrbanNav mean** | 78.6 / 92.8 / 95.0, 1.69 m | 78.2 / 93.5 / 95.5, 1.22 m | 72.5 / 93.4 / 94.4, 1.69 m | 77.9 / 93.5 / 94.5, 1.70 m |
| **PPC mean** | 74.1 / 84.6 / 90.4, 3.39 m | 73.9 / 84.8 / 90.5, 3.16 m | 73.9 / 85.3 / 90.9, 2.78 m | 73.9 / 84.9 / 91.0, 2.84 m |

tokyo run2/run3 and nagoya run3 are unchanged or within 0.2 points.

- The path-averaged direction helps or is neutral everywhere at <5 m and
  lowers RMS on both datasets (Shinjuku 2.58 → 1.69 m), so it is the default.
- Widening the spread after outages fixes nagoya run1 (86.2 → 89.1% <5 m,
  RMS 6.35 → 3.04 m, now above the anchor alone) but costs Shinjuku, whose
  outages are in deep canyons where a wider cloud follows biased ranges
  (−1.9 to −2.1 points at <5 m), and Odaiba precision when short
  gaps are included (<1 m −10.6 points). It stays opt-in.
- These are seed-42 changes; the five-seed numbers in the README predate them
  (Odaiba 0.79 → 0.75 m RMS at seed 42).

## Five seeds on the final code (2026-10-05)

gnss_gpu `330cce5` (robust Doppler speed, path-averaged outage direction),
seeds 42 and 101–104, mean ± sd (<0.5 / <1 / <3 / <5 m, P50, RMS):

| route | preset | <0.5 m | <1 m | <3 m | <5 m | P50 | RMS |
|---|---|---:|---:|---:|---:|---:|---:|
| Odaiba | `urbannav_rtk_anchored` | 70.4 ± 2.2 | 80.7 ± 4.1 | 97.9 ± 0.2 | 98.1 ± 0.0 | 0.10 | 0.80 |
| Shinjuku | `urbannav_rtk_anchored` | 62.9 ± 1.4 | 72.4 ± 0.7 | 88.8 ± 0.3 | 92.9 ± 0.3 | 0.27 | 1.70 |
| tokyo run1 | `rtk_anchored_doppler` | 75.0 ± 0.2 | 80.2 ± 0.2 | 88.1 ± 0.1 | 94.6 ± 0.1 | 0.04 | 2.36 |
| tokyo run2 | `rtk_anchored_doppler` | 84.9 ± 0.1 | 88.5 ± 0.3 | 94.6 ± 0.7 | 99.0 ± 0.1 | 0.02 | 1.05 |
| tokyo run3 | `rtk_anchored_doppler` | 85.3 ± 1.0 | 91.4 ± 0.1 | 95.1 ± 0.1 | 97.5 ± 0.7 | 0.02 | 1.54 |
| nagoya run1 | `rtk_anchored_doppler` | 71.0 ± 1.9 | 77.6 ± 0.2 | 82.2 ± 1.5 | 86.6 ± 0.6 | 0.13 | 5.27 |
| nagoya run2 | `rtk_anchored_doppler` | 53.3 ± 0.1 | 55.7 ± 0.6 | 71.9 ± 0.8 | 77.1 ± 0.2 | 0.18 | 4.69 |
| nagoya run3 | `rtk_anchored_doppler` | 46.7 ± 0.7 | 51.3 ± 0.8 | 76.8 ± 0.5 | 87.9 ± 0.5 | 0.84 | 4.20 |
| **PPC mean** | | **69.4 ± 0.3** | **74.1 ± 0.2** | **84.8 ± 0.3** | **90.5 ± 0.1** | 0.20 | 3.19 |

These are the numbers in `experiments/results/urbannav_current_checkpoint.json`,
the README, the site snapshot and the zero-data demo. Against the earlier
five-seed tables: Odaiba RMS 0.83 → 0.80 m, Shinjuku RMS 2.57 → 1.70 m and
<5 m 92.1 → 92.9%, PPC mean <5 m 88.9 → 90.5%.

## Smoother passes weighted by distance to their anchor (2026-10-05)

The smoother combined the forward and backward passes as a plain average.
Each pass starts from an RTK anchor and is most accurate next to it: after
nagoya run1's outage the forward pass was 73 m off and the backward pass,
1 s from the next FIX, nearly right; the average was still 36 m off.
`--smoother-anchor-weighting` (now in both anchored presets) weights the
forward pass by 1/(t_since + 1 s) and the backward pass by 1/(t_until + 1 s),
with t the time since the previous / until the next anchored epoch, clipped
to [0.1, 0.9]. The offset and the clip were chosen offline on all eight
routes (unclipped weights raised the PPC RMS by 0.4 m), so there is no
held-out route left for this choice.

Five seeds (mean ± sd), previous preset → new preset:

| route | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---:|---:|---:|---:|---:|
| Odaiba | 70.4 → 74.2 | 80.7 → 84.4 | 97.9 → 97.4 | 98.1 → 98.1 | 0.80 → 0.77 |
| Shinjuku | 62.9 → 65.1 | 72.4 → 75.1 | 88.8 → 90.4 | 92.9 → 93.9 | 1.70 → 1.55 |
| tokyo run1 | 75.0 → 76.9 | 80.2 → 81.4 | 88.1 → 91.3 | 94.6 → 93.1 | 2.36 → 2.51 |
| tokyo run2 | 84.9 → 85.3 | 88.5 → 88.7 | 94.6 → 97.4 | 99.0 → 98.9 | 1.05 → 0.91 |
| tokyo run3 | 85.3 → 86.6 | 91.4 → 92.7 | 95.1 → 96.9 | 97.5 → 98.0 | 1.54 → 1.51 |
| nagoya run1 | 71.0 → 71.6 | 77.6 → 79.3 | 82.2 → 84.3 | 86.6 → 86.9 | 5.27 → 4.48 |
| nagoya run2 | 53.3 → 54.4 | 55.7 → 58.5 | 71.9 → 72.5 | 77.1 → 79.0 | 4.69 → 5.18 |
| nagoya run3 | 46.7 → 48.4 | 51.3 → 56.8 | 76.8 → 77.4 | 87.9 → 86.9 | 4.20 → 6.13 |
| **PPC mean** | 69.4 → **70.5** | 74.1 → **76.2** | 84.8 → **86.6** | 90.5 → 90.5 | 3.19 → 3.45 |

Seed sd is at most 3.0 points (Odaiba <1 m) and at most 0.4 points on the
PPC means. Every route gains at <0.5 m and <1 m, and all but Odaiba at <3 m.
The cost: when one pass has failed near its own anchor the weighting follows
it, so tokyo run1 (−1.5) and nagoya run3 (−1.0) lose at <5 m and the PPC
RMS rises by 0.26 m (nagoya run3 4.20 → 6.13 m).

## Next

1. An outage model that helps both nagoya run1 and Shinjuku. Tried and
   rejected (2026-10-05): widening only when the pseudorange residuals at the
   dead-reckoned position disagree. At the first epoch after each gap > 1 s
   the clock-removed residual median did not track the true error: Shinjuku
   282938 s had a 30.4 m error at a 3.3 m residual median, 283005 s a 2.9 m
   error at 10.3 m. With 4–7 satellites right after an outage the horizontal
   error is largely absorbed by the clock and geometry. A trigger needs
   another signal (e.g. carrier-phase continuity or the RTK FLOAT position).
   The RTK FLOAT position was then tried and also rejected (2026-10-05). When
   the forward PF and a FLOAT solution disagree, FLOAT is usually right
   (FLOAT median error 0.5–0.7 m; when they disagree by more than 8 m FLOAT is
   closer in 65% of Shinjuku and 97% of nagoya run1 epochs), but FLOAT is
   sometimes wrong for seconds at a time. Redrawing the cloud around FLOAT
   after a persistent disagreement (seed 42, <5 m):

   | trigger | Odaiba | Shinjuku | nagoya run1 | nagoya run2 | PPC mean |
   |---|---:|---:|---:|---:|---:|
   | none (current presets) | 98.1 | 93.0 | 86.2 | 77.4 | 90.5 |
   | > 8 m for 5 FLOAT epochs | 93.5 | 88.6 | 91.0 | 79.2 | 90.5 |
   | > 15 m for ~3 s | 95.6 | 89.0 | 88.6 | 73.7 | 89.7 |

   Even two redraws on Odaiba cost 2.5 points, so this was not merged.
2. A third held-out set with RTK fixes. UrbanNav Hong Kong 2019-04-28
   (`experiments/fetch_urbannav_hk_subset.py`) does not qualify: 8 minutes of
   single-frequency u-blox data against a 30 s HKSC base; libgnss++ RTK gives
   11 solutions and no FIX, so there is nothing to anchor to.
