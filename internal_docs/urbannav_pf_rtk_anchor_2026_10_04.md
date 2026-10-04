# UrbanNav PF: anchoring the cloud to RTK fixed solutions (2026-10-04)

Goal: make the PF smoother hold RTK precision while libgnss++ RTK is FIXED,
so it enters each RTK gap from a centimetre-level state, and measure accuracy
inside the gaps.

## Why a position update was not enough

With RTK FIXED positions fed as a tight (σ 0.1 m) position update, the PF stayed
about 0.4 m from the fix (Odaiba, median at FIXED epochs). Instrumenting the
update showed why: re-centring put the weighted mean exactly on the fix
(0.000 m), but the σ 0.1 m update then moved it 0.41 m away. The sharp DD
carrier likelihood collapses the cloud into a few clumps; a weight-only update
can only select the clump nearest the fix.

## Method

`--rtk-anchor-pos PATH` (gnssplusplus or RTKLIB `.pos`, read with
`libgnsspp.load_solution`) and `--rtk-anchor-sigma-m` (default 0.1). At every
FIXED epoch of that file, the forward pass calls
`ParticleFilterDevice.reset_position(fix, sigma)`: resample, then redraw every
position as `fix + N(0, σ² I)` with uniform weights, keeping clock bias and
velocity states (new CUDA kernel `pf_device_reset_position`). The epoch store
replays the same redraw in the backward pass, so the smoothed output is pinned
at both ends of each gap. Off by default.

## Setup

Preset `odaiba_stop_detect`, calibrated base, libgnsspp develop `304798e7`
(DD terms running), anchor = libgnss++ `low-cost` RTK on the calibrated base
(5,812 FIXED epochs on Odaiba, 11,506 on Shinjuku). Seeds 42 and 101–104
(mean ± sd). Rates count every reference epoch; epochs without output fail.
"Gap fill" uses the RTK solution where it outputs and the PF elsewhere.

`σ_pos 0.6` lowers the preset's per-epoch position random walk from 1.2 m. It
was chosen on these same two routes, so treat that arm as exploratory.

## Results

Gap fill (RTK + PF), share of all reference epochs:

| Odaiba | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---:|---:|---:|---:|---:|
| no anchor | 62.4% | 68.5% | 79.7 ± 1.1% | 87.8 ± 0.6% | 3.55 |
| anchor | 62.3% | 68.5% | 80.6 ± 0.5% | 88.4 ± 0.7% | 3.31 |
| anchor, σ_pos 0.6 | 62.6% | 69.5% | 81.8 ± 1.3% | 88.7 ± 0.8% | 2.77 |
| no anchor, σ_pos 0.6 (seed 42) | 62.4% | 68.9% | 81.0% | 88.7% | 2.92 |
| RTKLIB demo5 | 50.5% | 56.8% | 69.0% | 86.6% | 40.89 |

| Shinjuku | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---:|---:|---:|---:|---:|
| no anchor | 59.7% | 68.8% | 81.3 ± 0.1% | 84.2 ± 0.1% | 5.89 |
| anchor | 59.9% | 69.3% | 81.9 ± 0.1% | 84.4 ± 0.1% | 5.42 |
| anchor, σ_pos 0.6 | 60.1% | 69.8% | 81.9 ± 0.1% | 84.6 ± 0.1% | 5.04 |
| no anchor, σ_pos 0.6 (seed 42) | 59.8% | 68.9% | 81.1% | 84.0% | 5.34 |
| RTKLIB demo5 | 41.0% | 44.4% | 63.4% | 73.5% | 11.75 |

Smoothed PF error inside RTK gaps (epochs without an RTK FIX), mean over the 5
seeds:

| | Odaiba P50 / <1 m | first 5 s after FIX | Shinjuku P50 / <1 m | first 5 s after FIX |
|---|---:|---:|---:|---:|
| no anchor | 2.50 m / 23% | 2.21 m / 22% | 2.91 m / 14% | 2.12 m / 18% |
| anchor | 1.80 m / 25% | 1.74 m / 26% | 2.52 m / 18% | 1.78 m / 25% |
| anchor, σ_pos 0.6 | 1.63 m / 34% | 1.32 m / 40% | 2.68 m / 20% | 1.44 m / 33% |

## Findings

- The anchor works as intended at FIXED epochs: the PF sits on the fix
  (forward error 0.12 m median vs 0.45 m with a position update), and the PF
  alone is within 0.5 m on 51–53% of epochs instead of 7–9%.
- Inside the gaps it helps, but modestly: gap P50 −0.4 to −0.7 m, gap-fill RMS
  −0.2 to −0.5 m, gap-fill <5 m +0.3 to +0.6 points. The Shinjuku gains are
  small but well outside its seed spread (sd ≤ 0.1 points).
- The limit is drift after FIX is lost, not the start state. Odaiba forward
  error after the last FIXED epoch (median over 66 transitions):

  | epochs after FIX loss (0.1 s) | 0 | 1 | 2 | 5 | 10 |
  |---|---:|---:|---:|---:|---:|
  | σ_pos 1.2 | 0.60 | 1.04 | 1.13 | 1.64 | 2.21 |
  | σ_pos 0.6 | 0.25 | 0.45 | 0.57 | 1.09 | 1.58 |
  | σ_pos 0.3 (seed 42 only) | 0.14 | 0.24 | 0.33 | 0.73 | 1.42 |

  Even with σ_pos 0.3 the error grows by about 0.1 m per epoch (~1 m/s), which
  points at the IMU-guided velocity, not the random walk.
- Lowering σ_pos helps on its own (Odaiba seed 42 without anchor reaches 88.7%
  <5 m), so the preset's 1.2 m random walk is worth revisiting, on data other
  than these two routes.

## Heading from RTK fixes (added the same day)

### Diagnosis

The velocity fed to the forward `predict` (wheel speed × IMU heading) was
compared with the reference trajectory's finite-difference velocity (Odaiba,
seed 42, moving epochs):

| heading correction | \|Δv\| P50 / P90 | cross-track sd | \|Δv\| P90 inside RTK gaps |
|---|---:|---:|---:|
| 10 Hz SPP finite differences (preset) | 0.24 / 1.67 m/s | 0.97 m/s | 1.69 m/s |
| RTK FIXED finite differences, gyro only elsewhere | 0.08 / 0.21 m/s | 0.10 m/s | 0.16 m/s |

The preset corrects the heading every 0.1 s from SPP position differences. At
10 m/s the vehicle moves 1 m per epoch while SPP errors are metres, so that
heading is mostly noise. Correcting it only between consecutive FIXED epochs
and integrating the gyro elsewhere cuts the velocity error by about 8×.

Better velocity alone did not help (gap-fill <5 m 88.0% vs 89.4% on Odaiba,
seed 42): the error still jumped 0.12 → 0.57 m in the first epoch after FIX
loss. The per-epoch random walk σ_pos 1.2 m lets biased measurements move the
cloud. Lowering σ_pos alone did not help either (Odaiba 85.4%, worse): dead
reckoning is only worth trusting when the heading is right. The two together
work.

### Method

`--rtk-anchor-heading` passes the FIXED-epoch lookup of `--rtk-anchor-pos` to
the IMU heading filter instead of the SPP lookup. Run with `--sigma-pos 0.1`.

### Results (5 seeds; σ_pos 0.1 was chosen on these two routes)

| Odaiba | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---:|---:|---:|---:|---:|
| PF, anchor | 50.7% | 60.2% | 80.8% | 90.0% | 3.23 |
| PF, anchor + RTK heading + σ_pos 0.1 | **61.9 ± 1.5%** | **69.1 ± 2.5%** | **93.0 ± 1.9%** | **98.1 ± 0.0%** | **1.26** |
| RTK + PF gap fill, same PF | 65.8% | 74.6% | 89.6% | 94.2% | 1.96 |
| libgnss++ `low-cost` RTK alone | 61.8% | 66.9% | 72.9% | 74.5% | 1.97 |
| RTKLIB demo5 | 50.5% | 56.8% | 69.0% | 86.6% | 40.89 |

| Shinjuku | <0.5 m | <1 m | <3 m | <5 m | RMS |
|---|---:|---:|---:|---:|---:|
| PF, anchor | 53.4% | 61.3% | 79.6% | 83.8% | 5.12 |
| PF, anchor + RTK heading + σ_pos 0.1 | **63.5 ± 0.9%** | **72.3 ± 1.2%** | 87.8 ± 0.4% | **91.8 ± 0.5%** | **2.42** |
| RTK + PF gap fill, same PF | 63.8% | 75.6% | 88.5% | 91.7% | 3.42 |
| libgnss++ `low-cost` RTK alone | 59.4% | 67.6% | 76.0% | 76.4% | 3.06 |
| RTKLIB demo5 | 41.0% | 44.4% | 63.4% | 73.5% | 11.75 |

PF error inside RTK gaps (median, mean over seeds): Odaiba 1.80 → 1.31 m and
1.74 → 0.37 m in the first 5 s; Shinjuku 2.52 → 1.40 m and 1.78 → 0.71 m.
The forward error 1 s after FIX loss is 0.39 m instead of 2.21 m (Odaiba).

- The anchored PF output can be used directly: at FIXED epochs it equals the
  fix. On Odaiba it beats the gap fill at <3 m and <5 m, because the gap fill
  keeps RTK FLOAT epochs with large errors. It is within 5 m on 98% of all
  reference epochs and within 3 m on 93%, against 87% and 69% for RTKLIB
  demo5. On Shinjuku the PF and the gap fill are within 3.3 points of each
  other: the gap fill leads at <1 m, the PF has the lower RMS.
- Caveat: σ_pos 0.1 was selected on Odaiba and Shinjuku themselves. The RTK
  heading needs no tuning, but the σ_pos choice should be confirmed on other
  data before the preset changes.

## Next

1. Confirm σ_pos (and the heading option) on data other than these two routes.
2. Make the anchored configuration a named preset once confirmed.
3. Long gaps (Shinjuku 15–60 s) remain metre-level; TDCP displacement or
   Doppler velocity could bound the along-track drift there.
