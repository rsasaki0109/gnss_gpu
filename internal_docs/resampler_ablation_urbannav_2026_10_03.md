# UrbanNav Odaiba ablation of ParticleFilterDevice resampling (2026-10-03)

Follow-up to `decisions.md` D-039 and `resampler_ablation_ppc_2026_10_03.md`.
The README-headline PF smoother runs on `ParticleFilterDevice`; this checks
whether the corrected Megopolis changes it.

## Setup

- Data: UrbanNav Tokyo Odaiba, `rover_trimble`. It was fetched with
  `experiments/fetch_urbannav_subset.py --run Odaiba` into
  `E:\datasets\urbannav\Tokyo`.
- Preset: `odaiba_stop_detect` (100k particles, IMU tight coupling, DD
  pseudorange/carrier, forward-backward smoother).
- `libgnsspp`: gnssplusplus `62bd0b73` (the gnss_gpu submodule pin at the
  time), built
  for Python 3.12 on Windows/MSVC.
- Script: `experiments/exp_urbannav_resampler_ablation.py`. It patches
  `pf_smoother_runtime.ParticleFilterDevice`; the backward filter inherits
  the setting.
- Arms:
  - `megopolis_b15`: the pre-2026-10 kernel, now `megopolis_legacy`, B=15.
  - `coalesced_b60`: the new default `megopolis`, B=60.
- Seeds: 6 for legacy (42, 43, 101–104) and 5 for the new default (42,
  101–104).
- Raw runs: [`resampler_ablation_urbannav_2026_10_03.json`](resampler_ablation_urbannav_2026_10_03.json).

## Results (2D, metres; mean ± sd over seeds)

| Arm | FWD P50 | FWD RMS | SMTH P50 | SMTH P95 | SMTH RMS |
|---|---:|---:|---:|---:|---:|
| legacy B=15 (n=6) | 2.71 ± 0.17 | 15.94 ± 0.40 | 2.26 ± 0.07 | 9.92 ± 0.15 | 13.52 ± 0.39 |
| new default B=60 (n=5) | 2.58 ± 0.10 | 15.34 ± 0.43 | 2.20 ± 0.04 | 10.06 ± 0.34 | 13.51 ± 0.25 |

Paired by seed (B=60 minus legacy, seeds 42 and 101–104):
- forward RMS improves in 4 of 5 pairs (−1.54 to +0.21 m);
- smoothed P50 improves in 4 of 5 pairs (−0.11 to +0.06 m);
- smoothed RMS is split (−0.60 to +0.59 m).

## Findings

- The forward filter improves: RMS −0.6 m (about 1.5 sd), P50 −0.13 m.
- The smoothed output, which is the headline metric, is unchanged within seed
  noise: P50 −0.06 m, RMS equal, P95 +0.14 m.
- A single seed cannot resolve these differences. Seed changes alone move the
  forward P50 by 0.3 m and the smoothed RMS by 0.8 m.
- Together with the PPC ablation (P50 −0.7 m), the corrected Megopolis is
  never worse beyond noise and is better on the forward filter. It is also
  correct and faster at 1M particles (B=60 12 ms vs legacy B=15 28 ms), so
  it became the default (D-039).

## Headline reproduction gap

**Update 2026-10-04:** every run in this record had the DD updates disabled
by a missing `prn` on libgnsspp rows (0 of 12,200 DD epochs). With it fixed,
the same preset gives smoothed P50 1.50 m / RMS 4.06 m over 5 seeds, which
matches the April RMS. See
[urbannav_pf_dd_satellite_ids_2026_10_04.md](urbannav_pf_dd_satellite_ids_2026_10_04.md).
The resampler comparison below is still valid as a comparison, but on a PF
without DD terms. The conclusion "gnss_gpu code is not the cause" below is
wrong; the cause was the gnss_gpu/libgnsspp interface.

The README headline for this preset was smoothed P50 1.36 m / RMS 4.11 m, from
`docs/assets/data/odaiba_pf_smoother_freeze.json`: a 2026-04-14 full run on
Linux with `n_epochs=12228`. It cannot be reproduced with the data available
today. The bisection (2026-10-03, all runs `odaiba_stop_detect`, seed 42) was:

| gnss_gpu | SPP fed to the position update | Epochs | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS |
|---|---|---:|---:|---:|---:|---:|
| main (6cb73ce) | gnssplusplus pin `62bd0b73` | 12184 | 2.60 | 15.12 | 2.25 | 13.56 |
| main (6cb73ce) | May pin `abd4abd1` (built in WSL) | 12184 | 1.86 | 13.01 | 1.93 | 13.74 |
| `421d284` (2026-04-23, the commit that recorded the headline) | May pin `abd4abd1` | 12184 | 1.91 | 13.14 | 1.99 | 12.71 |
| 2026-04-14 freeze | April pin `49326766` (not available) | **12228** | 1.19 | 4.57 | 1.36 | 4.11 |

- **gnss_gpu code is not the cause.** The commit that recorded the headline
  gives RMS 12.7 m on today's data, close to main.
- The gnssplusplus SPP version moves P50 by 0.3–0.7 m but not RMS. The May and
  current SPP solutions have similar accuracy on their own (P50 1.64 / 1.85 m,
  RMS 63.7 / 61.4 m) but differ per epoch by a median of 3 m.
- The evaluated epoch count is set by the input data, not the code or SPP:
  every run above gives 12184. The freeze evaluated 12228, so the April run
  used a different Odaiba data version than the public subset that
  `fetch_urbannav_subset.py` downloads (`reference.csv` 12410 rows,
  `rover_trimble.obs` 12399 epochs).
- The April data and the April gnssplusplus pin (`4932676`, absent from the
  local gnssplusplus clone) are not recoverable here. The README now reports
  the current-main result (#191).
