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
- `libgnsspp`: gnssplusplus `62bd0b73` (the gnss_gpu submodule pin), built
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

The README headline for this preset is smoothed P50 1.36 m / RMS 4.11 m,
recorded around 2026-04 on Linux. Current main reproduces P50 ≈2.2 m /
RMS ≈13.5 m with either resampler, so the gap is not the resampler. Both the
gnss_gpu PF smoother and the gnssplusplus pin (SPP feeds the position update)
have changed since. The May-era gnssplusplus pin (`abd4abd1`) does not build on
Windows, so bisecting needs WSL/Linux (plan.md item 15).
