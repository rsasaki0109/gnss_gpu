# PPC ablation of ParticleFilterDevice resampling (2026-10-03)

Follow-up to `decisions.md` D-039. Question: does replacing the historical
`megopolis` resampler change PF accuracy on real data?

## Setup

- Data: `PPC-Dataset-data`, all 6 routes, full length (≈58k epochs).
- Runner: `exp_ppc_ctrbpf_fgo.py`, methods `pf+pu` (PF-PR+PU) and `rbpf+pu`
  (RBPF-velKF+PU), default particle count and settings.
- Script: `experiments/exp_ppc_resampler_ablation.py`. It wraps `_build_pf` so
  only the resampler changes; the runner code is untouched.
- Metrics: horizontal error vs `reference.csv` on matched epochs, averaged per
  route. The honest PPC pass rate is also recorded but stays ≈0% at ≤3 m for
  PF-only methods.
- Noise reference: the default with PF seed 43 instead of 42.
- Summary: [`resampler_ablation_ppc_2026_10_03.json`](resampler_ablation_ppc_2026_10_03.json).
- GPU: GTX 1660 Ti (Turing, compute 7.5), Windows, all native modules built locally.

## Results (6-route mean, horizontal error in m)

| Condition | PF+PU P50 | P90 | RMS | <10 m | RBPF+PU P50 | P90 | RMS | <10 m |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| default `megopolis` B=15 | 30.99 | 59.07 | 38.96 | 5.01% | 31.08 | 59.13 | 39.18 | 4.98% |
| noise ref (PF seed 43) | 31.04 | 59.06 | 39.01 | 5.09% | 31.12 | 58.98 | 39.17 | 4.91% |
| default + fresh seed per resample | 31.00 | 59.07 | 38.96 | 5.09% | 31.07 | 59.17 | 39.16 | 4.90% |
| `megopolis_coalesced` B=15 | 30.99 | 58.72 | 38.91 | 5.02% | 31.08 | 58.98 | 39.19 | 4.93% |
| **`megopolis_coalesced` B=60** | **30.33** | **57.27** | **38.19** | 4.72% | **30.36** | **57.26** | **38.22** | 4.68% |
| `megopolis_coalesced` B=60 + fresh seed | 30.35 | 57.17 | 38.21 | 4.75% | 30.35 | 57.24 | 38.21 | 4.72% |

Per-route P50 change vs default:

| Condition | nagoya1 | nagoya2 | nagoya3 | tokyo1 | tokyo2 | tokyo3 |
|---|---:|---:|---:|---:|---:|---:|
| noise ref, PF+PU | +0.09 | −0.05 | +0.08 | +0.22 | −0.00 | −0.01 |
| coalesced B=60, PF+PU | −0.38 | +0.06 | +0.16 | −0.91 | +0.01 | −2.87 |
| coalesced B=60, RBPF+PU | −0.31 | −0.06 | +0.16 | −1.38 | +0.13 | −2.88 |

## Findings

- `megopolis_coalesced` with B=60 is the only condition that moves outside
  the noise band. P50 improves by 0.65–0.72 m, P90 by ≈1.8 m and RMS by
  0.8–1.0 m, with the same picture for both methods.
- The gain is uneven across routes: tokyo3 −2.9 m, tokyo1 −0.9 to −1.4 m,
  nagoya1 −0.35 m. nagoya3 gets slightly worse (+0.16–0.19 m, at the edge of
  the per-route noise). nagoya2 and tokyo2 are flat. The `<10 m` share drops
  by ≈0.3 pp.
- B=15 is not enough even for the correct algorithm (identical to default),
  matching the synthetic convergence test in D-039.
- A fresh resampling seed per call has no measurable effect.
- PF-only errors on PPC are dominated by urban multipath/NLOS: WLS P50 is
  36 m on tokyo1, and inuex35 reports a 47 m AllRMS on the same run. The
  official honest PPC score (≤0.5–3 m) cannot move, and the canonical 59.22%
  pipeline uses `AmbiguityBasinParticleFilter`, not this resampler.

## Decision status

The default stays `megopolis` (D-039). The README-headline UrbanNav PF
smoother (`pf_smoother_runtime.py`, default `megopolis`) is where a switch
matters most, but UrbanNav data is not on the dev machine. Re-evaluate the
default there, with B=60 as the candidate, before switching.
