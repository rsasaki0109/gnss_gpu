# Phase 2 Per-Particle Velocity RBPF Results

Date: 2026-04-20
Branch: `feature/carrier-phase-imu`

## Implemented

- Extended CUDA PF state from `{x, y, z, cb}` to `{x, y, z, vx, vy, vz, cb}`.
- Added per-particle Doppler velocity-domain likelihood/update.
- Used receiver/RINEX Doppler convention: `range_rate = -doppler_hz * wavelength_m`.
- Exposed experiment knobs:
  - `--doppler-per-particle`
  - `--doppler-sigma-mps`
  - `--doppler-velocity-update-gain`
  - `--doppler-max-velocity-update-mps`
  - `--doppler-min-sats`
  - `--pf-sigma-vel`
  - `--pf-velocity-guide-alpha`
  - `--pf-init-spread-vel`
- Disabled Doppler replay in the backward smoother because the backward PF uses reversed velocity.

## Full Odaiba Runs

All runs used `/tmp/UrbanNav-Tokyo`, 200k particles, `odaiba_best_accuracy` base settings.

| Config | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `odaiba_best_accuracy` baseline on current HEAD | 1.26 m | 5.74 m | 1.22 m | 4.44 m | Baseline intact, but differs from prior 1.14 m note |
| Doppler sign bug, sigma 0.5, gain 0.25 | 1623.71 m | 1737.27 m | 1627.18 m | 1781.29 m | Invalid; fixed by sign convention |
| Doppler fixed sign, sigma 0.5, gain 0.25, no velocity persistence | 1.37 m | 5.77 m | 1.36 m | 4.46 m | Negative |
| Doppler fixed sign, sigma 5.0, gain 0.0, no velocity persistence | 1.38 m | 5.16 m | 1.25 m | 4.28 m | Negative |
| Doppler fixed sign, sigma 20.0, gain 0.0, no velocity persistence | 1.13 m | 4.60 m | 1.30 m | 4.15 m | Forward improves; smoother still negative |
| Doppler fixed sign, sigma 5.0, gain 0.25, alpha 0.9, sigma_vel 0.05, init spread 0.2 | 1.63 m | 5.21 m | 1.48 m | 4.20 m | Negative |
| Doppler fixed sign, sigma 0.5, gain 0.25, alpha 0.5, sigma_vel 0.25, init spread 1.0 | 1.68 m | 6.33 m | 1.58 m | 6.00 m | Negative |

## Outcome

Submeter was not achieved in Phase 2 Step A+B. The safest retained preset,
`odaiba_rbpf_velocity`, is intentionally conservative:

- `--doppler-sigma-mps 20.0`
- `--doppler-velocity-update-gain 0.0`
- `--pf-velocity-guide-alpha 1.0`
- `--pf-sigma-vel 0.0`
- `--pf-init-spread-vel 0.0`

The main finding is that Doppler can improve forward P50 slightly when heavily
downweighted, but the smoother does not benefit. Strong per-particle velocity
persistence worsens Odaiba, likely because Doppler outliers and stop epochs
perturb the carrier/DD particle cloud more than they help.

## Next

- Add direction-aware Doppler replay before using Doppler in the backward pass.
- Add per-epoch Doppler residual gating against IMU/TDCP velocity before applying
  per-particle likelihood.
- Move to Step C: TDCP per-particle velocity likelihood with carrier-derived
  range-rate rows, because TDCP should be less noisy than raw Doppler.
