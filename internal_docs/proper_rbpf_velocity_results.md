# Proper RBPF Velocity Results

Date: 2026-04-20
Branch: `feature/carrier-phase-imu`

## Implemented

- Reworked PF device state to keep per-particle velocity as a Gaussian KF state:
  `{x, y, z, cb, mu_vx, mu_vy, mu_vz, Sigma_v[3][3]}`.
- Added covariance-aware predict:
  `x_new ~ N(x + mu_v * dt, sigma_pos^2 I + dt^2 Sigma_v)`.
- Added per-particle Doppler KF update for `mu_v` and `Sigma_v`.
- Kept legacy sampled-velocity Doppler path available, but default off.
- Updated `odaiba_rbpf_velocity` to use proper RBPF velocity instead of sampled
  velocity.

## Odaiba Full Runs

All runs used `/tmp/UrbanNav-Tokyo` and the Odaiba smoother stack.

| Config | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Reference `run_pf_smoother_odaiba_reference.sh` | 1.37 m | 5.50 m | 1.20 m | 4.01 m | Baseline guard passed |
| Proper RBPF default, `Q_v=1.0`, Doppler sigma `0.5` | 1.60 m | 5.49 m | 1.46 m | 4.87 m | Negative |
| Proper RBPF tuned, `Q_v=0.1`, Doppler sigma `2.0` | 1.27 m | 4.98 m | 1.21 m | 4.29 m | Better than default RBPF, not submeter |

## Odaiba 3k Sweep

| Config | Doppler KF | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `Q_v=1.0`, Doppler sigma `0.5` | on | 1.15 m | 2.06 m | 0.96 m | 2.25 m | Short-window submeter |
| `Q_v=1.0`, Doppler KF disabled | off | 1.04 m | 5.10 m | 1.02 m | 5.09 m | RMS regresses |
| `Q_v=1.0`, Doppler sigma `2.0` | on | 1.03 m | 2.15 m | 0.99 m | 2.33 m | Marginal short-window submeter |
| `Q_v=1.0`, Doppler sigma `5.0` | on | 1.15 m | 2.77 m | 1.04 m | 2.81 m | Worse |
| `Q_v=0.1`, Doppler sigma `2.0` | on | 0.88 m | 1.97 m | 0.89 m | 2.10 m | Best 3k candidate |

## Outcome

Proper RBPF fixed the dimensionality issue of the sampled-velocity Phase 2 path,
but it did not reach the Odaiba full-run submeter target. The best full Odaiba
candidate was `Q_v=0.1`, Doppler sigma `2.0`, with SMTH P50 `1.21 m`.

The 3k sweep shows the Doppler KF can help the early Odaiba segment, but the
benefit does not survive the full route. Because full Odaiba remains above
`1.00 m`, the proper RBPF preset should not be promoted over the existing
reference stack.

## Shinjuku Regression

The tuned candidate (`Q_v=0.1`, Doppler sigma `2.0`) was also run on full
Shinjuku:

| Config | FWD P50 | FWD RMS | SMTH P50 | SMTH RMS | Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| Proper RBPF tuned, Shinjuku full | 2.27 m | 7.54 m | 2.27 m | 7.74 m | Doppler KF used 20127/20127 epochs |

Compared with the documented Shinjuku `odaiba_reference` check
(`2.61 m` SMTH P50, `6.87 m` SMTH RMS), proper RBPF improves median error but
regresses RMS. This reinforces keeping it as an experimental path instead of a
promoted preset.

## Next

- Region-aware DD/ESS/Doppler residual gating was tried on 2026-04-21 and was
  negative; see `internal_docs/rbpf_velocity_gated_results.md`.
- Try TDCP-derived velocity rows for the KF update; raw Doppler appears helpful
  only in easier route segments.
- Keep naive sampled velocity default off.
