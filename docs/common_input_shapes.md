# Common input shapes

Quick reference for major public Python APIs. Invalid shapes raise `ValueError`
at the wrapper (CPU-side, no GPU required). ECEF [m], pseudorange [m], carrier
[cycles], GPS time [s]. See `tests/test_*_wrapper.py` for examples.

## Ephemeris

| API | argument | shape | notes |
|---|---|---|---|
| `compute_satellite_position` | `params_flat` | `(n_sat × EPHEMERIS_PARAMS_SIZE,)` | packed `EphemerisParams` buffer (`float64`); use `EPHEMERIS_PARAMS_SIZE` from `_gnss_gpu_ephemeris` |
| | `gps_time` | scalar | finite seconds |
| | `n_sat` | int | positive |
| `compute_satellite_position_batch` | `gps_times` | `(n_epoch,)` | non-empty, finite |
| | returns `sat_ecef`, `sat_clk` | `(n_epoch, n_sat, 3)`, `(n_epoch, n_sat)` | |
| `Ephemeris.compute` | returns `sat_ecef`, `sat_clk` | `(n_sat, 3)`, `(n_sat,)` | high-level wrapper over RINEX nav |
| `Ephemeris.compute_batch` | `gps_times` | `(n_epoch,)` | returns `(n_epoch, n_sat, 3)`, `(n_epoch, n_sat)` |

## Multipath

| API | argument | shape | notes |
|---|---|---|---|
| `MultipathSimulator` | `reflector_planes` | `(n_ref, 6)` | `[px, py, pz, nx, ny, nz]` per row; `n_ref ≥ 1` |
| `.simulate` | `rx_ecef` | `(3,)` or `(n_rx, 3)` | |
| | `sat_ecef` | `(n_sat, 3)` | |
| | returns `delays`, `attenuations` | `(n_rx, n_sat)` each | |
| `.corrupt_pseudoranges` | `clean_pr` | `(n_epoch, n_sat)` | |
| | `rx_ecef` | `(n_epoch, 3)` | one receiver per epoch |
| | `sat_ecef` | `(n_epoch, n_sat, 3)` | |

## EKF (`EKFPositioner`)

State vector length 8: `[x, y, z, vx, vy, vz, clock_bias, clock_drift]`.

| API | argument | shape | notes |
|---|---|---|---|
| `.initialize` | `position_ecef` | `(3,)` | |
| `.update` | `sat_ecef` | `(n_sat, 3)` | `n_sat ≥ 1` |
| | `pseudoranges` | `(n_sat,)` | |
| | `weights` | `(n_sat,)` | optional; non-negative; default `1/σ_pr²` |
| `.get_position` / `.get_velocity` | returns | `(3,)` | |
| `.get_covariance` | returns | `(8, 8)` | |

## Particle filter (`ParticleFilter`, `SVGDParticleFilter`)

| API | argument | shape | notes |
|---|---|---|---|
| `.initialize` | `position_ecef` | `(3,)` | scatters `n_particles` samples |
| `.predict` | `velocity` | `(3,)` | optional ECEF m/s; default zero |
| `.update` | `sat_ecef` | `(n_sat, 3)` | |
| | `pseudoranges` | `(n_sat,)` | |
| | `weights` | `(n_sat,)` | optional; default ones |
| `.estimate` | returns | `(4,)` | `[x, y, z, clock_bias]` |
| `.get_particles` | returns | `(n_particles, 4)` | |

`ParticleFilter3D` / `ParticleFilterDevice`: same measurement shapes; PF3D adds mesh at init, device adds DD/carrier buffer paths.

## RTK (`RTKSolver`)

| API | argument | shape | notes |
|---|---|---|---|
| `__init__` | `base_ecef` | `(3,)` | |
| `.solve_float` / `.solve_fixed` | `rover_pr`, `base_pr`, `rover_carrier`, `base_carrier` | `(n_sat,)` each | `n_sat ≥ 2` for DD |
| | `sat_ecef` | `(n_sat, 3)` | |
| | returns `position` | `(3,)` | |
| | returns `ambiguities` | `(n_sat − 1,)` | float DD cycles |
| `.solve_batch` | obs arrays | `(n_epoch, n_sat)` | all four obs arrays |
| | `sat_ecef` | `(n_epoch, n_sat, 3)` | |
| | returns `positions` | `(n_epoch, 3)` | |

## Skyplot (`VulnerabilityMap`)

| API | argument | shape | notes |
|---|---|---|---|
| `__init__` | `origin_lla` | `(3,)` | `(lat_deg, lon_deg, alt_m)` |
| `.evaluate` | `sat_ecef` | `(n_sat, 3)` | `n_sat ≥ 1` |
| | returns `pdop`, `hdop`, `vdop`, `gdop`, `n_visible` | `(n_side, n_side)` each | `n_side` from grid config |

## Validation policy

Public Python wrappers raise **`ValueError`** for invalid inputs (CPU-side, no GPU
required). Direct `_gnss_gpu_*` pybind entrypoints raise **`RuntimeError`** with the
same message text so bypass callers still fail fast before native kernels run.

**`dt` (seconds)** — EKF predict/update and tracking vector updates require **`dt > 0`**
finite. Particle filters (`ParticleFilter`, `SVGDParticleFilter`, `ParticleFilterDevice`)
allow **`dt ≥ 0`**; zero means no motion step (position/clock unchanged, weights only).

**`ess_threshold`** — `ParticleFilter` resample gate accepts **`[0, 1]`** inclusive.
`0` disables automatic resampling (ESS check skipped); values in `(0, 1]` trigger
resample when normalized ESS falls below the threshold.

Shared finite/shape/range helpers live in **`gnss_gpu.input_validation`**; wrappers
and bindings call these instead of duplicating checks. Module-specific rules (e.g.
measurement counts, state vector length) stay in each API section above.
