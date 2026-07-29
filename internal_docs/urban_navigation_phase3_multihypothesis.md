# Phase 3 multi-hypothesis navigation and recovery

Phase 3 extends the existing `AmbiguityBasinParticleFilter`; it does not add
a competing PF implementation.

## Integrated state

- Every integer basin retains its six-state ECEF position/velocity Kalman
  conditional.
- `PreintegratedNavigationDelta` adds full 6x6 IMU process covariance and
  accel/gyro bias Jacobians.
- `InertialBiasState` carries accel/gyro bias covariance and random-walk
  growth. Bias uncertainty is projected through the preintegration Jacobian
  into each basin's navigation covariance.
- Missing IMU falls back to the existing constant-velocity prediction.

## Map-aware proposal

`MapProposal` supplies a road/map position and covariance. The controller
fuses proposals into clones of the leading basins, preserves proposal
provenance, and relies on the existing position-diversity reserve and finite
deduplication radius to keep distinct road branches alive.

## Outage safety

The state machine is `tracking -> coasting -> degraded -> reacquiring`.

- Any GNSS outage immediately clears the temporal FIX streak and suppresses
  FIX output.
- Navigation covariance inflates while coasting.
- Reacquisition requires three consecutive accepted truth-free Evidence API
  decisions by default.
- Unsafe or missing evidence resets the reacquisition streak.
- A high-gamma single basin during an outage is reported as premature
  collapse, never as a safe fix.

Run `python experiments/audit_phase3_outage_recovery.py`. The deterministic
two-road audit shows the retained alternative recovering in 3 evidence epochs
after outage, while a greedy one-road baseline remains 10 m from the selected
road. M4 hashes remain pinned. This synthetic controller audit verifies state
logic; city-scale accuracy remains part of Phase 5.
