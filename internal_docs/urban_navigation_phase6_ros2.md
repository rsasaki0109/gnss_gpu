# Phase 6: ROS 2 lifecycle and deterministic replay

Date: 2026-07-29

## Delivered

- `IntegratedNavigationLifecycleNode`, an auto-configured/activated ROS 2
  lifecycle node.
- GNSS (`NavSatFix`), IMU (`Imu`), map-context (`String`), navigation-fix, and
  `DiagnosticArray` topics.
- Fail-closed parameter validation; active nodes reject parameter mutation.
- Per-sensor monotonic timestamp enforcement, future-skew rejection,
  idempotent duplicate handling, conflicting-duplicate latching, and finite
  payload validation.
- GNSS/IMU/map watchdogs with missing/stale diagnostics and counters.
- Safe fallback that preserves only the last accepted fix and inflates its
  covariance.
- Explicit configure, activate, deactivate, cleanup, shutdown, error, and
  in-process restart semantics.
- A ROS-independent deterministic bag replay command and locked audit.

## Locked replay

Inputs and output:

- `internal_docs/phase6_ros2_replay_input_2026_07_29.json`
- `internal_docs/phase6_ros2_replay_result_2026_07_29.json`

The ten-event audit contains six accepted events and one each of duplicate,
conflicting duplicate, out-of-order, and future-skew disposition. A restart is
injected before event 7. Normal operation is recovered only after fresh IMU,
map, and GNSS events arrive after that restart.

Replay SHA-256:

`A003EE14D05547F4020EDD2E18916AEA132142AE6F1CB72686AE9030A0C66BBA`

Reproduce from the repository root:

```bash
PYTHONPATH=ros2/gnss_gpu_ros python -m gnss_gpu_ros.replay_contract \
  --input internal_docs/phase6_ros2_replay_input_2026_07_29.json \
  --output internal_docs/phase6_ros2_replay_result_2026_07_29.json
```

## Safety state machine

`UNCONFIGURED → INACTIVE → ACTIVE` is the only activation path. Cleanup is
allowed only from inactive. Lifecycle errors force `ERROR + SAFE_FALLBACK`.
Restart clears timestamps, payloads, integrity latches, and the last safe fix,
so stale data cannot leak across a process-equivalent restart.

While active:

- all required inputs fresh and integrity-clean: `NORMAL`;
- missing/stale input or latched integrity error: `SAFE_FALLBACK`;
- a valid newer event clears only its own sensor's integrity latch;
- an exact duplicate has no state effect;
- an older timestamp cannot replace current state.

## Validation

The ROS-free suite runs on any development host and covers all named anomaly
classes. The wrapper also has message-conversion and missing-runtime tests.
Actual lifecycle integration still requires a ROS 2 Jazzy environment; this
Windows checkout does not provide `rclpy`, so `colcon` execution is deferred to
the Phase 7 ROS image build.
