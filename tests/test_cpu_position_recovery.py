"""Paired publication comparison retains missing epochs and recovery events."""

import numpy as np

from experiments.run_cpu_position_recovery import compare_case
from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionRecoveryConfig
from gnss_gpu.positioning_cpu import solve_pseudorange_position


def test_three_policy_comparison_preserves_raw_stream_and_reanchor_event():
    satellites = constellation()
    channels, positions = [], []
    for block in range(30):
        target = np.array([0. if block < 2 else 50., 0., 0.])
        ranges = np.linalg.norm(satellites-target, axis=1)
        raw = solve_pseudorange_position(satellites, ranges, initial_position_m=(-25., 5., 0.))
        assert raw is not None
        for prn, distance in enumerate(ranges, 1):
            channels.append(dict(seed=1, environment="open", scenario="test", block=block,
                                 prn=prn, used=True, delay_m=distance-22e6))
        positions.append(dict(block=block, stage="dry", available=True, fresh_channels=12,
                              east_m=raw.position_m[0], north_m=raw.position_m[1], up_m=raw.position_m[2]))
    result = compare_case(dict(channels=channels, positions=positions), "test",
                          PositionGateConfig(confirmation_blocks=2), PositionRecoveryConfig(),
                          satellites, np.zeros(3))
    summary = {row["mode"]:row for row in result["summary"]}
    assert len(result["positions"]) == 90
    assert summary["raw"]["available"] == 30
    assert summary["legacy_gate"]["available"] < summary["recovery_gate"]["available"] < 30
    assert summary["recovery_gate"]["reanchor_events"] == 1
    assert summary["raw"]["reanchor_events"] == summary["legacy_gate"]["reanchor_events"] == 0
    event = next(row for row in result["positions"] if row["reanchored"])
    assert event["mode"] == "recovery_gate" and not event["available"]
    assert event["position_error_m"] is None
    assert result["max_raw_replay_delta_m"] < 1e-6
