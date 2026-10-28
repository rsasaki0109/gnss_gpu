"""Saved receiver schema, truth boundary and unavailable-error denominators."""

from dataclasses import asdict

import numpy as np
import pytest

from experiments.run_cpu_position_quality import decode_thresholds, error_summary, replay_channels
from experiments.run_cpu_rain_positioning import constellation
from gnss_gpu.code_detection_cpu import calibrate_ca_threshold
from gnss_gpu.position_quality_cpu import PositionGateConfig, PositionRecoveryConfig
from gnss_gpu.signal_sim_cpu import CHIP_RATE


def test_saved_flat_threshold_schema_roundtrips_without_new_calibration():
    gate = calibrate_ca_threshold(np.arange(1., 100.), 2*CHIP_RATE, prn=2)
    metadata = dict(thresholds={"2":asdict(gate)}, sampling_freq_hz=2*CHIP_RATE)
    assert decode_thresholds(metadata)[2] == gate
    with pytest.raises(ValueError):
        decode_thresholds(dict(thresholds={"1":asdict(gate)}, sampling_freq_hz=2*CHIP_RATE))


@pytest.mark.parametrize("recovery", [None, PositionRecoveryConfig()])
def test_replay_decisions_ignore_truth_weather_and_visibility_fields(recovery):
    sat = constellation()
    channels = []
    for block in range(2):
        for prn, satellite in enumerate(sat, 1):
            channels.append(dict(block=block, prn=prn, used=True,
                                 delay_m=np.linalg.norm(satellite)-22e6,
                                 direct_visible=True, truth_error_m=0., stage="dry"))
    changed = [dict(row, direct_visible=False, truth_error_m=1e9, stage="wet") for row in channels]
    config = PositionGateConfig(confirmation_blocks=2)
    first = replay_channels(channels, config, sat, recovery_config=recovery)
    second = replay_channels(changed, config, sat, recovery_config=recovery)
    assert [(block, d.accepted, d.reason) for block,d in first] == [(block, d.accepted, d.reason) for block,d in second]
    assert first[1][1].accepted
    np.testing.assert_array_equal(first[1][1].position_m, second[1][1].position_m)


def test_withheld_and_failed_positions_are_not_turned_into_zero_error():
    summary = error_summary([dict(available=False, position_error_m=None, horizontal_error_m=None),
                             dict(available=True, position_error_m=150., horizontal_error_m=120.)])
    assert summary["blocks"] == 2 and summary["unavailable"] == 1
    assert summary["available_error_over_100_m"] == 1
    assert summary["position_max_m_conditional"] == 150.
    missing = error_summary([dict(available=False)])
    assert missing["position_p95_m_conditional"] is None
