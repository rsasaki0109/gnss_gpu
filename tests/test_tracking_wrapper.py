"""CPU-side validation tests for tracking wrappers."""

import numpy as np
import pytest

from gnss_gpu.tracking import (
    ChannelState,
    ScalarTracker,
    TrackingConfig,
    VectorTracker,
)


def test_tracking_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="sampling_freq must be positive"):
        TrackingConfig(sampling_freq=0.0)
    with pytest.raises(ValueError, match="intermediate_freq must be finite"):
        TrackingConfig(intermediate_freq=np.nan)
    with pytest.raises(ValueError, match="integration_time must be positive"):
        TrackingConfig(integration_time=0.0)
    with pytest.raises(ValueError, match="dll_bandwidth must be non-negative"):
        TrackingConfig(dll_bandwidth=-1.0)
    with pytest.raises(ValueError, match="pll_bandwidth must be finite"):
        TrackingConfig(pll_bandwidth=np.inf)
    with pytest.raises(ValueError, match="correlator_spacing must be positive"):
        TrackingConfig(correlator_spacing=0.0)


def test_channel_state_rejects_invalid_values_before_gpu_check():
    with pytest.raises(ValueError, match="prn must be in"):
        ChannelState(0)
    with pytest.raises(ValueError, match="code_phase must be finite"):
        ChannelState(1, code_phase=np.nan)
    with pytest.raises(ValueError, match="code_freq must be positive"):
        ChannelState(1, code_freq=0.0)
    with pytest.raises(ValueError, match="carrier_freq must be finite"):
        ChannelState(1, carrier_freq=np.inf)


def test_scalar_tracker_initialize_rejects_invalid_inputs_before_gpu_check():
    tracker = ScalarTracker(TrackingConfig())

    with pytest.raises(ValueError, match="prn_list must be 1-D"):
        tracker.initialize([[1]], [0.0], [0.0])
    with pytest.raises(ValueError, match="prn_list must contain at least one PRN"):
        tracker.initialize([], [], [])
    with pytest.raises(ValueError, match="prn_list must contain integers"):
        tracker.initialize([1.5], [0.0], [0.0])
    with pytest.raises(ValueError, match="prn_list values must be in"):
        tracker.initialize([33], [0.0], [0.0])
    with pytest.raises(ValueError, match="code_phases length must match"):
        tracker.initialize([1, 2], [0.0], [0.0, 0.0])
    with pytest.raises(ValueError, match="doppler_freqs length must match"):
        tracker.initialize([1], [0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="code_phases must be finite"):
        tracker.initialize([1], [np.nan], [0.0])
    with pytest.raises(ValueError, match="doppler_freqs must be finite"):
        tracker.initialize([1], [0.0], [np.inf])


def test_scalar_tracker_process_rejects_invalid_signal_before_gpu_check():
    tracker = ScalarTracker(TrackingConfig())

    with pytest.raises(ValueError, match="tracker has no initialized channels"):
        tracker.process(np.ones(8, dtype=np.float32))

    tracker.channels = [object()]

    with pytest.raises(ValueError, match="signal_block must be 1-D"):
        tracker.process(np.ones((2, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="signal_block must contain at least one sample"):
        tracker.process(np.array([], dtype=np.float32))
    with pytest.raises(ValueError, match="signal_block must be finite"):
        tracker.process([0.0, np.nan])


def test_vector_tracker_rejects_invalid_inputs_before_gpu_work():
    with pytest.raises(ValueError, match="initial_pos_ecef must have shape"):
        VectorTracker(TrackingConfig(), [0.0, 0.0])
    with pytest.raises(ValueError, match="initial_pos_ecef must be finite"):
        VectorTracker(TrackingConfig(), [0.0, np.nan, 0.0])

    tracker = VectorTracker(TrackingConfig(), [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="tracker has no initialized channels"):
        tracker.process(np.ones(8, dtype=np.float32), np.ones((1, 3)), np.ones((1, 3)))

    tracker.channels = [object()]

    with pytest.raises(ValueError, match="sat_ecef must have shape"):
        tracker.process(np.ones(8, dtype=np.float32), np.ones((2, 3)), np.ones((1, 3)))
    with pytest.raises(ValueError, match="sat_vel must have shape"):
        tracker.process(np.ones(8, dtype=np.float32), np.ones((1, 3)), np.ones((1, 2)))
    bad_sat = np.ones((1, 3), dtype=np.float64)
    bad_sat[0, 0] = np.inf
    with pytest.raises(ValueError, match="sat_ecef must be finite"):
        tracker.process(np.ones(8, dtype=np.float32), bad_sat, np.ones((1, 3)))
