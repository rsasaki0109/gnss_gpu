"""CPU-side validation tests for the signal_sim wrapper."""

import numpy as np
import pytest

from gnss_gpu.signal_sim import SignalSimulator, prn_label_to_system


def _valid_channel():
    return {
        "prn": 1,
        "code_phase": 0.0,
        "carrier_phase": 0.0,
        "doppler_hz": 1000.0,
        "amplitude": 1.0,
        "nav_bit": 1,
    }


def test_signal_simulator_init_rejects_invalid_config():
    with pytest.raises(ValueError, match="sampling_freq must be positive"):
        SignalSimulator(sampling_freq=0.0)

    with pytest.raises(ValueError, match="intermediate_freq must be finite"):
        SignalSimulator(intermediate_freq=np.nan)

    with pytest.raises(ValueError, match="noise_seed must be an integer"):
        SignalSimulator(noise_seed="bad")


def test_generate_epoch_rejects_invalid_channels_before_gpu_import():
    sim = SignalSimulator()

    with pytest.raises(ValueError, match="channels must be a list"):
        sim.generate_epoch({"prn": 1})

    with pytest.raises(ValueError, match="missing required key 'doppler_hz'"):
        sim.generate_epoch([{
            "prn": 1,
            "code_phase": 0.0,
            "carrier_phase": 0.0,
            "amplitude": 1.0,
            "nav_bit": 1,
        }])

    with pytest.raises(ValueError, match="n_samples must be a positive integer"):
        sim.generate_epoch([_valid_channel()], n_samples=0)


def test_write_bin_rejects_invalid_inputs():
    with pytest.raises(ValueError, match="iq_data must be non-empty"):
        SignalSimulator.write_bin(np.array([], dtype=np.float32), "out.bin")

    with pytest.raises(ValueError, match="Unknown format"):
        SignalSimulator.write_bin(np.array([0.0, 0.0], dtype=np.float32), "out.bin", fmt="bad")


def test_prn_label_to_system_rejects_malformed_labels():
    with pytest.raises(ValueError, match="Malformed PRN label"):
        prn_label_to_system("")

    with pytest.raises(ValueError, match="Unknown GNSS system prefix"):
        prn_label_to_system("X05")

    with pytest.raises(ValueError, match="Invalid PRN number"):
        prn_label_to_system("Gxx")
