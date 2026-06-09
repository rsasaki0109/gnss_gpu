"""CPU-side validation tests for the acquisition wrapper."""

import numpy as np
import pytest

from gnss_gpu.acquisition import Acquisition


def test_acquisition_init_rejects_invalid_config():
    with pytest.raises(ValueError, match="sampling_freq must be positive"):
        Acquisition(sampling_freq=0.0)
    with pytest.raises(ValueError, match="intermediate_freq must be finite"):
        Acquisition(sampling_freq=4.092e6, intermediate_freq=np.nan)
    with pytest.raises(ValueError, match="doppler_range must be non-negative"):
        Acquisition(sampling_freq=4.092e6, doppler_range=-1.0)
    with pytest.raises(ValueError, match="doppler_step must be positive"):
        Acquisition(sampling_freq=4.092e6, doppler_step=0.0)
    with pytest.raises(ValueError, match="threshold must be finite"):
        Acquisition(sampling_freq=4.092e6, threshold=np.inf)


def test_acquire_rejects_invalid_inputs_before_gpu_import():
    acq = Acquisition(sampling_freq=4.092e6)

    with pytest.raises(ValueError, match="signal must be 1-D"):
        acq.acquire(np.ones((2, 4), dtype=np.float32), prn_list=[1])
    with pytest.raises(ValueError, match="signal must contain at least one sample"):
        acq.acquire([], prn_list=[1])
    with pytest.raises(ValueError, match="signal must be finite"):
        acq.acquire([0.0, np.nan], prn_list=[1])
    with pytest.raises(ValueError, match="prn_list must be 1-D"):
        acq.acquire([0.0, 1.0], prn_list=[[1]])
    with pytest.raises(ValueError, match="prn_list must contain at least one PRN"):
        acq.acquire([0.0, 1.0], prn_list=[])
    with pytest.raises(ValueError, match="prn_list must contain integers"):
        acq.acquire([0.0, 1.0], prn_list=[1.5])
    with pytest.raises(ValueError, match="prn_list values must be in"):
        acq.acquire([0.0, 1.0], prn_list=[33])


def test_generate_test_signal_rejects_invalid_inputs_before_gpu_import():
    kwargs = dict(
        code_phase=0.0,
        doppler=0.0,
        snr_db=20.0,
        sampling_freq=4.092e6,
    )

    with pytest.raises(ValueError, match="prn must be in"):
        Acquisition.generate_test_signal(0, **kwargs)
    with pytest.raises(ValueError, match="code_phase must be finite"):
        Acquisition.generate_test_signal(1, **{**kwargs, "code_phase": np.nan})
    with pytest.raises(ValueError, match="doppler must be finite"):
        Acquisition.generate_test_signal(1, **{**kwargs, "doppler": np.inf})
    with pytest.raises(ValueError, match="snr_db must be finite"):
        Acquisition.generate_test_signal(1, **{**kwargs, "snr_db": np.nan})
    with pytest.raises(ValueError, match="sampling_freq must be positive"):
        Acquisition.generate_test_signal(1, **{**kwargs, "sampling_freq": 0.0})
    with pytest.raises(ValueError, match="duration_s must be positive"):
        Acquisition.generate_test_signal(1, **kwargs, duration_s=0.0)
