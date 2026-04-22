import numpy as np

from gnss_gpu.pf_smoother_alignment import append_forward_alignment
from gnss_gpu.pf_smoother_epoch_state import create_epoch_forward_state
from gnss_gpu.pf_smoother_runtime import ForwardRunBuffers


def test_append_forward_alignment_appends_matching_epoch():
    buffers = ForwardRunBuffers()
    state = create_epoch_forward_state(0.5)
    state.imu_stop_detected = True

    result = append_forward_alignment(
        buffers,
        run_name="Odaiba",
        tow=10.0,
        pf_estimate_now=np.array([1.0, 2.0, 3.0]),
        gt=np.array([[4.0, 5.0, 6.0]]),
        our_times=np.array([10.01]),
        measurements=[object(), object()],
        epoch_index=2,
        skip_valid_epochs=1,
        use_smoother=False,
        collect_epoch_diagnostics=False,
        epoch_state=state,
        rbpf_velocity_kf=False,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
    )

    assert result.aligned is True
    assert result.gt_index == 0
    assert result.aligned_epoch_index == 0
    np.testing.assert_allclose(buffers.forward_aligned[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(buffers.gt_aligned[0], [4.0, 5.0, 6.0])
    assert buffers.aligned_stop_flags == [True]
    assert buffers.aligned_indices == []


def test_append_forward_alignment_skips_before_requested_window_and_time_miss():
    buffers = ForwardRunBuffers()
    state = create_epoch_forward_state(0.5)

    before_window = append_forward_alignment(
        buffers,
        run_name="Odaiba",
        tow=10.0,
        pf_estimate_now=np.array([1.0, 2.0, 3.0]),
        gt=np.array([[4.0, 5.0, 6.0]]),
        our_times=np.array([10.0]),
        measurements=[],
        epoch_index=0,
        skip_valid_epochs=1,
        use_smoother=False,
        collect_epoch_diagnostics=False,
        epoch_state=state,
        rbpf_velocity_kf=False,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
    )
    time_miss = append_forward_alignment(
        buffers,
        run_name="Odaiba",
        tow=10.0,
        pf_estimate_now=np.array([1.0, 2.0, 3.0]),
        gt=np.array([[4.0, 5.0, 6.0]]),
        our_times=np.array([10.2]),
        measurements=[],
        epoch_index=1,
        skip_valid_epochs=1,
        use_smoother=False,
        collect_epoch_diagnostics=False,
        epoch_state=state,
        rbpf_velocity_kf=False,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
    )

    assert before_window.aligned is False
    assert before_window.gt_index is None
    assert time_miss.aligned is False
    assert time_miss.gt_index == 0
    assert buffers.forward_aligned == []


def test_append_forward_alignment_adds_epoch_diagnostics():
    buffers = ForwardRunBuffers()
    buffers.append_smoother_observations(None, None, None)
    state = create_epoch_forward_state(0.75)
    state.used_imu = True
    state.used_tdcp = True
    state.doppler_update_epoch = {"doppler_hz": []}
    state.dd_pr_input_pairs = 3
    state.wl_input_pairs = 4
    state.wl_fixed_pairs = 2

    result = append_forward_alignment(
        buffers,
        run_name="Odaiba",
        tow=10.0,
        pf_estimate_now=np.array([1.0, 2.0, 3.0]),
        gt=np.array([[4.0, 5.0, 6.0]]),
        our_times=np.array([10.0]),
        measurements=[1, 2, 3],
        epoch_index=1,
        skip_valid_epochs=0,
        use_smoother=True,
        collect_epoch_diagnostics=True,
        epoch_state=state,
        rbpf_velocity_kf=True,
        gate_ess_ratio=0.5,
        gate_spread_m=2.0,
        carrier_anchor_sigma_m=0.25,
    )

    assert result.aligned is True
    assert buffers.aligned_indices == [0]
    row = buffers.aligned_epoch_diagnostics[0]
    assert row["run"] == "Odaiba"
    assert row["store_epoch_index"] == 0
    assert row["n_measurements"] == 3
    assert row["used_imu"] is True
    assert row["used_tdcp"] is True
    assert row["used_doppler_kf"] is True
    assert row["dd_pr_input_pairs"] == 3
    assert row["widelane_input_pairs"] == 4
