import numpy as np

from gnss_gpu.pf_smoother_epoch_history import ForwardEpochHistory


def test_forward_epoch_history_defaults_and_limit():
    history = ForwardEpochHistory()

    assert history.dt_for(100.0) == 0.1
    assert history.has_previous_motion(0.1) is False
    assert history.reached_limit(max_epochs=0, skip_valid_epochs=10) is False
    assert history.reached_limit(max_epochs=2, skip_valid_epochs=1) is False


def test_forward_epoch_history_advance_copies_previous_epoch_state():
    history = ForwardEpochHistory()
    measurements = [{"sat": 1}]
    estimate = np.array([1.0, 2.0, 3.0])
    pf_state = np.array([1.0, 2.0, 3.0, 4.0])

    history.advance(
        tow=123.4,
        measurements=measurements,
        pf_estimate_now=estimate,
        pf_state=pf_state,
    )
    estimate[:] = 0.0
    pf_state[:] = 0.0
    measurements.append({"sat": 2})

    assert history.prev_tow == 123.4
    assert history.dt_for(124.0) == 0.5999999999999943
    assert history.has_previous_motion(0.6) is True
    assert history.prev_measurements == [{"sat": 1}]
    assert np.array_equal(history.prev_estimate, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(history.prev_pf_estimate, np.array([1.0, 2.0, 3.0]))
    assert np.array_equal(history.prev_pf_state, np.array([1.0, 2.0, 3.0, 4.0]))
    assert history.epochs_done == 1


def test_forward_epoch_history_reached_limit_after_skip_window():
    history = ForwardEpochHistory(epochs_done=3)

    assert history.reached_limit(max_epochs=2, skip_valid_epochs=1) is True
    assert history.reached_limit(max_epochs=3, skip_valid_epochs=1) is False
