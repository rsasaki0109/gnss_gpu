import numpy as np
import pytest

from experiments.audit_boreas_reference import predict_sensor_positions, sensor_rotation


def test_extrinsic_translation_rotates_with_sensor_attitude():
    reference = np.array([[0., 1., 2., 3., 0., 0., np.pi/2], [.005, 1., 2., 3., 0., 0., np.pi/2]])
    predicted, valid = predict_sensor_positions(reference, [.002], np.array([1., 0., 0.]))
    assert valid.all()
    np.testing.assert_allclose(predicted, [[1., 1., 3.]], atol=1e-14)


def test_no_extrapolation_or_interpolation_over_missing_reference():
    reference = np.array([[0., 0., 0., 0., 0., 0., 0.], [1., 1., 0., 0., 0., 0., 0.]])
    predicted, valid = predict_sensor_positions(reference, [-.1, .5, 1.1], np.zeros(3))
    assert not valid.any()
    assert np.isnan(predicted).all()


def test_attitude_wrap_does_not_flip_a_lever_arm():
    reference = np.array([[0., 0., 0., 0., 0., 0., np.pi-.01], [.005, 0., 0., 0., 0., 0., -np.pi+.01]])
    predicted, valid = predict_sensor_positions(reference, [.0025], np.array([1., 0., 0.]))
    np.testing.assert_allclose(predicted[valid], [[-1., 0., 0.]], atol=1e-14)


def test_rotation_is_proper_and_duplicate_reference_times_fail():
    rotation = sensor_rotation(.3, -.2, .1)
    np.testing.assert_allclose(rotation@rotation.T, np.eye(3), atol=1e-14)
    assert np.linalg.det(rotation) == pytest.approx(1.)
    with pytest.raises(ValueError, match="chronological"):
        predict_sensor_positions(np.zeros((2, 7)), [.0], np.zeros(3))
