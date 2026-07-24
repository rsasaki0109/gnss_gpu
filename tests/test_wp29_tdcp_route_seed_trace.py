import numpy as np
import pytest

from experiments.build_wp29_tdcp_route_seed_trace import (
    expand_axis_position_seeds,
    integrate_static_anchored_template,
    retime_and_close_template,
)


def test_integrate_static_anchored_template_keeps_static_stop_fixed() -> None:
    displacement = np.zeros((8, 3))
    displacement[1:, 0] = 1.0

    positions = integrate_static_anchored_template(
        displacement,
        np.array([10.0, 0.0, 0.0]),
        start=0,
        static_start=2,
        static_end=5,
        end=7,
    )

    np.testing.assert_allclose(positions[:, 0], [8, 9, 10, 10, 10, 11, 12, 13])


def test_retime_and_close_template_uses_doppler_progress() -> None:
    target = np.array([[0.0, 0, 0], [1.0, 0, 0], [4.0, 0, 0]])
    template = np.array([[2.0, 0, 0], [4.0, 0, 0], [6.0, 0, 0]])

    seed, metrics = retime_and_close_template(
        target,
        np.array([1.0, 3.0]),
        template,
        max_endpoint_closure_m=3.0,
        max_arc_relative_error=0.01,
    )

    np.testing.assert_allclose(seed, target)
    assert metrics["arc_relative_error"] == pytest.approx(0.0)


def test_retime_and_close_template_fails_closed_on_arc() -> None:
    with pytest.raises(RuntimeError, match="Doppler arc"):
        retime_and_close_template(
            np.array([[0.0, 0, 0], [1.0, 0, 0]]),
            np.array([1.0]),
            np.array([[0.0, 0, 0], [2.0, 0, 0]]),
            max_endpoint_closure_m=2.0,
            max_arc_relative_error=0.1,
        )


def test_expand_axis_position_seeds_is_center_first_and_deterministic() -> None:
    seeds = expand_axis_position_seeds(np.array([1.0, 2.0, 3.0]), (2.0, 4.0))

    assert len(seeds) == 13
    np.testing.assert_allclose(seeds[0][0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(seeds[1][0], [3.0, 2.0, 3.0])
    np.testing.assert_allclose(seeds[2][0], [-1.0, 2.0, 3.0])
    assert seeds[1][1] == -2.0
