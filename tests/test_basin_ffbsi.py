from __future__ import annotations

import numpy as np

from gnss_gpu.ambiguity_basin_pf import AmbiguityBasinParticleFilter, BasinKalmanState
from gnss_gpu.basin_ffbsi import FixedLagBasinFFBSi


def _state(x: float) -> BasinKalmanState:
    return BasinKalmanState.from_position(np.array([x, 0.0, 0.0]), np.eye(3))


def _key(satellite: str) -> tuple[tuple[str, str, int], int]:
    return (("G01", satellite, 190293673), 0)


def test_ffbsi_traces_future_mass_to_fixed_lag_ancestor() -> None:
    pf = AmbiguityBasinParticleFilter(dedup_position_radius_m=0.1)
    first = {_key("G02"): 1}
    second = {_key("G03"): 2}
    pf.spawn([first, second], [_state(0.0), _state(10.0)])
    pf.basins[0].log_weight = np.log(0.5)
    pf.basins[1].log_weight = np.log(0.5)
    smoother = FixedLagBasinFFBSi(lag_epochs=1, backward_samples=256)
    smoother.capture(pf, 1.0)

    parent_ids = {tuple(basin.assignment): basin.basin_id for basin in pf.basins}
    pf.predict(1.0)
    pf.replace_with_transitions(
        [first, second],
        [_state(1.0), _state(11.0)],
        [-8.0, 0.0],
        [parent_ids[tuple(first.items())], parent_ids[tuple(second.items())]],
    )
    smoother.capture(pf, 2.0)
    estimate = smoother.estimate(seed=7)
    assert estimate is not None
    assert estimate.target_tow_s == 1.0
    assert estimate.assignment_probability > 0.99
    assert dict(estimate.map_assignment) == second
    assert estimate.position_ecef_m[0] > 9.9


def test_ffbsi_waits_for_requested_lag() -> None:
    pf = AmbiguityBasinParticleFilter()
    pf.spawn([{}], [_state(0.0)])
    smoother = FixedLagBasinFFBSi(lag_epochs=2)
    smoother.capture(pf, 1.0)
    assert smoother.estimate() is None
