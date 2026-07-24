from __future__ import annotations

import numpy as np

from gnss_gpu.static_shared_integer import (
    SharedIntegerConfig,
    _CarrierRow,
    _build_arcs,
    _satellite_potential_constraints,
)
from gnss_gpu.stop_segment_static import _dd_expected_and_jacobian_m


def _row(
    epoch: int,
    ref_id: str,
    sat_id: str,
    ref_position: np.ndarray,
    sat_position: np.ndarray,
    rover_position: np.ndarray,
    integer: int,
) -> _CarrierRow:
    wavelength = 0.190293673
    expected, _jac = _dd_expected_and_jacobian_m(
        rover_position, sat_position, ref_position, 20_000_000.0, 20_000_000.0
    )
    return _CarrierRow(
        epoch=epoch,
        key=(ref_id, sat_id, int(round(wavelength * 1e9))),
        observed_cycles=expected / wavelength + integer,
        sat_k=sat_position,
        sat_ref=ref_position,
        base_k=20_000_000.0,
        base_ref=20_000_000.0,
        wavelength_m=wavelength,
        weight=1.0,
    )


def test_exact_pair_arc_keeps_one_integer_over_epochs() -> None:
    rover = np.array([-3_900_000.0, 3_300_000.0, 3_700_000.0])
    ref = np.array([20_000_000.0, 10_000_000.0, 15_000_000.0])
    sat = np.array([18_000_000.0, -12_000_000.0, 17_000_000.0])
    rows = [_row(epoch, "G01@L1", "G02@L1", ref, sat, rover, 17) for epoch in range(5)]

    arcs = _build_arcs(rows, rover, SharedIntegerConfig(min_arc_samples=3))

    assert len(arcs) == 1
    assert len(arcs[0]) == 5


def test_satellite_potentials_preserve_integers_across_pivot_change() -> None:
    rover = np.array([-3_900_000.0, 3_300_000.0, 3_700_000.0])
    satellites = {
        "G01@L1": np.array([20_000_000.0, 10_000_000.0, 15_000_000.0]),
        "G02@L1": np.array([18_000_000.0, -12_000_000.0, 17_000_000.0]),
        "G03@L1": np.array([-13_000_000.0, 19_000_000.0, 16_000_000.0]),
    }
    potentials = {"G01@L1": 0, "G02@L1": 17, "G03@L1": -8}
    pairs = [("G01@L1", "G02@L1"), ("G01@L1", "G03@L1"), ("G02@L1", "G03@L1")]
    rows = [
        _row(
            epoch,
            ref_id,
            sat_id,
            satellites[ref_id],
            satellites[sat_id],
            rover,
            potentials[sat_id] - potentials[ref_id],
        )
        for epoch, (ref_id, sat_id) in enumerate(pairs)
    ]

    constraints, integers, n_potentials = _satellite_potential_constraints(rover, rows)

    assert n_potentials == 2
    assert [arc[0] for arc in constraints] == rows
    assert integers == [17, -8, -25]
