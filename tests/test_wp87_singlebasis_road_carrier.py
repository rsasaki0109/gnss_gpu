from __future__ import annotations

import numpy as np
from pyproj import Transformer
from shapely.geometry import LineString
from shapely.strtree import STRtree

from experiments.select_wp87_singlebasis_road_carrier import select_single_basis
from experiments.select_wp76_affine_multibasis_road_carrier import profile_scales


def _source() -> dict:
    hypotheses = []
    for candidate_id, offset, carrier_rms, cppr in (
        (1, [0.0, 0.0, 0.0], 0.1, (50, 1)),
        (2, [0.0, 2.0, 0.0], 0.2, (45, 2)),
        (3, [0.0, 4.0, 0.0], 0.3, (40, 2)),
        (4, [0.0, 6.0, 0.0], 0.4, (40, 2)),
        (5, [0.0, 8.0, 0.0], 0.5, (40, 2)),
    ):
        hypotheses.append(
            {
                "seed_id": candidate_id,
                "offset_ecef_m": offset,
                "block_offsets_ecef_m": [offset, offset],
                "block_spread_m": 0.0,
                "carrier_rms_cycles": carrier_rms,
                "cp_pr_consistency": {
                    "checked_pairs": cppr[0],
                    "bad_pairs": cppr[1],
                    "rms_innovation_m": float(candidate_id),
                    "median_abs_innovation_m": float(candidate_id),
                    "p95_abs_innovation_m": float(candidate_id),
                },
            }
        )
    return {
        "production_input_truth": False,
        "segment": [0, 2],
        "offset_model": {"mode": "right_boundary_affine_zero", "boundary_epoch": 2},
        "hypotheses": hypotheses,
    }


def test_single_basis_selector_requires_three_family_top_ranks() -> None:
    source = _source()
    route = np.asarray([[6378137.0, 0.0, 0.0], [6378137.0, 0.0, 0.0]])
    road = STRtree([LineString([(-10.0, 0.0), (10.0, 0.0)])])
    transformer = Transformer.from_crs("EPSG:4978", "EPSG:3857", always_xy=True)

    result = select_single_basis(
        source,
        route,
        road,
        transformer,
        scales=np.asarray([1.0, 0.5]),
        road_lower_m=0.0,
        road_upper_m=100.0,
    )

    assert result["mode_count"] == 5
    assert result["winner"]["candidate_id"] == 1
    assert set(result["winner"]["family_ranks"]) == {"road_band", "carrier_rms", "cppr"}


def test_constant_profile_uses_unit_scales() -> None:
    assert np.array_equal(
        profile_scales(10, 13, {"mode": "constant"}), np.ones(3)
    )
