from __future__ import annotations

import numpy as np
import pytest

from experiments.apply_wp42_moving_block_offset import (
    apply_linear_bootstrap_profile,
    apply_offset,
)
from experiments.promote_wp42_moving_temporal_trifrequency_ddpr import (
    _candidate_hash,
    _sanitized_candidates,
    promote,
)


def _documents(offset: list[float]) -> tuple[dict, dict]:
    source = {
        "segment": [10, 20],
        "hypotheses": [
            {
                "seed_id": 3,
                "offset_ecef_m": offset,
                "integer_arcs": 8,
                "carrier_rows": 80,
                "carrier_rms_cycles": 0.2,
                "block_spread_m": 0.1,
                "block_offsets_ecef_m": [
                    offset,
                    offset,
                    offset,
                    offset,
                ],
                "audit_median_error_m": 999.0,
            }
        ],
    }
    selector = {
        "production_input_truth": False,
        "segment": [10, 20],
        "selected_candidate_id": 3,
        "reason": "unique_moving_temporal_trifrequency_ddpr_rank_consensus",
        "winner": {
            "candidate_id": 3,
            "supply_pass": True,
            "family_ranks": {"primary": 1, "secondary": 1, "tertiary": 1},
            "rank_sum": 3,
        },
        "runner_margin": 0.5,
        "family_rank_limit": 1,
    }
    selector["candidate_source_sha256"] = _candidate_hash(_sanitized_candidates(source))
    return selector, source


def test_promote_ignores_audit_and_accepts_small_offset() -> None:
    selector, source = _documents([0.1, -0.1, 0.05])
    result = promote(selector, source)
    assert result["production_promoted"] is True
    assert result["selected_candidate_id"] == 3
    assert result["offset_norm_m"] < 0.5


def test_promote_rejects_large_boundary_jump() -> None:
    selector, source = _documents([1.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="boundary-continuity"):
        promote(selector, source)


def test_apply_offset_changes_only_half_open_segment() -> None:
    positions = np.zeros((5, 3))
    output = apply_offset(positions, start=1, end=4, offset=np.asarray([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(output[[0, 4]], 0.0)
    np.testing.assert_allclose(output[1:4], [[1.0, 2.0, 3.0]] * 3)


def test_linear_bootstrap_profile_interpolates_between_block_centers() -> None:
    positions = np.zeros((8, 3))
    output = apply_linear_bootstrap_profile(
        positions,
        start=0,
        end=8,
        block_offsets=np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    )
    assert output[0, 0] == 0.0
    assert output[-1, 0] == 2.0
    assert np.all(np.diff(output[:, 0]) >= 0.0)
