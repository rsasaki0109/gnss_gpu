from experiments.audit_wp31_moving_block_raw_temporal_pool import pair_pools


def _row(offset, carrier=0.2, ddpr=5.0):
    return {
        "offset_ecef_m": offset, "integer_arcs": 4, "retained_carrier_rows": 8,
        "carrier_rms_cycles": carrier, "ddpr_rms_m": ddpr,
        "proposal_score": carrier + 0.002 * ddpr,
        "map_translation_xyh_m": offset,
    }


def test_pair_pools_requires_mutual_nearest_and_measurement_gates() -> None:
    support = {"segment": [0, 10], "candidates": [_row([1, 2, 3]), _row([20, 0, 0], carrier=0.4)]}
    primary = {"segment": [10, 20], "candidates": [_row([1.2, 2, 3]), _row([20.1, 0, 0])]}
    result = pair_pools(
        support, primary, support_baseline_ddpr_m=10, primary_baseline_ddpr_m=10,
        max_seeds=8,
    )
    assert result["mutual_temporal_pairs"] == 1
    assert result["selected_seed_count"] == 1
    assert result["seeds"][0]["offset_ecef_m"] == [1.2, 2, 3]
