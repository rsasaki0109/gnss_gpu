from experiments.audit_wp31_moving_block_temporal_consensus import temporal_support_pairs


def test_temporal_support_requires_metrics_and_half_meter_match() -> None:
    primary = [{"seed_id": 52, "offset_ecef_m": [1.0, 2.0, 3.0]}]
    base = {"integer_arcs": 4, "carrier_rows": 8, "ddpr_rows": 40, "carrier_rms_cycles": 0.5}
    support = [
        {"seed_id": 1, "offset_ecef_m": [1.3, 2.0, 3.0], **base},
        {"seed_id": 2, "offset_ecef_m": [1.6, 2.0, 3.0], **base},
        {"seed_id": 3, "offset_ecef_m": [1.1, 2.0, 3.0], **base, "carrier_rows": 7},
    ]
    assert temporal_support_pairs(primary, support, [52]) == [
        {"primary_seed_id": 52, "support_seed_id": 1, "offset_distance_m": 0.30000000000000004}
    ]
