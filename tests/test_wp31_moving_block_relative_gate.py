from experiments.audit_wp31_moving_block_relative_gate import passing_relative_gate


def test_relative_gate_requires_every_frozen_condition() -> None:
    base = {"carrier_rms_cycles": 0.19, "ddpr_rms_m": 6.0, "road_p95_m": 0.8, "block_spread_m": 0.09}
    rows = [{"seed_id": 1, **base}]
    for seed, field, value in ((2, "carrier_rms_cycles", 0.21), (3, "ddpr_rms_m", 6.6), (4, "road_p95_m", 1.1), (5, "block_spread_m", 0.11)):
        rows.append({"seed_id": seed, **base, field: value})
    assert passing_relative_gate(rows, baseline_ddpr_rms_m=10.0) == [1]
