import json

from experiments.select_wp31_moving_block_path_posterior import select_path


def _pool(path, segment, offsets_and_costs):
    rows = []
    for item in offsets_and_costs:
        offset, carrier, ddpr, *signature = item
        rows.append({"offset_ecef_m": offset, "integer_arcs": 4, "retained_carrier_rows": 8,
                     "carrier_rms_cycles": carrier, "ddpr_rms_m": ddpr,
                     "integer_signature": signature[0] if signature else {}})
    path.write_text(json.dumps({"schema": "wp31_moving_block_truth_free_local_pool_v1",
                                "production_input_truth": False, "segment": segment,
                                "candidates": rows}), encoding="utf-8")


def test_anchor_path_rejects_stronger_unreachable_measurement_mode(tmp_path) -> None:
    one = tmp_path / "one.json"; two = tmp_path / "two.json"
    _pool(one, [10, 20], [([0.5, 0, 0], .20, 6), ([20, 0, 0], .10, 2)])
    _pool(two, [20, 30], [([1.0, 0, 0], .20, 6), ([20, 0, 0], .10, 2)])
    result = select_path({"anchor_epoch": 9, "anchor_offset_ecef_m": [0, 0, 0],
                          "config": {"drift_radius_per_epoch": .1, "transition_base_m": .5,
                                     "min_gamma": .5},
                          "blocks": [{"pool_path": str(one), "baseline_ddpr_rms_m": 10},
                                     {"pool_path": str(two), "baseline_ddpr_rms_m": 10}]})
    assert result["selected_path_offsets_ecef_m"] == [[0.5, 0.0, 0.0], [1.0, 0.0, 0.0]]


def test_ambiguous_path_fails_confidence_gate(tmp_path) -> None:
    pool = tmp_path / "pool.json"
    _pool(pool, [1, 2], [([0.5, 0, 0], .2, 5), ([-0.5, 0, 0], .2, 5)])
    result = select_path({"anchor_epoch": 0, "anchor_offset_ecef_m": [0, 0, 0],
                          "blocks": [{"pool_path": str(pool), "baseline_ddpr_rms_m": 10}]})
    assert result["declaration_eligible"] is False
    assert result["selection_reason"] == "posterior_not_confident"


def test_integer_lineage_rejects_nearby_cycle_changed_mode(tmp_path) -> None:
    one = tmp_path / "one.json"; two = tmp_path / "two.json"
    good = {"G01|G02|1": 7, "G01|G03|1": 4}
    changed = {"G01|G02|1": 8, "G01|G03|1": 4}
    _pool(one, [10, 20], [([0.5, 0, 0], .2, 5, good)])
    _pool(two, [20, 30], [([1.0, 0, 0], .2, 5, good), ([0.6, 0, 0], .1, 2, changed)])
    result = select_path({"anchor_epoch": 9, "anchor_offset_ecef_m": [0, 0, 0],
                          "config": {"min_shared_integer_arcs": 2,
                                     "max_integer_disagreements": 0,
                                     "integer_tolerance_cycles": 0,
                                     "drift_radius_per_epoch": .1},
                          "blocks": [{"pool_path": str(one), "baseline_ddpr_rms_m": 10},
                                     {"pool_path": str(two), "baseline_ddpr_rms_m": 10}]})
    assert result["selected_path_offsets_ecef_m"][-1] == [1.0, 0.0, 0.0]
