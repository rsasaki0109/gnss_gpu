from experiments.analyze_wp29_moving_offset_widelane_shadow import candidate_offsets_from_artifact


def test_candidate_offsets_accepts_wp31_hypotheses() -> None:
    rows = candidate_offsets_from_artifact({"hypotheses": [{"seed_id": 52, "offset_ecef_m": [1, 2, 3], "audit_sub50cm_epochs": 39, "audit_median_error_m": 0.44}]})
    assert rows == [{"candidate_id": 52, "offset_ecef_m": [1, 2, 3], "audit_sub50cm_epochs": 39, "audit_rms_m": 0.44}]
