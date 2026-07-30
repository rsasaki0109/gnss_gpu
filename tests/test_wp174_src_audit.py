from experiments.prepare_wp174_src_audit import normalize, normalize_satellite


def test_normalize_maps_src_decision_fields_fail_closed() -> None:
    row = {
        "tow": "1",
        "src_par_best_sub50cm": "1",
        "lambda_src_par_shadow_subset_size": "12",
        "lambda_src_par_shadow_ratio": "3",
        "lambda_src_par_shadow_bsr": "0.9995",
        "lambda_src_par_shadow_second_position_delta_m": "0.02",
        "lambda_src_par_shadow_best_correction_x": "0.1",
        "lambda_src_par_shadow_best_correction_y": "0.2",
        "lambda_src_par_shadow_best_correction_z": "0.3",
    }
    output = normalize(row)
    assert output["pair_count"] == "12"
    assert output["shadow_best_sub50cm"] == "1"
    assert output["lambda_shadow_bsr_qscale16"] == "0.9995"
    assert output["lambda_shadow_bsr_qscale8"] == ""
    assert output["lambda_shadow_best_correction_z"] == "0.3"


def test_normalize_maps_satellite_par_fields_fail_closed() -> None:
    output = normalize_satellite(
        {
            "lambda_satellite_par_shadow_attempted": "1",
            "lambda_satellite_par_shadow_solved": "1",
            "lambda_satellite_par_shadow_subset_size": "14",
            "lambda_satellite_par_shadow_ratio": "12",
            "lambda_satellite_par_shadow_bsr": "0.9999",
            "lambda_satellite_par_shadow_second_position_delta_m": "0.02",
            "lambda_satellite_par_shadow_best_ecef_x": "1",
            "lambda_satellite_par_shadow_best_ecef_y": "2",
            "lambda_satellite_par_shadow_best_ecef_z": "3",
            "lambda_satellite_par_shadow_best_correction_x": "0.01",
            "lambda_satellite_par_shadow_best_correction_y": "0.02",
            "lambda_satellite_par_shadow_best_correction_z": "0.03",
            "satellite_par_best_sub50cm": "1",
            "satellite_par_best_error_m": "0.1",
        }
    )
    assert output["pair_count"] == "14"
    assert output["lambda_shadow_solved"] == "1"
    assert output["lambda_shadow_bsr_qscale16"] == "0.9999"
    assert output["shadow_best_sub50cm"] == "1"
