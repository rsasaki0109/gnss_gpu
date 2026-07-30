from experiments.analyze_wp174_quality_diverse_par import analyze


def _row(tow: str, satellite_pass: str, safe: str) -> dict[str, str]:
    return {
        "tow": tow,
        "lambda_satellite_par_shadow_ffrt_passed": satellite_pass,
        "lambda_satellite_par_shadow_best_ecef_x": "1.1",
        "lambda_satellite_par_shadow_best_ecef_y": "2",
        "lambda_satellite_par_shadow_best_ecef_z": "3",
        "safe_fix_shadow_declared_fixed": safe,
        "lambda_shadow_best_ecef_x": "1.1",
        "lambda_shadow_best_ecef_y": "2",
        "lambda_shadow_best_ecef_z": "3",
        "lambda_shadow_runtime_ms": "4",
    }


def test_quality_diverse_audit_counts_additional_safe_candidate() -> None:
    baseline = [_row("10.0", "0", "0")]
    diverse = [_row("10.0", "1", "1")]
    reference = [
        {
            "GPS TOW (s)": "10.0",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        }
    ]

    result = analyze(baseline, diverse, reference, "tokyo")

    assert result["additional_satellite_par_ffrt_epochs"] == 1
    assert result["quality_diverse_ranking"]["safe_fix_false_epochs"] == 0
    assert result["pass_runtime_p95_100ms"]
