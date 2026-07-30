from __future__ import annotations

from experiments.analyze_wp174_safe_route import analyze


def test_route_audit_checks_safe_fix_and_output_targets() -> None:
    debug = [
        {
            "tow": "10.0",
            "safe_fix_shadow_declared_fixed": "1",
            "lambda_shadow_best_ecef_x": "1.1",
            "lambda_shadow_best_ecef_y": "2",
            "lambda_shadow_best_ecef_z": "3",
            "lambda_shadow_runtime_ms": "2",
            "safe_fix_shadow_change_point_acquisition": "1",
            "safe_fix_shadow_strong_acquisition": "0",
        },
        {
            "tow": "10.2",
            "safe_fix_shadow_declared_fixed": "0",
            "lambda_shadow_runtime_ms": "3",
        },
    ]
    positions = {
        10.0: {"ecef_x": 1.1, "ecef_y": 2.0, "ecef_z": 3.0},
        10.2: {"ecef_x": 1.2, "ecef_y": 2.0, "ecef_z": 3.0},
    }
    reference = [
        {
            "GPS TOW (s)": "10.0",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        },
        {
            "GPS TOW (s)": "10.2",
            "ECEF X (m)": "1",
            "ECEF Y (m)": "2",
            "ECEF Z (m)": "3",
        },
    ]

    result = analyze(debug, positions, reference, "tokyo", "abc")

    assert result["safe_fix_rate"] == 0.5
    assert result["safe_fix_false_epochs"] == 0
    assert result["pass_safe_fix_formal_target"]
    assert result["pass_safe_fix_stretch_target"]
    assert result["pass_tokyo_output_sub50cm_46p5112"]
    assert result["pass_runtime_p95_100ms"]
