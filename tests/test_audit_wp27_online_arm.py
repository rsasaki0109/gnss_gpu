import csv
import json

import pytest

from experiments.audit_wp27_online_arm import main


FIELDS = [
    "tow",
    "ecef_x",
    "ecef_y",
    "ecef_z",
    "fix",
    "gamma_fixed",
    "map_assignment_id",
    "gamma",
    "integrity_anchor_available",
    "integrity_tdcp_available",
    "integrity_map_error_m",
]


def _write_diagnostics(path, *, ecef_x="1.0"):
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "tow": "10.0",
                "ecef_x": ecef_x,
                "ecef_y": "2.0",
                "ecef_z": "3.0",
                "fix": "0",
                "gamma_fixed": "0",
                "map_assignment_id": "candidate",
                "gamma": "0.8",
                "integrity_anchor_available": "1",
                "integrity_tdcp_available": "1",
                "integrity_map_error_m": "0.4",
            }
        )


def _arguments(control, integrity, control_trajectory, integrity_trajectory, output):
    return [
        "--control-diagnostics",
        str(control),
        "--integrity-diagnostics",
        str(integrity),
        "--control-trajectory",
        str(control_trajectory),
        "--integrity-trajectory",
        str(integrity_trajectory),
        "--out-summary",
        str(output),
    ]


def test_online_arm_audit_accepts_neutral_diagnostic(tmp_path):
    control = tmp_path / "control.csv"
    integrity = tmp_path / "integrity.csv"
    control_trajectory = tmp_path / "control_pos.csv"
    integrity_trajectory = tmp_path / "integrity_pos.csv"
    output = tmp_path / "audit.json"
    _write_diagnostics(control)
    _write_diagnostics(integrity)
    control_trajectory.write_text("same trajectory\n")
    integrity_trajectory.write_text("same trajectory\n")

    main(_arguments(control, integrity, control_trajectory, integrity_trajectory, output))

    summary = json.loads(output.read_text())
    assert summary["operational_mismatches"] == 0
    assert summary["trajectory_bit_identical"] is True
    assert summary["integrity_map_sub50cm_epochs"] == 1


def test_online_arm_audit_rejects_operational_change(tmp_path):
    control = tmp_path / "control.csv"
    integrity = tmp_path / "integrity.csv"
    control_trajectory = tmp_path / "control_pos.csv"
    integrity_trajectory = tmp_path / "integrity_pos.csv"
    _write_diagnostics(control)
    _write_diagnostics(integrity, ecef_x="1.1")
    control_trajectory.write_text("control\n")
    integrity_trajectory.write_text("changed\n")

    with pytest.raises(RuntimeError, match="changed operational output"):
        main(
            _arguments(
                control,
                integrity,
                control_trajectory,
                integrity_trajectory,
                tmp_path / "audit.json",
            )
        )
