"""PPC runs are rewritten into the UrbanNav layout with a z-down, rad/s IMU."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))

from materialize_ppc_urbannav_layout import convert_imu_row, materialize_run  # noqa: E402

from gnss_gpu.imu import load_imu_csv  # noqa: E402


def test_convert_imu_row_rotates_about_x_and_converts_gyro():
    out = convert_imu_row(["100.0", "2324", "0.1", "0.2", "9.8", "1.0", "2.0", "-3.0"])
    assert out[:2] == ["100.000", "2324"]
    assert [float(v) for v in out[2:5]] == [0.1, -0.2, -9.8]
    gx, gy, gz = (float(v) for v in out[5:8])
    assert math.isclose(gx, math.radians(1.0), rel_tol=1e-6)
    assert math.isclose(gy, -math.radians(2.0), rel_tol=1e-6)
    assert math.isclose(gz, math.radians(3.0), rel_tol=1e-6)
    assert out[8] == ""


def test_materialize_run_writes_urbannav_files(tmp_path):
    src = tmp_path / "ppc"
    src.mkdir()
    for name in ("rover.obs", "base.obs", "base.nav", "reference.csv"):
        (src / name).write_text(name, encoding="utf-8")
    (src / "imu.csv").write_text(
        "tow,week,ax,ay,az,gx,gy,gz\n100.0,2324,0,0,9.8,0,0,10\n100.01,2324,0,0,9.8,0,0,10\n",
        encoding="utf-8",
    )
    dst = tmp_path / "out"
    materialize_run(src, dst)

    for name in ("rover_trimble.obs", "base_trimble.obs", "base.nav", "reference.csv"):
        assert (dst / name).exists()
    imu = load_imu_csv(dst / "imu.csv")
    assert math.isclose(imu["gyro"][0, 2], -math.radians(10.0), rel_tol=1e-6)
    assert math.isnan(imu["wheel_vel"][0])
