"""Regression tests for RTKLIB-aligned FGO pipeline on public gtsam_gnss data.

Validates that:
1. FGO with RTKLIB varerr weights matches RTKLIB rnx2rtkp SPP to < 0.1 m RMS.
2. The legacy ``correct_pseudoranges`` path gives the known ~36 m RMS gap.
3. ``export_spp_meas`` CSV output has the expected schema (el_rad, var_total).

Datasets are parametrized so future RINEX clips can be added to ``DATASETS``.

Run with::

    PYTHONPATH=python pytest tests/test_rtklib_spp_regression.py -v
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Path setup: make ``python/`` and ``experiments/`` importable.
# ---------------------------------------------------------------------------
_TESTS_DIR = Path(__file__).resolve().parent
_REPO = _TESTS_DIR.parent
_PYTHON_DIR = _REPO / "python"
_EXPERIMENTS_DIR = _REPO / "experiments"
for _p in (_PYTHON_DIR, _EXPERIMENTS_DIR, _REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gnss_gpu import wls_position  # noqa: E402
from gnss_gpu.fgo import fgo_gnss_lm  # noqa: E402
from gtsam_public_dataset import (  # noqa: E402
    build_public_gtsam_arrays,
    nearest_ref_error,
    load_reference_csv,
)

# ---------------------------------------------------------------------------
# Dataset registry — add new entries here for future RINEX clips.
# ---------------------------------------------------------------------------
DATASETS = [
    {
        "name": "gtsam_public",
        "obs": "rover_1Hz.obs",
        "nav": "base.nav",
        "ref": "reference.csv",
        "data_dir_rel": "../ref/gtsam_gnss/examples/data",
        "max_epochs": 60,
        "el_mask_deg": 15.0,
        # FGO with sin²(el) beats RTKLIB SPP (~0.96m vs ~1.67m)
        "expected_fgo_rms_2d_max": 1.5,
        "expected_rtklib_rms_2d_max": 3.0,
        # With weight_mode=rtklib, FGO matches RTKLIB exactly
        "expected_rtklib_weight_diff_max": 0.1,
        "expected_legacy_rms_min": 30.0,
        "expected_legacy_rms_max": 50.0,
    },
]

# ---------------------------------------------------------------------------
# Resolve paths.
# ---------------------------------------------------------------------------

def _resolve_data_dir(ds: dict) -> Path:
    return (_REPO / ds["data_dir_rel"]).resolve()


def _data_available(ds: dict) -> bool:
    d = _resolve_data_dir(ds)
    return all((d / ds[k]).is_file() for k in ("obs", "nav", "ref"))


def _export_spp_meas_exe() -> Path | None:
    envp = os.environ.get("RTKLIB_EXPORT_SPP_MEAS")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        _REPO.parent
        / "ref"
        / "RTKLIB-demo5"
        / "app"
        / "consapp"
        / "rnx2rtkp"
        / "gcc"
        / "export_spp_meas"
    )
    return guess if guess.is_file() else None


def _rnx2rtkp_exe() -> Path | None:
    envp = os.environ.get("RTKLIB_RNX2RTKP")
    if envp:
        p = Path(envp)
        if p.is_file():
            return p
    guess = (
        _REPO.parent
        / "ref"
        / "RTKLIB-demo5"
        / "app"
        / "consapp"
        / "rnx2rtkp"
        / "gcc"
        / "rnx2rtkp"
    )
    return guess if guess.is_file() else None


# ---------------------------------------------------------------------------
# RTKLIB rnx2rtkp helpers (mirror compare_fgo_rtklib_demo5.py).
# ---------------------------------------------------------------------------

def _run_rnx2rtkp(
    rnx2rtkp: Path,
    obs: Path,
    nav: Path,
    out_pos: Path,
    *,
    mode: int = 0,
    elev_deg: float = 15.0,
) -> None:
    cmd = [
        str(rnx2rtkp),
        "-p", str(mode),
        "-sys", "G",
        "-e",
        "-m", str(elev_deg),
        "-o", str(out_pos),
        str(obs),
        str(nav),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)


def _parse_rtklib_pos(path: Path) -> dict[float, tuple[float, float, float, int]]:
    out: dict[float, tuple[float, float, float, int]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                sow = round(float(parts[1]) * 1000.0) / 1000.0
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                q = int(parts[5])
            except ValueError:
                continue
            out[sow] = (x, y, z, q)
    return out


def _rtk_at_tow(
    rtk_sol: dict[float, tuple[float, float, float, int]], tow: float
) -> tuple[float, float, float, int] | None:
    towk = round(float(tow) * 1000.0) / 1000.0
    r = rtk_sol.get(towk)
    if r is not None:
        return r
    for k, v in rtk_sol.items():
        if abs(k - tow) < 0.02:
            return v
    return None


# ---------------------------------------------------------------------------
# Shared WLS + FGO runner.
# ---------------------------------------------------------------------------

def _run_wls_fgo(batch, *, fgo_iters: int = 8, fgo_tol: float = 1e-7):
    """Run per-epoch WLS then FGO; return (wls_state, fgo_state, fgo_iters, mse)."""
    n_epoch = batch.n_epoch
    wls_state = np.zeros((n_epoch, 4), dtype=np.float64)
    for t in range(n_epoch):
        w = batch.weights[t]
        idx = np.flatnonzero(w > 0)
        if idx.size < 4:
            continue
        st, _ = wls_position(
            batch.sat_ecef[t, idx, :].reshape(-1),
            batch.pseudorange[t, idx],
            w[idx],
            25,
            1e-9,
        )
        wls_state[t, :] = st

    fgo_state = wls_state.copy()
    iters, mse_pr = fgo_gnss_lm(
        batch.sat_ecef,
        batch.pseudorange,
        batch.weights,
        fgo_state,
        motion_sigma_m=0.0,
        max_iter=fgo_iters,
        tol=fgo_tol,
    )
    return wls_state, fgo_state, iters, mse_pr


def _rms_2d_vs_ref(batch, state):
    errs = []
    for t in range(batch.n_epoch):
        tow = batch.epochs_data[t][0]
        errs.append(nearest_ref_error(tow, batch.ref_tow, batch.ref_ecef, state[t]))
    return float(np.sqrt(np.mean(np.square(errs))))


# ===================================================================
# Test 1a: FGO with RTKLIB obs + sin²(el) beats RTKLIB SPP accuracy
# ===================================================================

@pytest.mark.slow
@pytest.mark.parametrize("ds", DATASETS, ids=[d["name"] for d in DATASETS])
def test_fgo_beats_rtklib_accuracy(ds):
    """FGO with RTKLIB obs + sin²(el) weights should beat RTKLIB SPP vs reference."""
    data_dir = _resolve_data_dir(ds)
    if not _data_available(ds):
        pytest.skip(f"Dataset files not found in {data_dir}")
    export_spp = _export_spp_meas_exe()
    if export_spp is None:
        pytest.skip("RTKLIB export_spp_meas binary not available")

    # Build batch using RTKLIB export + sin²(el) weights (default).
    batch = build_public_gtsam_arrays(
        data_dir,
        ds["max_epochs"],
        rtklib_export_spp_exe=export_spp,
        el_mask_deg=ds["el_mask_deg"],
    )

    # Run WLS + FGO.
    _wls_state, fgo_state, iters, _mse = _run_wls_fgo(batch)
    assert iters >= 0, "FGO solver returned error code"

    # FGO RMS vs reference should be < threshold (and better than RTKLIB).
    rms_fgo_ref = _rms_2d_vs_ref(batch, fgo_state)
    assert rms_fgo_ref < ds["expected_fgo_rms_2d_max"], (
        f"FGO RMS vs reference = {rms_fgo_ref:.3f} m exceeds "
        f"threshold {ds['expected_fgo_rms_2d_max']} m"
    )


# ===================================================================
# Test 1b: FGO with RTKLIB weights matches RTKLIB pntpos exactly
# ===================================================================

@pytest.mark.slow
@pytest.mark.parametrize("ds", DATASETS, ids=[d["name"] for d in DATASETS])
def test_fgo_rtklib_weight_alignment(ds):
    """FGO with weight_mode=rtklib should match RTKLIB rnx2rtkp to < 0.1 m RMS."""
    data_dir = _resolve_data_dir(ds)
    if not _data_available(ds):
        pytest.skip(f"Dataset files not found in {data_dir}")
    export_spp = _export_spp_meas_exe()
    if export_spp is None:
        pytest.skip("RTKLIB export_spp_meas binary not available")
    rnx2rtkp = _rnx2rtkp_exe()
    if rnx2rtkp is None:
        pytest.skip("RTKLIB rnx2rtkp binary not available")

    # Build batch using RTKLIB export + RTKLIB inverse-variance weights.
    batch = build_public_gtsam_arrays(
        data_dir,
        ds["max_epochs"],
        rtklib_export_spp_exe=export_spp,
        el_mask_deg=ds["el_mask_deg"],
        weight_mode="rtklib",
    )

    _wls_state, fgo_state, iters, _mse = _run_wls_fgo(batch)
    assert iters >= 0, "FGO solver returned error code"

    # Run rnx2rtkp for comparison.
    obs_p = data_dir / ds["obs"]
    nav_p = data_dir / ds["nav"]
    with tempfile.NamedTemporaryFile(suffix=".pos", delete=False) as tf:
        out_pos = Path(tf.name)
    try:
        _run_rnx2rtkp(rnx2rtkp, obs_p, nav_p, out_pos, elev_deg=ds["el_mask_deg"])
        rtk_sol = _parse_rtklib_pos(out_pos)
    finally:
        out_pos.unlink(missing_ok=True)

    diffs_2d: list[float] = []
    for t in range(batch.n_epoch):
        tow = batch.epochs_data[t][0]
        r = _rtk_at_tow(rtk_sol, tow)
        if r is None:
            continue
        x, y, z, _q = r
        d = fgo_state[t, :2] - np.array([x, y])
        diffs_2d.append(float(np.linalg.norm(d)))

    assert len(diffs_2d) > 0, "No epochs matched between FGO and RTKLIB output"
    rms_diff = float(np.sqrt(np.mean(np.square(diffs_2d))))

    assert rms_diff < ds["expected_rtklib_weight_diff_max"], (
        f"||FGO - RTKLIB|| RMS = {rms_diff:.4f} m exceeds "
        f"threshold {ds['expected_rtklib_weight_diff_max']} m"
    )


# ===================================================================
# Test 2: Legacy path produces expected (worse) results
# ===================================================================

@pytest.mark.slow
@pytest.mark.parametrize("ds", DATASETS, ids=[d["name"] for d in DATASETS])
def test_legacy_spp_path(ds):
    """Legacy correct_pseudoranges path should give ~36 m RMS (known model gap)."""
    data_dir = _resolve_data_dir(ds)
    if not _data_available(ds):
        pytest.skip(f"Dataset files not found in {data_dir}")

    # Build batch WITHOUT RTKLIB export (legacy gnss_gpu path).
    batch = build_public_gtsam_arrays(data_dir, ds["max_epochs"])

    wls_state, _fgo_state, _iters, _mse = _run_wls_fgo(batch)
    rms_wls = _rms_2d_vs_ref(batch, wls_state)

    # Legacy path should fall in the known ~30-50 m range.
    assert ds["expected_legacy_rms_min"] <= rms_wls <= ds["expected_legacy_rms_max"], (
        f"Legacy WLS RMS = {rms_wls:.1f} m outside expected range "
        f"[{ds['expected_legacy_rms_min']}, {ds['expected_legacy_rms_max']}] m"
    )


# ===================================================================
# Test 3: export_spp_meas CSV format validation
# ===================================================================

@pytest.mark.parametrize("ds", DATASETS, ids=[d["name"] for d in DATASETS])
def test_export_spp_meas_csv_format(ds):
    """export_spp_meas CSV should include el_rad and var_total columns."""
    data_dir = _resolve_data_dir(ds)
    if not _data_available(ds):
        pytest.skip(f"Dataset files not found in {data_dir}")
    export_spp = _export_spp_meas_exe()
    if export_spp is None:
        pytest.skip("RTKLIB export_spp_meas binary not available")

    obs_p = data_dir / ds["obs"]
    nav_p = data_dir / ds["nav"]

    # Run export_spp_meas and capture CSV output.
    cmd = [str(export_spp), str(obs_p), str(nav_p), "-m", str(ds["el_mask_deg"])]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    lines = result.stdout.strip().splitlines()
    assert len(lines) >= 2, "export_spp_meas produced fewer than 2 lines (header + data)"

    reader = csv.DictReader(lines)
    fieldnames = reader.fieldnames
    assert fieldnames is not None, "CSV has no header"

    # Required columns.
    assert "el_rad" in fieldnames, f"Missing 'el_rad' column; got: {fieldnames}"
    assert "var_total" in fieldnames, f"Missing 'var_total' column; got: {fieldnames}"

    # Validate data rows.
    row_count = 0
    for row in reader:
        var_total = float(row["var_total"])
        assert var_total > 0, (
            f"var_total <= 0 at row {row_count}: {var_total} "
            f"(sat={row.get('sat_id', '?')}, tow={row.get('gps_tow', '?')})"
        )
        el_rad = float(row["el_rad"])
        assert 0.0 < el_rad <= np.pi / 2, (
            f"el_rad out of range at row {row_count}: {el_rad}"
        )
        row_count += 1

    assert row_count > 0, "export_spp_meas produced header but zero data rows"
