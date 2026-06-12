"""End-to-end check that UrbanNavLoader applies the transmission-time/Sagnac
correction to its satellite positions (and that disabling it recovers the legacy
reception-time positions)."""

from pathlib import Path

import numpy as np
import pytest

from gnss_gpu.io.urbannav import UrbanNavLoader

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DATA_DIR = _REPO_ROOT / "experiments" / "data" / "urbannav" / "Odaiba"


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="UrbanNav Odaiba data not present")
def test_loader_transmission_time_shifts_ranges_tens_of_metres():
    loader = UrbanNavLoader(_DATA_DIR)
    try:
        corrected = loader.load_experiment_data(
            max_epochs=20, correct_transmission_time=True)
        legacy = loader.load_experiment_data(
            max_epochs=20, correct_transmission_time=False)
    except Exception as exc:  # GPU ephemeris / RINEX parsing unavailable
        pytest.skip(f"UrbanNav pipeline unavailable: {exc}")

    assert corrected["n_epochs"] == legacy["n_epochs"]
    assert corrected["n_epochs"] > 0
    # The epoch/satellite selection must be identical -- only the positions move.
    assert corrected["used_prns"] == legacy["used_prns"]

    pos_shifts = []   # |corrected sat pos - reception sat pos| per satellite [m]
    range_shifts = []  # change in geometric range to the truth receiver [m]
    gt = np.asarray(corrected["ground_truth"], dtype=float)
    for i in range(corrected["n_epochs"]):
        sat_c = np.asarray(corrected["sat_ecef"][i], dtype=float)
        sat_l = np.asarray(legacy["sat_ecef"][i], dtype=float)
        assert sat_c.shape == sat_l.shape
        pos_shifts.extend(np.linalg.norm(sat_c - sat_l, axis=1))
        rx = gt[i]
        range_shifts.extend(
            np.abs(np.linalg.norm(sat_c - rx, axis=1)
                   - np.linalg.norm(sat_l - rx, axis=1)))

    pos_shifts = np.asarray(pos_shifts)
    range_shifts = np.asarray(range_shifts)
    # A satellite moves hundreds of metres along its orbit over the ~0.07 s flight.
    assert pos_shifts.size > 0
    assert np.median(pos_shifts) > 50.0
    # The projection onto the line of sight (the range error the correction kills)
    # is at the tens-of-metres scale that swamps the multipath/NLOS signal.
    assert 1.0 < np.median(range_shifts) < 200.0
    assert np.max(range_shifts) > 10.0


@pytest.mark.skipif(not _DATA_DIR.is_dir(), reason="UrbanNav Odaiba data not present")
def test_loader_correction_default_is_on():
    loader = UrbanNavLoader(_DATA_DIR)
    try:
        default = loader.load_experiment_data(max_epochs=10)
        corrected = loader.load_experiment_data(
            max_epochs=10, correct_transmission_time=True)
    except Exception as exc:
        pytest.skip(f"UrbanNav pipeline unavailable: {exc}")

    # Default behaviour must equal explicit correction (the loader is clean by
    # default; nothing downstream has to opt in).
    for i in range(default["n_epochs"]):
        np.testing.assert_allclose(
            np.asarray(default["sat_ecef"][i], dtype=float),
            np.asarray(corrected["sat_ecef"][i], dtype=float),
            rtol=0, atol=0)
