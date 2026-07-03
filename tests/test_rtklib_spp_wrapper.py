"""CPU-side validation tests for the RTKLIB SPP export wrapper."""

import pytest

from gnss_gpu import rtklib_spp


def test_export_spp_meas_rejects_empty_paths():
    with pytest.raises(ValueError, match="obs_file must be a non-empty path"):
        rtklib_spp.export_spp_meas("", "nav.rnx")

    with pytest.raises(ValueError, match="nav_file must be a non-empty path"):
        rtklib_spp.export_spp_meas("obs.rnx", "   ")


def test_export_spp_meas_rejects_invalid_el_mask():
    with pytest.raises(ValueError, match="el_mask_deg must be finite"):
        rtklib_spp.export_spp_meas("obs.rnx", "nav.rnx", el_mask_deg=float("nan"))

    with pytest.raises(ValueError, match="el_mask_deg must be in"):
        rtklib_spp.export_spp_meas("obs.rnx", "nav.rnx", el_mask_deg=95.0)
