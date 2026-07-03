"""RTKLIB pntpos SPP measurement export."""

from __future__ import annotations

from gnss_gpu.input_validation import finite_float


def export_spp_meas(obs_file, nav_file, el_mask_deg=15.0):
    """Run RTKLIB pntpos SPP and return per-satellite measurements.

    See ``gnss_gpu._gnss_gpu_rtklib_spp.export_spp_meas`` for output keys.
    """
    obs = str(obs_file).strip()
    nav = str(nav_file).strip()
    if not obs:
        raise ValueError("obs_file must be a non-empty path")
    if not nav:
        raise ValueError("nav_file must be a non-empty path")

    el_mask = finite_float("el_mask_deg", el_mask_deg)
    if el_mask < 0.0 or el_mask > 90.0:
        raise ValueError("el_mask_deg must be in [0, 90]")

    from gnss_gpu._gnss_gpu_rtklib_spp import export_spp_meas as _export_spp_meas

    return _export_spp_meas(obs, nav, el_mask)
