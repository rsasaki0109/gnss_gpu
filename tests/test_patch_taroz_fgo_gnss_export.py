from __future__ import annotations

from pathlib import Path

import pytest

from experiments.patch_taroz_fgo_gnss_export import (
    DOPPLER_FACTOR_ANCHOR,
    GRAPH_INIT_ANCHOR,
    OPTSTATUS_ANCHOR,
    PSEUDORANGE_FACTOR_ANCHOR,
    TDCP_XXCC_ANCHOR,
    TDCP_XXDD_ANCHOR,
    TDCP_XXDD_OFFSET_ANCHOR,
    main,
    patch_fgo_gnss_text,
)


def _minimal_fgo_text() -> str:
    return "\n".join(
        [
            "function optstatus = fgo_gnss(datapath, setting, initflag)",
            GRAPH_INIT_ANCHOR.rstrip("\n"),
            "% pseudorange block",
            PSEUDORANGE_FACTOR_ANCHOR.rstrip("\n"),
            "% doppler block",
            DOPPLER_FACTOR_ANCHOR.rstrip("\n"),
            "% tdcp block",
            TDCP_XXDD_OFFSET_ANCHOR.rstrip("\n"),
            TDCP_XXDD_ANCHOR.rstrip("\n"),
            TDCP_XXCC_ANCHOR.rstrip("\n"),
            "% Optimize!",
            OPTSTATUS_ANCHOR.rstrip("\n"),
            'save(fname,"posest","clkest","velest","dclkest");',
            "",
        ]
    )


def test_patch_fgo_gnss_text_inserts_factor_export_hooks() -> None:
    patched = patch_fgo_gnss_text(_minimal_fgo_text())

    assert "gnss_export_factors = {};" in patched
    assert "gnss_export_factor_model = strings(0, 1);" in patched
    assert "gnss_factor = gtsam_gnss.PseudorangeFactor_XC" in patched
    assert "gnss_factor = gtsam_gnss.DopplerFactor_VD" in patched
    assert "gnss_factor = gtsam_gnss.TDCPFactor_XXDD" in patched
    assert "gnss_factor = gtsam_gnss.TDCPFactor_XXCC" in patched
    assert 'gnss_export_field(end + 1, 1) = "P";' in patched
    assert 'gnss_export_field(end + 1, 1) = "D";' in patched
    assert 'gnss_export_field(end + 1, 1) = "L";' in patched
    assert 'gnss_export_factor_model(end + 1, 1) = "XXCC";' in patched
    assert "gnss_export_sigma(end + 1, 1) = obserr.(f).P(i,j);" in patched
    assert "gnss_export_measurement(end + 1, 1) = tdcp_measurement;" in patched
    assert "gnss_export_los(end + 1, :) = losvec';" in patched
    assert "gnss_export_svid(end + 1, 1) = obs.prn(j);" in patched


def test_patch_fgo_gnss_text_inserts_residual_export_hooks() -> None:
    patched = patch_fgo_gnss_text(_minimal_fgo_text())

    assert "phone_data_gnss_factor_mask.csv" in patched
    assert "phone_data_gnss_factor_residuals.csv" in patched
    assert "phone_data_gnss_factor_summary.csv" in patched
    assert "phone_data_gnss_graph_state.csv" in patched
    assert "phone_data_gnss_initial_state.csv" in patched
    assert "results.atVector(sym('x', gnss_epoch))" in patched
    assert "results.atVector(sym('x', gnss_state_epoch))" in patched
    assert "gnss_state_pose = results.atPose3(sym('p', gnss_state_epoch));" in patched
    assert "gnss_state_rpy_col(gnss_state_idx, :) = gnss_state_pose.rotation.rpy';" in patched
    assert "initials.atVector(sym('x', gnss_state_epoch))" in patched
    assert "gnss_initial_pose = initials.atPose3(sym('p', gnss_state_epoch));" in patched
    assert "gnss_initial_rpy_col(gnss_state_idx, :) = gnss_initial_pose.rotation.rpy';" in patched
    assert "'position_x', 'position_y', 'position_z', 'roll', 'pitch', 'yaw'" in patched
    assert "results.atVector(sym('c', gnss_next_epoch))" in patched
    assert "initials.atVector(sym('d', gnss_next_epoch))" in patched
    assert "gnss_export_factors{gnss_export_idx}.error(results)" in patched
    assert "'initial_residual', 'residual', 'residual_delta'" in patched
    assert "'factor_count', 'p_count', 'd_count', 'l_count'" in patched


def test_patch_fgo_gnss_text_rejects_missing_anchor() -> None:
    with pytest.raises(ValueError, match="TDCP XXCC factor anchor"):
        patch_fgo_gnss_text(_minimal_fgo_text().replace(TDCP_XXCC_ANCHOR.rstrip("\n"), "% missing"))


def test_patch_cli_writes_patched_copy(tmp_path: Path) -> None:
    source = tmp_path / "fgo_gnss.m"
    output = tmp_path / "patched" / "fgo_gnss.m"
    source.write_text(_minimal_fgo_text(), encoding="utf-8")

    main([str(source), str(output)])

    patched = output.read_text(encoding="utf-8")
    assert "writetable(gnss_graph_state_table, gnss_export_graph_state_file);" in patched
    assert "writetable(gnss_initial_state_table, gnss_export_initial_state_file);" in patched
    assert "writetable(gnss_factor_mask, gnss_export_mask_file);" in patched
    assert "writetable(gnss_factor_residuals, gnss_export_residual_file);" in patched
    assert "writetable(gnss_summary, gnss_export_summary_file);" in patched
