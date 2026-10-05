"""RTK FIXED epochs redraw the PF smoother cloud in both passes."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

import gnss_gpu.pf_smoother_epoch_updates as updates_mod
from gnss_gpu.pf_smoother_cli_config import namespace_to_run_config
from gnss_gpu.pf_smoother_cli_parser import build_pf_smoother_arg_parser
from gnss_gpu.pf_smoother_epoch_updates import apply_forward_epoch_updates
from gnss_gpu.rtk_anchor import RTK_FIXED_STATUS, load_rtk_fix_lookup
from gnss_gpu.smoother_epoch_store import build_smoother_epoch_store_inputs
from pf_smoother_forward_helpers import (
    base_config,
    make_forward_context,
    make_measurement_inputs,
)


def _record(tow, status, ecef):
    return SimpleNamespace(
        time=SimpleNamespace(tow=tow),
        status=status,
        position_ecef_m=list(ecef),
    )


def test_lookup_keeps_only_fixed_epochs(tmp_path):
    pos = tmp_path / "rtk.pos"
    pos.write_text("", encoding="utf-8")
    records = [
        _record(100.04, RTK_FIXED_STATUS, (1.0, 2.0, 3.0)),
        _record(100.1, 3, (9.0, 9.0, 9.0)),
        _record(100.2, RTK_FIXED_STATUS, (np.nan, 0.0, 0.0)),
    ]
    lookup = load_rtk_fix_lookup(
        pos,
        load_solution_func=lambda path: SimpleNamespace(records=lambda: records),
    )
    assert list(lookup) == [100.0]
    np.testing.assert_allclose(lookup[100.0], [1.0, 2.0, 3.0])


def test_lookup_requires_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_rtk_fix_lookup(tmp_path / "missing.pos", load_solution_func=lambda path: None)


def test_cli_flows_into_config():
    parser = build_pf_smoother_arg_parser(1.2)
    args = parser.parse_args(
        ["--data-root", "unused", "--rtk-anchor-pos", "rtk.pos", "--rtk-anchor-sigma-m", "0.05"]
    )
    cfg = namespace_to_run_config(args, position_update_sigma=1.9, use_smoother=True)
    assert cfg.rtk_anchor_pos == "rtk.pos"
    assert cfg.particle_filter.rtk_anchor_sigma_m == 0.05
    default = namespace_to_run_config(
        parser.parse_args(["--data-root", "unused"]),
        position_update_sigma=1.9,
        use_smoother=True,
    )
    assert default.rtk_anchor_pos == ""


class _AnchorPf:
    def __init__(self):
        self.resets = []

    def reset_position(self, ref, sigma):
        self.resets.append((np.asarray(ref).copy(), sigma))


@pytest.mark.parametrize("tow, expect_reset", [(123.4, True), (123.5, False)])
def test_forward_epoch_redraws_cloud_at_fixed_epochs(monkeypatch, tow, expect_reset):
    for name in (
        "apply_widelane_dd_pseudorange_update",
        "apply_carrier_epoch_update",
        "apply_doppler_epoch_update",
        "apply_position_epoch_updates",
    ):
        monkeypatch.setattr(updates_mod, name, lambda *args, **kwargs: None)
    context = make_forward_context([], config=base_config(rtk_anchor_sigma_m=0.2))
    pf = _AnchorPf()
    context.pf = pf
    fix = np.array([7.0, 8.0, 9.0])
    context.rtk_anchor_lookup = {123.4: fix}
    epoch_state = cast(Any, SimpleNamespace(velocity=None, rtk_anchor_ref=None))

    apply_forward_epoch_updates(
        context,
        epoch_state,
        make_measurement_inputs(),
        SimpleNamespace(position_ecef_m=np.zeros(3)),
        [],
        tow=tow,
        dt=0.1,
    )

    if expect_reset:
        assert len(pf.resets) == 1
        np.testing.assert_allclose(pf.resets[0][0], fix)
        assert pf.resets[0][1] == 0.2
        assert epoch_state.rtk_anchor_ref is fix
        assert context.stats.n_rtk_anchor_used == 1
    else:
        assert pf.resets == []
        assert epoch_state.rtk_anchor_ref is None
        assert context.stats.n_rtk_anchor_used == 0


def test_store_inputs_carry_anchor_for_backward_pass():
    common: dict[str, Any] = dict(
        spp_pos=np.zeros(3),
        position_update_sigma=None,
        dd_pseudorange_result=None,
        dd_pseudorange_sigma=1.0,
        used_widelane_epoch=False,
        dd_carrier_result=None,
        dd_carrier_sigma_cycles=None,
        anchor_attempt=None,
        carrier_anchor_sigma_m=0.25,
        fallback_attempt=None,
        carrier_afv_wavelength_m=0.19,
        doppler_update=None,
        doppler_sigma_mps=None,
        doppler_velocity_update_gain=0.25,
        doppler_max_velocity_update_mps=10.0,
    )
    anchored = build_smoother_epoch_store_inputs(
        **common, rtk_anchor_ref=np.array([1.0, 2.0, 3.0]), rtk_anchor_sigma_m=0.1
    ).as_store_kwargs()
    np.testing.assert_allclose(anchored["rtk_anchor"], [1.0, 2.0, 3.0])
    assert anchored["rtk_anchor_sigma"] == 0.1

    plain = build_smoother_epoch_store_inputs(**common, rtk_anchor_sigma_m=0.1).as_store_kwargs()
    assert plain["rtk_anchor"] is None
    assert plain["rtk_anchor_sigma"] is None


def test_heading_option_flows_into_config():
    parser = build_pf_smoother_arg_parser(1.2)
    cfg = namespace_to_run_config(
        parser.parse_args(
            ["--data-root", "unused", "--rtk-anchor-pos", "rtk.pos", "--rtk-anchor-heading"]
        ),
        position_update_sigma=1.9,
        use_smoother=True,
    )
    assert cfg.rtk_anchor_heading is True
    assert namespace_to_run_config(
        parser.parse_args(["--data-root", "unused"]),
        position_update_sigma=1.9,
        use_smoother=True,
    ).rtk_anchor_heading is False


def test_anchor_distance_weights_favor_the_nearer_anchor():
    from gnss_gpu.pf_device_smoother import anchor_distance_weights

    # Anchors at epochs 0 and 10 (1 s apart each), none in between.
    anchored = [True] + [False] * 9 + [True]
    w = anchor_distance_weights([1.0] * 11, anchored)
    # At an anchored epoch both passes were just redrawn on the fix.
    assert w[0] == 0.5 and w[10] == 0.5
    assert w[1] > 0.5 > w[9]  # forward wins after the first anchor, backward before the last
    assert abs(w[5] - 0.5) < 1e-12  # midway
    assert np.all((w >= 0.1) & (w <= 0.9))


def test_anchor_distance_weights_without_anchors_are_equal():
    from gnss_gpu.pf_device_smoother import anchor_distance_weights

    np.testing.assert_allclose(anchor_distance_weights([0.1] * 5, [False] * 5), 0.5)
