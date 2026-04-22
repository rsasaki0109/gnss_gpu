from dataclasses import dataclass

import pytest

from gnss_gpu.dd_carrier_update import build_dd_carrier_update_decision
from gnss_gpu.pf_smoother_config import DDCarrierConfig


@dataclass
class _DummyDDResult:
    n_dd: int


def test_build_dd_carrier_update_decision_skips_missing_or_weak_result():
    config = DDCarrierConfig(enabled=True, sigma_cycles=0.05)

    assert not build_dd_carrier_update_decision(
        None,
        config,
        raw_abs_afv_median_cycles=None,
        ess_ratio=None,
    ).apply_update
    assert not build_dd_carrier_update_decision(
        _DummyDDResult(2),
        config,
        raw_abs_afv_median_cycles=None,
        ess_ratio=None,
    ).apply_update


def test_build_dd_carrier_update_decision_uses_base_sigma():
    decision = build_dd_carrier_update_decision(
        _DummyDDResult(4),
        DDCarrierConfig(enabled=True, sigma_cycles=0.05),
        raw_abs_afv_median_cycles=None,
        ess_ratio=None,
    )

    assert decision.apply_update
    assert decision.sigma_cycles == 0.05
    assert decision.sigma_scale == 1.0
    assert not decision.sigma_relaxed


def test_build_dd_carrier_update_decision_relaxes_sigma_for_low_support():
    decision = build_dd_carrier_update_decision(
        _DummyDDResult(4),
        DDCarrierConfig(
            enabled=True,
            sigma_cycles=0.05,
            sigma_support_low_pairs=4,
            sigma_support_high_pairs=8,
            sigma_support_max_scale=1.8,
        ),
        raw_abs_afv_median_cycles=None,
        ess_ratio=None,
    )

    assert decision.sigma_support_scale == 1.8
    assert decision.sigma_scale == 1.8
    assert decision.sigma_cycles == pytest.approx(0.09)
    assert decision.sigma_relaxed


def test_build_dd_carrier_update_decision_combines_and_clips_sigma_scales():
    decision = build_dd_carrier_update_decision(
        _DummyDDResult(5),
        DDCarrierConfig(
            enabled=True,
            sigma_cycles=0.1,
            sigma_afv_good_cycles=0.1,
            sigma_afv_bad_cycles=0.3,
            sigma_afv_max_scale=2.0,
            sigma_ess_low_ratio=0.2,
            sigma_ess_high_ratio=0.8,
            sigma_ess_max_scale=1.5,
            sigma_max_scale=2.5,
        ),
        raw_abs_afv_median_cycles=0.3,
        ess_ratio=0.2,
    )

    assert decision.sigma_afv_scale == 2.0
    assert decision.sigma_ess_scale == 1.5
    assert decision.sigma_scale == 2.5
    assert decision.sigma_cycles == pytest.approx(0.25)


def test_build_dd_carrier_update_decision_ignores_afv_scale_without_raw_metric():
    decision = build_dd_carrier_update_decision(
        _DummyDDResult(5),
        DDCarrierConfig(
            enabled=True,
            sigma_cycles=0.1,
            sigma_afv_good_cycles=0.1,
            sigma_afv_bad_cycles=0.3,
            sigma_afv_max_scale=2.0,
        ),
        raw_abs_afv_median_cycles=None,
        ess_ratio=None,
    )

    assert decision.sigma_afv_scale == 1.0
    assert decision.sigma_cycles == 0.1
