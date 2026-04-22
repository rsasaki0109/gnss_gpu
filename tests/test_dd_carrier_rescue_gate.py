from dataclasses import dataclass

from gnss_gpu.dd_carrier_rescue_gate import evaluate_dd_carrier_rescue_gate
from gnss_gpu.pf_smoother_config import CarrierRescueConfig


@dataclass
class _DummyDDResult:
    n_dd: int


def test_evaluate_dd_carrier_rescue_gate_ignores_missing_or_weak_result():
    decision = evaluate_dd_carrier_rescue_gate(
        None,
        None,
        CarrierRescueConfig(skip_low_support_max_pairs=5),
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )

    assert decision.result is None
    assert not decision.support_skipped
    assert not decision.replace_weak_with_fallback

    weak = _DummyDDResult(2)
    decision = evaluate_dd_carrier_rescue_gate(
        weak,
        None,
        CarrierRescueConfig(skip_low_support_max_pairs=5),
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )

    assert decision.result is weak
    assert not decision.support_skipped


def test_evaluate_dd_carrier_rescue_gate_skips_low_support_result():
    result = _DummyDDResult(4)

    decision = evaluate_dd_carrier_rescue_gate(
        result,
        None,
        CarrierRescueConfig(skip_low_support_max_pairs=5),
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )

    assert decision.result is None
    assert decision.support_skipped
    assert not decision.replace_weak_with_fallback


def test_evaluate_dd_carrier_rescue_gate_marks_weak_dd_replacement():
    result = _DummyDDResult(4)

    decision = evaluate_dd_carrier_rescue_gate(
        result,
        None,
        CarrierRescueConfig(fallback_weak_dd_max_pairs=5),
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )

    assert decision.result is result
    assert not decision.support_skipped
    assert decision.replace_weak_with_fallback


def test_evaluate_dd_carrier_rescue_gate_requires_no_dd_pr_when_configured():
    result = _DummyDDResult(4)
    dd_pr_result = _DummyDDResult(3)
    config = CarrierRescueConfig(
        fallback_weak_dd_max_pairs=5,
        fallback_weak_dd_require_no_dd_pr=True,
    )

    with_dd_pr = evaluate_dd_carrier_rescue_gate(
        result,
        dd_pr_result,
        config,
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )
    without_dd_pr = evaluate_dd_carrier_rescue_gate(
        result,
        None,
        config,
        ess_ratio=None,
        spread_m=None,
        raw_abs_afv_median_cycles=None,
    )

    assert not with_dd_pr.replace_weak_with_fallback
    assert without_dd_pr.replace_weak_with_fallback
