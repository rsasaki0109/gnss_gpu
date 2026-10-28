import numpy as np
import pytest

from experiments.run_cpu_noncoherent_bit_validation import signal_scene, exact_paired_sign_p, summarize_signals
from experiments.run_cpu_batch_receiver import render, COUNT


def test_random_start_forces_interior_period_flip_and_preserves_paired_noise():
    a, starts, seeds, nominal, flip = signal_scene(37., 700., 1000, "constant")
    b, starts2, seeds2, nominal2, flip2 = signal_scene(37., 700., 1000, "flip")
    assert starts == starts2 and seeds == seeds2 and nominal == nominal2 and flip == flip2
    assert 0 < flip < 20*COUNT and flip % COUNT != 0
    # Isolate PRN 1, with noise removed: bit timing must affect precisely the
    # expected absolute samples, without resetting code/carrier between rows.
    control = render("batch", [[row[0]] for row in a], starts, seeds, -10000.).ravel()
    changed = render("batch", [[row[0]] for row in b], starts, seeds, -10000.).ravel()
    np.testing.assert_array_equal(changed[:2*flip], control[:2*flip])
    np.testing.assert_array_equal(changed[2*flip:], -control[2*flip:])
    np.testing.assert_array_equal(render("individual", b, starts2, seeds2, -10000.),
                                  render("batch", b, starts2, seeds2, -10000.))


def test_bit_modes_do_not_mutate_other_satellite_polarities():
    rows, *_ = signal_scene(37., 170., 1001, "flip")
    assert all("nav_bits" not in channel for row in rows for channel in row[1:])
    with pytest.raises(ValueError):
        signal_scene(37., 170., 1001, "bad")


def test_exact_paired_sign_test_and_zero_discordants():
    assert exact_paired_sign_p(5, 0) == .0625
    assert exact_paired_sign_p(0, 5) == .0625
    assert exact_paired_sign_p(3, 3) == 1.
    assert exact_paired_sign_p(0, 0) == 1.
    with pytest.raises(ValueError):
        exact_paired_sign_p(-1, 0)


def test_primary_family_keeps_failures_and_adjusts_only_declared_endpoints():
    def row(replica, mode, a, b):
        return dict(replica=replica, bit_mode=mode, nominal_primary_cn0_db_hz=37., primary_doppler_hz=170.,
                    detectors={"1": dict(accepted=a, passes_declared_criterion=a),
                               "4": dict(accepted=b, passes_declared_criterion=b)})
    records = [row(i, mode, False, i < 5) for mode in ("constant", "flip") for i in range(6)]
    grouped, paired = summarize_signals(records)
    assert all(result["replicas"] == 6 for result in grouped)
    primary = next(result for result in paired if result["primary_endpoint"])
    assert primary["only_4ms"] == 5 and primary["only_1ms"] == 0
    assert primary["primary_bonferroni_p"] == .125
    assert next(result for result in paired if result["bit_mode"] == "constant")["primary_bonferroni_p"] is None
