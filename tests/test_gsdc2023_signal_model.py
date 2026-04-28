import numpy as np
import pandas as pd

from experiments import gsdc2023_signal_model as signal_model


def test_signal_frequency_labels_distinguish_factor_unknowns():
    assert signal_model.slot_frequency_label("GPS_L5_Q") == "L5"
    assert signal_model.slot_frequency_label("GAL_E1_C_P") == "L1"
    assert signal_model.slot_frequency_label("unexpected") == "L1"

    assert signal_model.factor_frequency_label("GPS_L5_Q") == "L5"
    assert signal_model.factor_frequency_label("GAL_E1_C_P") == "L1"
    assert signal_model.factor_frequency_label("unexpected") is None


def test_multi_gnss_mask_uses_configured_signal_set():
    frame = pd.DataFrame(
        [
            {"ConstellationType": 1, "SignalType": "GPS_L1_CA"},
            {"ConstellationType": 1, "SignalType": "GPS_L5_Q"},
            {"ConstellationType": 6, "SignalType": "GAL_E1_C_P"},
            {"ConstellationType": 6, "SignalType": "GAL_E5A_Q"},
            {"ConstellationType": 3, "SignalType": "GLO_G1"},
        ],
    )

    np.testing.assert_array_equal(
        signal_model.multi_gnss_mask(frame, dual_frequency=False),
        np.array([True, False, True, False, False]),
    )
    np.testing.assert_array_equal(
        signal_model.multi_gnss_mask(frame, dual_frequency=True),
        np.array([True, True, True, True, False]),
    )


def test_clock_kind_and_common_bias_group_mapping_are_stable():
    slot_keys = (
        (1, 3, "GPS_L1_CA"),
        (1, 3, "GPS_L5_Q"),
        (6, 11, "GAL_E1_C_P"),
        (4, 194, "QZS_L5_Q"),
    )

    assert signal_model.clock_kind_for_observation(1, "GPS_L1_CA", dual_frequency=True, multi_gnss=True) == 0
    assert signal_model.clock_kind_for_observation(1, "GPS_L5_Q", dual_frequency=True, multi_gnss=True) == 4
    assert signal_model.clock_kind_for_observation(6, "GAL_E1_C_P", dual_frequency=True, multi_gnss=True) == 2
    assert signal_model.constellation_to_matlab_sys(6) == 8

    np.testing.assert_array_equal(
        signal_model.slot_pseudorange_common_bias_groups(slot_keys),
        np.array([0, 1, 2, 3], dtype=np.int32),
    )
    remapped = signal_model.remap_pseudorange_isb_by_group(
        source_slot_keys=slot_keys,
        source_isb_by_group={0: 10.0, 1: 20.0, 2: 30.0, 3: 40.0},
        target_slot_keys=((4, 194, "QZS_L5_Q"), (6, 11, "GAL_E1_C_P")),
    )
    assert remapped == {0: 40.0, 1: 30.0}


def test_l5_thresholds_are_slot_local():
    thresholds = signal_model.slot_frequency_thresholds(
        [(1, 1, "GPS_L1_CA"), (1, 2, "GPS_L5_Q"), (6, 3, "GAL_E5A_Q")],
        20.0,
        default_l1_threshold=20.0,
        default_l5_threshold=15.0,
    )
    np.testing.assert_allclose(thresholds, np.array([20.0, 15.0, 15.0]))
    assert signal_model.slot_frequency_thresholds(
        [(1, 1, "GPS_L5_Q")],
        12.0,
        default_l1_threshold=20.0,
        default_l5_threshold=15.0,
    ) == 12.0
