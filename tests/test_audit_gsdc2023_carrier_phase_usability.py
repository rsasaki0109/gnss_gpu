from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from experiments.audit_gsdc2023_carrier_phase_usability import carrier_usability_row


def test_carrier_usability_row_counts_adr_continuity_and_tdcp_pairs() -> None:
    batch = SimpleNamespace(
        times_ms=np.array([1000, 2000, 3000], dtype=np.float64),
        n_sat_slots=3,
        dual_frequency=True,
        tdcp_consistency_mask_count=2,
        tdcp_geometry_correction_count=4,
        slot_keys=((1, 1, "GPS_L1_CA"), (1, 1, "GPS_L5_Q"), (6, 11, "GAL_E1_C_P")),
        adr=np.array(
            [
                [1.0, 2.0, np.nan],
                [2.0, 3.0, 4.0],
                [3.0, 0.0, 5.0],
            ],
            dtype=np.float64,
        ),
        adr_state=np.array(
            [
                [1, 1, 0],
                [1, 3, 1],
                [1, 0, 5],
            ],
            dtype=np.int32,
        ),
        adr_uncertainty=np.array(
            [
                [0.02, 0.03, np.nan],
                [0.02, 0.04, 0.05],
                [0.03, np.nan, 0.06],
            ],
            dtype=np.float64,
        ),
        tdcp_weights=np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        ),
    )

    row = carrier_usability_row("train/run/pixel5", batch)

    assert row["slot_gps"] == 2
    assert row["slot_galileo"] == 1
    assert row["slot_l5e5"] == 1
    assert row["adr_observations"] == 7
    assert row["valid_adr_observations"] == 5
    assert row["reset_observations"] == 1
    assert row["cycle_slip_observations"] == 1
    assert row["continuous_valid_pairs"] == 2
    assert row["tdcp_pairs"] == 3
    assert row["tdcp_mean_pairs_per_interval"] == 1.5


def test_carrier_usability_row_handles_missing_adr() -> None:
    batch = SimpleNamespace(
        times_ms=np.array([1000], dtype=np.float64),
        n_sat_slots=0,
        dual_frequency=False,
        tdcp_consistency_mask_count=0,
        tdcp_geometry_correction_count=0,
        slot_keys=(),
        adr=None,
        adr_state=None,
        adr_uncertainty=None,
        tdcp_weights=None,
    )

    row = carrier_usability_row("train/run/pixel5", batch)

    assert row["adr_observations"] == 0
    assert row["valid_adr_ratio"] == 0.0
    assert row["tdcp_pairs"] == 0
