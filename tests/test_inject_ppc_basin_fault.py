from experiments.inject_ppc_basin_fault import inject_fault


def _rows() -> list[dict]:
    return [
        {
            "schema": "gnsspp_multisd_basin_v1",
            "epoch_index": 2,
            "tow": 10.0,
            "gps_week": 2325,
            "group_index": 0,
            "rank": rank,
            "evaluated": True,
            "pass": rank == 0,
            "position_ecef": [1.0, 2.0, 3.0],
            "fixed_integers": [{"fixed_cycles": 4}],
            "validation_residuals": [{"pass": True}],
            "imu_fgo": {"available": True, "position_ecef": [1, 2, 3]},
        }
        for rank in range(2)
    ]


def test_fault_injection_fails_closed_by_fault_kind() -> None:
    outage = inject_fault(_rows(), fault="outage", first_epoch=2, last_epoch=2)
    assert len(outage) == 1 and outage[0]["evaluated"] is False
    assert outage[0]["gps_week"] == 2325
    assert outage[0]["imu_fgo"]["available"] is True
    ambiguous = inject_fault(
        _rows(), fault="ambiguous_holdout", first_epoch=2, last_epoch=2
    )
    assert sum(row["pass"] for row in ambiguous) == 2
    slipped = inject_fault(_rows(), fault="cycle_slip", first_epoch=2, last_epoch=2)
    assert not any(row["pass"] for row in slipped)
    assert all(row["fixed_integers"][0]["fixed_cycles"] == 5 for row in slipped)
    nlos = inject_fault(_rows(), fault="nlos", first_epoch=2, last_epoch=2)
    assert not any(row["pass"] for row in nlos)
    assert all(row["position_ecef"][0] == 6.0 for row in nlos)
    assert all(not row["validation_residuals"][0]["pass"] for row in nlos)
