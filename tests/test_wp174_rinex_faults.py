from __future__ import annotations

from experiments.inject_wp174_rinex_faults import _parse, inject


HEADER = [
    "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n",
    "G    4 C1C L1C D1C S1C                                  SYS / # / OBS TYPES\n",
    "                                                            END OF HEADER\n",
]


def _epoch(second: int) -> list[str]:
    record = (
        "G01"
        f"{20000000.0:14.3f}  "
        f"{100000000.0:14.3f}  "
        f"{-1000.0:14.3f}  "
        f"{45.0:14.3f}  \n"
    )
    other = "G02" + record[3:]
    return [
        f"> 2024 07 23 04 04 {30 + second:2d}.0000000  0  2\n",
        record,
        other,
    ]


def _fixture() -> list[str]:
    return HEADER + [
        line for second in range(12) for line in _epoch(second)
    ]


def test_raw_outage_keeps_epoch_and_sets_zero_satellites() -> None:
    mutated, manifest = inject(
        _fixture(), fault="outage", event_count=1, duration_s=1.0
    )
    _, epochs = _parse(mutated)
    assert len(epochs) == 12
    assert any(not epoch.records for epoch in epochs)
    assert manifest["truth_used_for_mutation"] is False


def test_cycle_slip_changes_phase_and_sets_lli_only_on_selected_satellite() -> None:
    source = _fixture()
    mutated, _ = inject(
        source, fault="cycle_slip", event_count=1, duration_s=0.2
    )
    _, before = _parse(source)
    _, after = _parse(mutated)
    changed = [
        (old, new)
        for old_epoch, new_epoch in zip(before, after)
        for old, new in zip(old_epoch.records, new_epoch.records)
        if old != new
    ]
    assert changed
    old_phase = float(changed[0][0][19:33])
    new_phase = float(changed[0][1][19:33])
    assert new_phase - old_phase == 100.0
    assert changed[0][1][33] == "1"


def test_satellite_loss_never_drops_every_satellite() -> None:
    mutated, _ = inject(
        _fixture(), fault="satellite_loss", event_count=1, duration_s=1.0
    )
    _, epochs = _parse(mutated)
    assert all(epoch.records for epoch in epochs)


def test_anchor_schedule_requires_prior_library_fixed_streak() -> None:
    source = _fixture()
    _, epochs = _parse(source)
    fixed_tows = {
        round(epoch.tow, 3) for epoch in epochs[2:10]
    }
    _, manifest = inject(
        source,
        fault="cycle_slip",
        event_count=1,
        duration_s=0.2,
        fixed_anchor_tows=fixed_tows,
        anchor_streak_epochs=3,
        recovery_horizon_s=1.0,
    )
    event = manifest["events"][0]
    start = event["start_index"]
    assert manifest["event_selection"] == "baseline_library_status4_anchor"
    assert all(
        round(epochs[index].tow, 3) in fixed_tows
        for index in range(start - 3, start)
    )
