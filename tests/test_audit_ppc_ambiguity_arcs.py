from pathlib import Path

from experiments.audit_ppc_ambiguity_arcs import audit_rinex_arcs


def _field(value: float, lli: str = "") -> str:
    return f"{value:14.3f}{lli:1s} "


def test_arc_audit_splits_on_lli_and_gap(tmp_path: Path) -> None:
    path = tmp_path / "rover.obs"
    header = (
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G    2 C1C L1C                                          SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
    )
    epochs = []
    for second, lli in ((0.0, ""), (0.2, ""), (0.4, "1"), (2.2, "")):
        epochs.append(
            f"> 2024 10 19 00 00 {second:010.7f}  0  1\n"
            f"G01{_field(20_000_000.0)}{_field(100_000.0, lli)}\n"
        )
    path.write_text(header + "".join(epochs), encoding="ascii")

    audit = audit_rinex_arcs(path, "tokyo/test", fold_count=3)

    assert audit["epochs"] == 4
    assert audit["ambiguity_arcs"] == 3
    assert audit["lli_started_arcs"] == 1
    assert audit["arc_epochs"] == 4
    assert sum(audit["arc_fold_counts"].values()) == 3
