from pathlib import Path

from experiments.audit_ppc_ambiguity_arcs import _fold
from experiments.mask_ppc_ambiguity_arc_fold import write_arc_fold_mask


def _field(value: float, lli: str = "") -> str:
    return f"{value:14.3f}{lli:1s} "


def test_mask_keeps_complete_arc_and_preserves_code(tmp_path: Path) -> None:
    source = tmp_path / "source.obs"
    output = tmp_path / "fold.obs"
    header = (
        "     3.04           OBSERVATION DATA    M                   RINEX VERSION / TYPE\n"
        "G    2 C1C L1C                                          SYS / # / OBS TYPES\n"
        "                                                            END OF HEADER\n"
    )
    epochs = []
    for second in (0.0, 0.2):
        epochs.append(
            f"> 2024 10 19 00 00 {second:010.7f}  0  1\n"
            f"G01{_field(20_000_000.0)}{_field(100_000.0)}\n"
        )
    source.write_text(header + "".join(epochs), encoding="ascii")
    arc_id = "tokyo/test:G01:L1C:518400.000:0"
    selected_fold = _fold(arc_id, 3)

    manifest = write_arc_fold_mask(
        source, output, "tokyo/test", selected_fold, fold_count=3
    )

    text = output.read_text(encoding="ascii")
    assert text.count("100000.000") == 2
    assert text.count("20000000.000") == 2
    assert manifest["kept_carrier_fields"] == 2
    assert manifest["masked_carrier_fields"] == 0

    other = (selected_fold + 1) % 3
    manifest = write_arc_fold_mask(
        source, output, "tokyo/test", other, fold_count=3
    )
    text = output.read_text(encoding="ascii")
    assert "100000.000" not in text
    assert text.count("20000000.000") == 2
    assert manifest["masked_carrier_fields"] == 2
