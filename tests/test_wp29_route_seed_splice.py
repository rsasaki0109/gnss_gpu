import pytest

from experiments.splice_wp29_route_seed_traces import splice_rows


def _row(epoch: int, source: str) -> dict[str, str]:
    return {"epoch": str(epoch), "source": source}


def test_splice_rows_switches_without_gap() -> None:
    rows = splice_rows(
        [_row(epoch, "a") for epoch in range(2, 7)],
        [_row(epoch, "b") for epoch in range(5, 9)],
        5,
    )
    assert [int(row["epoch"]) for row in rows] == list(range(2, 9))
    assert [row["source"] for row in rows] == ["a", "a", "a", "b", "b", "b", "b"]


def test_splice_rows_rejects_gap() -> None:
    with pytest.raises(RuntimeError):
        splice_rows([_row(1, "a")], [_row(3, "b")], 3)
