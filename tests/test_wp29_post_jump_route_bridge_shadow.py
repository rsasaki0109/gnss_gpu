import pytest

from experiments.apply_wp29_post_jump_route_bridge_shadow import final_mode_jump_epoch


def _row(epoch: int, residual: float) -> dict[str, str]:
    return {
        "epoch": str(epoch),
        "selected": "1",
        "previous_selected_transition_residual_m": str(residual),
    }


def test_final_mode_jump_selects_last_jump_with_tail() -> None:
    rows = [_row(epoch, 3.0 if epoch in (20, 40) else 0.1) for epoch in range(10, 100, 5)]
    assert final_mode_jump_epoch(rows, start=5, end=100, min_tail_anchors=10) == 40


def test_final_mode_jump_rejects_short_tail() -> None:
    rows = [_row(epoch, 3.0 if epoch == 80 else 0.1) for epoch in range(10, 100, 5)]
    with pytest.raises(RuntimeError):
        final_mode_jump_epoch(rows, start=5, end=100, min_tail_anchors=10)
