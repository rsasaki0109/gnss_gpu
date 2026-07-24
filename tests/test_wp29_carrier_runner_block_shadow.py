from experiments.apply_wp29_carrier_runner_block_shadow import (
    contiguous_anchor_blocks,
)


def test_contiguous_anchor_blocks_fail_closed_on_short_runs() -> None:
    assert contiguous_anchor_blocks(
        [10, 15, 20, 30, 35, 40, 45, 50], stride=5, min_anchors=5
    ) == [[30, 35, 40, 45, 50]]
