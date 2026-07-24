from __future__ import annotations

import numpy as np
import pytest

from experiments.apply_wp29_route_template_bridge_shadow import route_template_bridge


def test_route_template_bridge_uses_target_progress() -> None:
    template = np.column_stack([np.linspace(0.0, 10.0, 21), np.zeros(21), np.zeros(21)])
    target = np.column_stack(
        [np.asarray([0.0, 1.0, 4.0, 9.0, 10.0]), np.zeros(5), np.zeros(5)]
    )
    bridge, metrics = route_template_bridge(
        target,
        template,
        start=0,
        end=4,
        endpoint_candidates=3,
        max_endpoint_distance_m=0.1,
        max_arc_relative_error=0.01,
    )
    np.testing.assert_allclose(bridge[:, 0], target[:, 0], atol=1.0e-12)
    assert metrics["template_start_index"] == 0
    assert metrics["template_end_index"] == 20


def test_route_template_bridge_rejects_distant_endpoint() -> None:
    target = np.column_stack([np.arange(3.0), np.zeros(3), np.zeros(3)])
    template = target + np.asarray([10.0, 0.0, 0.0])
    with pytest.raises(RuntimeError, match="too far"):
        route_template_bridge(
            target,
            template,
            start=0,
            end=2,
            endpoint_candidates=3,
            max_endpoint_distance_m=1.0,
            max_arc_relative_error=0.1,
        )
