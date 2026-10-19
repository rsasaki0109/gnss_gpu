"""Literal prior accuracy driver with only the selected scorer class replaced."""

from types import FunctionType

import bench_rbpf_cluster_batch_field as base
from rbpf_cluster_fast_guard import ClusterBatchScorer


def main():
    function = base.main
    run = FunctionType(function.__code__, dict(function.__globals__, ClusterBatchScorer=ClusterBatchScorer),
        function.__name__, function.__defaults__, function.__closure__)
    return run()


if __name__ == '__main__':
    raise SystemExit(main())
