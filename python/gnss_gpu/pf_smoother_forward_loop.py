"""Forward-pass loop control for PF smoother evaluations."""

from __future__ import annotations

import time

from gnss_gpu.pf_smoother_forward_context import PfSmootherForwardPassContext
from gnss_gpu.pf_smoother_forward_epoch import process_pf_smoother_forward_epoch


def run_pf_smoother_forward_pass(context: PfSmootherForwardPassContext) -> float:
    t0 = time.perf_counter()
    for sol_epoch, measurements in context.dataset.epochs:
        if not sol_epoch.is_valid() or len(measurements) < 4:
            continue
        if context.history.reached_limit(
            context.run_config.max_epochs,
            context.run_config.skip_valid_epochs,
        ):
            break
        process_pf_smoother_forward_epoch(context, sol_epoch, measurements)
    return (time.perf_counter() - t0) * 1000
