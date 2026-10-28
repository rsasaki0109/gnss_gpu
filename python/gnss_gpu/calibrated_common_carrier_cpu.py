"""Model selection for common-carrier fits; gate must match the full pipeline."""
import numpy as np


def select_common_carrier_paths(score, gate):
    """No generating model order or carrier truth is accepted.

    A score above the frozen null gate requests two paths. Invalid requested
    fits return ambiguous, without falling back to the other model.
    """
    if not np.isfinite(score['score']) or not np.isfinite(gate.score):
        raise ValueError('finite score and calibrated gate required')
    two = score['score'] > gate.score
    fit = score['two' if two else 'single']
    if not fit.locally_usable:
        return dict(model='ambiguous',delay_chips=None,reason=fit.reason)
    return dict(model='two_path' if two else 'single_path',
                delay_chips=fit.first_delay_chips,
                reason='calibrated gate and local guards passed' if two else 'below calibrated gate and single-path guards passed')
