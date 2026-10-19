"""Add ordinary three-float base-position lists to the frozen QC experiment.

The field runner stores nav.rb as a list. Perform the original asarray
conversion once before the existing guard; custom lists/scalars still fall
back. The first implementation and its completed tests remain unchanged.
"""

import numpy as np

import rbpf_qcedit_batch as base
from rbpf_qcedit_batch import (  # noqa: F401
    NAV_FIELDS,
    OBS_FIELDS,
    capture_input,
    capture_output,
    exact,
    restore,
)


class QcEngine(base.QcEngine):
    def __call__(self, rtk, obs, rs, dts, svh, rr=None):
        if (
            type(rr) is list
            and len(rr) == 3
            and all(type(v) is float for v in rr)
            and np.asarray is base._NP["asarray"]
        ):
            rr = np.asarray(rr, dtype=np.float64)
        return super().__call__(rtk, obs, rs, dts, svh, rr)


def make_type(original):
    engine = QcEngine(original)

    class BatchedListQc(original):
        def qcedit(self, obs, rs, dts, svh, rr=None):
            return engine(self, obs, rs, dts, svh, rr)

    return BatchedListQc, engine
