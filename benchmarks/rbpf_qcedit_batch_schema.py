"""Accept field C/N0 thresholds stored as ordinary exact integers.

Only the guard sees float views of these integers. Original metadata-cache
conversion and the actual Nav object/threshold values remain unchanged.
"""

from types import SimpleNamespace

from cssrlib.gnss import Nav

import rbpf_qcedit_batch_list as base
from rbpf_qcedit_batch import NAV_FIELDS
from rbpf_qcedit_batch_list import (  # noqa: F401
    capture_input,
    capture_output,
    exact,
    restore,
)


class QcEngine(base.QcEngine):
    def eligible(self, rtk, obs, rs, dts, svh, rr):
        nav = rtk.nav
        if type(nav) is Nav and all(n in vars(nav) for n in NAV_FIELDS):
            names = ("cnr_min", "cnr_min_gpy")
            integers = [
                n
                for n in names
                if type(getattr(nav, n)) is int and abs(getattr(nav, n)) <= 2**53
            ]
            if integers:
                guard_nav = object.__new__(Nav)
                vars(guard_nav).update({n: getattr(nav, n) for n in NAV_FIELDS})
                for n in integers:
                    setattr(guard_nav, n, float(getattr(nav, n)))
                rtk = SimpleNamespace(nav=guard_nav)
        return super().eligible(rtk, obs, rs, dts, svh, rr)


def make_type(original):
    engine = QcEngine(original)

    class BatchedSchemaQc(original):
        def qcedit(self, obs, rs, dts, svh, rr=None):
            return engine(self, obs, rs, dts, svh, rr)

    return BatchedSchemaQc, engine
