"""Read-only work/ownership audit of the frozen hypothesis copy function.

This observer measures counts, not elapsed time. It does not replace a numeric
function, share dictionaries, suppress a copy, or change the GC policy.
"""

import sys
from types import FunctionType


def _plain(hypothesis):
    if type(hypothesis) is not dict:
        return False
    return all(type(g) is tuple and len(g) == 2
        and all(type(x) is int for x in g) and type(z) is dict
        and all(type(s) is int and type(v) is int for s, v in z.items())
        for g, z in hypothesis.items())


def inventory(hypotheses):
    """Count repeated dict identities without retaining hypotheses afterwards.

    The caller supplies named hypotheses. Repetition within and across them is
    counted; values/order are never canonicalized. Custom mappings are skipped.
    """
    outer, inner = set(), set()
    result = dict(hypotheses=0, unsupported=0, repeated_outer=0,
        repeated_inner=0, groups=0, entries=0)
    for _, h in hypotheses:
        result["hypotheses"] += 1
        if not _plain(h):
            result["unsupported"] += 1
            continue
        result["repeated_outer"] += int(id(h) in outer)
        outer.add(id(h))
        for z in h.values():
            result["repeated_inner"] += int(id(z) in inner)
            inner.add(id(z))
            result["groups"] += 1
            result["entries"] += len(z)
    return result


class CopyWork:
    """Observe one exact code object and restore the owned profile callback.

    Call sites distinguish temporary move construction from particle commit,
    resampling, and shadow/cluster copies. For ordinary integer dictionaries,
    check the literal ordered result and fresh outer/inner dictionary identities.
    A temporary move's unchanged groups are an allocation-only opportunity count:
    it is not a speed estimate or permission to share a mutable particle state.
    """

    def __init__(self, function, temporary_callers=None):
        if type(function) is not FunctionType:
            raise TypeError("An exact Python function is required")
        self.code = function.__code__
        self.temporary_callers = dict(temporary_callers or {})
        self.sites, self.pending = {}, {}
        self.callback = self._profile
        self.active = False
        self.restored = False

    def __enter__(self):
        if self.active or sys.getprofile() is not None:
            raise RuntimeError("An existing profiler must not be replaced")
        self.active = True
        sys.setprofile(self.callback)
        return self

    def __exit__(self, typ, value, traceback):
        if sys.getprofile() is not self.callback:
            raise RuntimeError("The observer no longer owns the profiler")
        sys.setprofile(None)
        self.active = False
        self.restored = True
        self.pending.clear()

    def _profile(self, frame, event, arg):
        if frame.f_code is not self.code:
            return
        if event == "call":
            caller = frame.f_back
            key = (caller.f_code.co_filename, caller.f_code.co_name, caller.f_lineno)
            if key not in self.sites:
                self.sites[key] = dict(caller=key[1], line=key[2], filename=key[0],
                    calls=0, completed=0, unsupported=0, source_groups=0,
                    copied_groups=0, copied_entries=0, dropped_empty_groups=0,
                    result_mismatches=0, outer_aliases=0, inner_aliases=0,
                    temporary_unchanged_groups=0, temporary_unchanged_entries=0)
            row = self.sites[key]
            row["calls"] += 1
            h = frame.f_locals["H"]
            if not _plain(h):
                row["unsupported"] += 1
                self.pending[id(frame)] = (row, None)
                return
            expected = tuple((g, tuple(z.items())) for g, z in h.items() if z)
            source_ids = {id(z) for z in h.values()}
            row["source_groups"] += len(h)
            row["copied_groups"] += len(expected)
            row["dropped_empty_groups"] += len(h) - len(expected)
            row["copied_entries"] += sum(len(z) for _, z in expected)
            # Only these two audited frozen callers use a temporary copy whose
            # untouched submaps could remain read-only until the later commit.
            touched = None
            kind = self.temporary_callers.get(caller.f_code)
            if kind == "extend":
                loc = caller.f_locals
                touched = {loc["prob"].pairs[int(i)][0] for i in loc["idxC"]}
            elif kind == "shrink":
                touched = {caller.f_locals["g"]}
            if touched is not None:
                for g, entries in expected:
                    if g not in touched:
                        row["temporary_unchanged_groups"] += 1
                        row["temporary_unchanged_entries"] += len(entries)
            self.pending[id(frame)] = (row, (h, source_ids, expected))
        elif event == "return":
            row, source = self.pending.pop(id(frame))
            if arg is None:
                return  # Python's profile return event also covers exceptions.
            row["completed"] += 1
            if source is None:
                return
            h, source_ids, expected = source
            row["outer_aliases"] += int(arg is h)
            if not _plain(arg):
                row["result_mismatches"] += 1
                return
            row["result_mismatches"] += int(tuple((g, tuple(z.items()))
                for g, z in arg.items()) != expected)
            row["inner_aliases"] += sum(id(z) in source_ids for z in arg.values())

    def records(self):
        if self.active:
            raise RuntimeError("Stop observing before reading records")
        return [dict(row) for _, row in sorted(self.sites.items())]
