"""Release completed call-local bound-method cycles without changing PF math."""

from types import MethodType

from rbpf_feedback_mass import NativeFeedbackMassStep
from rbpf_native_canonical import _CanonicalView


class ViewReleaseStep(NativeFeedbackMassStep):
    """Keep D236 verbatim and release only its ordinary completed view cache.

    No GC setting, collection call, cross-epoch view reuse, numeric rewrite,
    changed source module or transient runtime method patch. Retained external
    bound methods remain valid. Unusual cache values/types retain old lifetime.
    """

    def __init__(self, runtime, scorer=None, mode="scalar", **extensions):
        super().__init__(runtime, scorer, mode, **extensions)
        self.release_counts = dict(calls=0, released_methods=0, declined=0)
        base = self.compiled
        self.original_compiled = base

        def call(view, rtk, ctx):
            try:
                return base(view, rtk, ctx)
            finally:
                self.release_view(view)

        self.compiled = call

    def release_view(self, view):
        self.release_counts["calls"] += 1
        if type(view) is _CanonicalView:
            cache = object.__getattribute__(view, "_bound")
            if type(cache) is dict and all(
                type(key) is str and type(value) is MethodType
                and value.__self__ is view
                for key, value in cache.items()
            ):
                count = len(cache)
                cache.clear()
                self.release_counts["released_methods"] += count
                return
        self.release_counts["declined"] += 1

    def __call__(self, rtk, ctx):
        before, previous = self.release_counts.copy(), self.last
        try:
            return super().__call__(rtk, ctx)
        finally:
            if self.last is not previous and self.last is not None:
                self.last["view_release"] = {
                    key: value - before[key]
                    for key, value in self.release_counts.items()
                }
