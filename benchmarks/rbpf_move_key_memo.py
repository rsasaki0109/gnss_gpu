"""Explicit lightweight alternative: canonical memo only during move staging."""

from types import SimpleNamespace

from rbpf_epoch_memo import _EpochMemo
from rbpf_move_reuse import MoveReuseStep


class MoveKeyMemoStep(MoveReuseStep):
    """Keep all original bound methods/globals; memoize staging lookup only.

    The previous move construction/weight/choice code is reused verbatim.
    No DD cache or full-epoch method/state view is created. Canonical results
    are retained only until the move block returns, before original commit.
    """

    def _moves(self, prob, H_cons, cand_H, spawn_H, cache, ll_burst):
        memo = _EpochMemo(self.canonical, self.runtime._pair_dd, cache_dd=False)
        context = SimpleNamespace(runtime=self.runtime, oracle=self.oracle, canonical=memo.canonical,
            scorer=self.scorer, mode=self.mode, last=None)
        result = MoveReuseStep._moves(context, prob, H_cons, cand_H, spawn_H, cache, ll_burst)
        self.last = context.last
        self.last['move_key_memo'] = memo.summary()
        return result
