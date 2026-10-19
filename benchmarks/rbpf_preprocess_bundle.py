"""Compose frozen exact QC and ambiguity bookkeeping experiments.

Only class method dispatch is combined. Original numerical functions and
both guarded implementations remain unchanged. An explicit experiment;
individual historical whole-RTK timing failures are still failures.
"""

from rbpf_ambiguity_bookkeeping import make_type as make_bookkeeping
from rbpf_qcedit_batch_field_schema import make_type as make_qc


class BundleEngine:
    def __init__(self, qc, bookkeeping):
        self.qc, self.bookkeeping = qc, bookkeeping

    def counts(self):
        """Preserve the existing QC field/route runner counter contract."""
        return self.qc.counts()

    def details(self):
        return dict(
            qc=self.qc.counts(),
            initx=self.bookkeeping.counts(),
            bookkeeping=self.bookkeeping.scan_counts(),
            bookkeeping_AST_proof=self.bookkeeping.proof,
        )


def make_type(original):
    book_type, bookkeeping = make_bookkeeping(original)
    chosen, qc = make_qc(book_type)
    assert chosen.qcedit is not original.qcedit
    assert chosen.update_ambiguities is book_type.update_ambiguities
    assert chosen.initx is book_type.initx
    assert qc.original is original.qcedit
    assert bookkeeping.original is original.update_ambiguities
    assert bookkeeping.initx_engine.original is original.initx
    return chosen, BundleEngine(qc, bookkeeping)
