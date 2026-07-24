# WP24 Report — evidence ledger and deterministic FIX replay

## Verdict

**Pass.** The PF-only trusted FIX path now has an auditable observation ledger
and a deterministic truth-free replay surface. The integration is exactly
neutral over the measured Tokyo run2/200 reference.

## Delivered

- `python/gnss_gpu/rtk_evidence.py`: evidence records, strict ledger audit,
  common epoch trace, stable ambiguity-assignment ID, and replayable commit
  policy;
- basin update log marginals for Doppler, DDPR, and fixed DDCP;
- opt-in `--out-trace` and `--out-evidence` runner artifacts;
- online assertions against the legacy basin gamma streak;
- end-of-run evidence audit and online/replay equivalence check;
- focused synthetic tests for duplicate/partial evidence, streak/reset/guard,
  assignment generation identity, and mixture evidence.

The trace contains no truth-derived error or reference position. Truth remains
in the separate diagnostics/scoring path after the emitted position and FIX
decision have been made.

## Real-data evidence

Command:

```powershell
python experiments/exp_wp23b_basin_ar.py --run tokyo/run2 --max-epochs 200 `
  --out-trace results/wp24/csv/wp24_smoke_run2_200_trace.csv `
  --out-evidence results/wp24/csv/wp24_smoke_run2_200_evidence.csv
```

The smoke declared 14 correct and zero false FIX epochs. All 793 applied
posterior updates consumed beta exactly once; 273 basin updates carry finite
mixture log marginal evidence. Replay produced zero FIX/streak mismatch.

For default-neutrality, the same window was rerun in a detached worktree at
pre-WP24 commit `b3f1106`. Across all 200 epochs, maximum ECEF position delta
and gamma delta were both exactly zero, with zero FIX and gamma-FIX mismatch.

## Boundary

WP24 does not increase RTK coverage by itself. It makes the next claim
testable: WP25 must show that multi-epoch lineage/motion evidence selects a
correct candidate more often than single-epoch posterior mass, without
manufacturing confidence from tightened DDPR.
