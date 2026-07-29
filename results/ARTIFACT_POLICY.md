# Result artifact policy

`results/` contains both durable evidence and local experiment workspaces.
The distinction is based on reproducibility and review value, not merely file
size.

## What is committed

- A compact report, table, or regression fixture needed to substantiate a
  documented result.
- An immutable selector/promotion/rejection lock referenced from
  `internal_docs/`.
- A small website input referenced from `docs/assets/`.
- A reproduction manifest containing the command, input hashes, configuration,
  schema version, and expected summary metrics.

Every committed artifact must be referenced by a public or internal document.
Large binary data must use an external dataset release rather than Git.

## What remains local

- Parameter sweeps, per-epoch traces, candidate pools, caches, logs,
  trajectories, visualisation frames, and intermediate refits.
- Files that can be recreated from a checked-in CLI and an identified dataset.
- Ad-hoc development, truth-audit, and debugging outputs.

These files should be written under an ignored experiment directory. Do not
use `git add -f` to bypass this policy.

## WP29-WP31 classification (2026-07-29)

| Workspace | Files | Approx. size | Classification | Durable record |
| --- | ---: | ---: | --- | --- |
| `wp29` | 830 | 2,686 MiB | Local GPU-scale sweeps and CSV/JSON intermediates | Summarise selected metrics in `benchmarks/RESULTS.md` before publication |
| `wp30` | 17 | 1.6 MiB | Local M4-lock reproduction bundle | Copy only the final immutable lock to `internal_docs/` when referenced |
| `wp31` | 1,964 | 1,823 MiB | Local PF/RTK candidate, refit, screen, and trajectory workspace | Conclusions and rejection/promotion locks already belong in `internal_docs/` |

All three workspaces were untracked at classification time and are ignored.
Their root JSON reports are generated summaries, not authoritative records,
until a document references a deliberately copied lock.

Before retaining a result, answer all of the following:

1. Which checked-in document references it?
2. Which command and input dataset reproduce it?
3. Is the schema/version recorded?
4. Can a smaller summary or regression fixture prove the same claim?

If any answer is missing, keep the artifact local.
