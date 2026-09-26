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

## Enforcement

`python scripts/ci/check_artifact_policy.py` checks Git-tracked files under
`results/` and `experiments/results/` (excluding this policy and the allowlist).
The Ubuntu lint CI job runs it and fails if any artifact is unreferenced.
Use `--list-unreferenced` for one offending path per line.

References come from other tracked UTF-8 text files up to 2 MiB; binaries and
both result trees are excluded. An artifact is retained by an exact path
substring, or an ancestor path strictly below a result root followed by `/`,
a quote, whitespace, `)`, a backtick, or end of line. A whole-word workspace
name in a `.py`, `.sh`, or `.yml` file containing `results` also counts, as does
a case-sensitive glob in `results/artifact_allowlist.txt`.

To keep a new artifact, reference it from a public or internal document, or
add an allowlist entry with a comment explaining its dependency or retention
reason. Stage the reference document along with the artifact before checking.

The 2026-09-26 audit untracked 314 files with `git rm --cached` (working-tree
copies kept, now ignored): `results/wp25` (8), `results/wp26` (6),
`results/wp27` (84), `results/wp28` (193), `experiments/results/libgnss_viz`
(18), and five files directly under `experiments/results/`. Another 22 files
remain via justified allowlist entries for audit and website consumers.
