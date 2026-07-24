# WP34 Tokyo relative-secondary development checkpoint

Status: development only. Nothing in this checkpoint is production-promoted.
M4 and the WP33 production benchmark remain unchanged.

## Target interval

Tokyo run1 6945--7076 has two adjacent three-radius horizontal clusters. The
old temporal-direction selector chose IDs 30/94/158 (truth-only mean audit
about 1.07 m). Uncalibrated secondary-family DD pseudorange ranks the adjacent
IDs 31/95/159 first as a three-radius group (mean median residual 0.99 m), but
the coarse runner margin is only about 4.7%, so the coarse result is not used
directly as a production anchor.

The top three coarse secondary groups were resampled identically from central
IDs 95, 94, and 96 with fixed 0.2/0.4 m cube26 shells. Truth-only supply audits
are 44, 0, and 0 sub-50 cm children respectively; best errors are 0.159,
0.706, and 0.520 m. These audit values do not enter selection.

After resampling, the parents' secondary top-three mean residuals are 0.855,
0.934, and 0.989 m. The fixed 7.5% relative-margin gate is passed at 9.23%.
Within the winning parent, the primary DD fit's top three IDs 36/44/62 have
0.282 m spread. Their mean is selected by the development selector and has a
separate truth-only audit error of 0.432 m.

Development artifact:

- `results/wp31/tokyo_run1_static_6945_7076_relative_secondary_parent_development.json`
- SHA-256 `F73BAD3DE9195349EB87FA5DE3BB46DF9249A923FB1E8D420F7DA486187CF4D3`
- audit SHA-256 `58FA8BE8DD865DE799BAB2498F9CE7C321B5B7538AA4A83590C7719AD3E9CF02`

## Positive holdout

The already accepted Tokyo 9883--10248 production cluster IDs 4/68/132 is
independently ranked first by the same uncalibrated secondary three-radius
aggregation. Its runner margin is 33.8%. The secondary artifact SHA-256 is
`B22DC2DF377D6AB2E5E2DF926009B12140233F662A20D619C0F3847DD9CCA390`.

Accepted-anchor pair-bias calibration was also tested and rejected: changing
DD reference-pair state made the target residuals worse (about 6 m median).
The production path therefore remains uncalibrated and truth-free.

## Remaining promotion gate

Before production promotion, reproduce the relative-parent rule on at least
one more independent accepted Tokyo stop and lock explicit fail-closed negative
holdouts. Only then create a hash-verifying promotion script, add the reason to
the smoother allowlist, and measure the full-denominator Tokyo benchmark.

This gate was subsequently satisfied and locked by WP34/WP35. A follow-up on
the large 2464--5866 outage tested the existing 4518--4612 parent28 shell. Its
only sub-50 candidate (ID49, 0.459 m audit) ranks only 22nd by secondary DDPR;
the secondary top three audit at 1.34--1.58 m. It remains rejected. The
secondary artifact SHA-256 is
`E6C67C638AD3F417DAAB2198B9CEA1B9DF0EB81167C31D08BA42E89D9249732C`.
