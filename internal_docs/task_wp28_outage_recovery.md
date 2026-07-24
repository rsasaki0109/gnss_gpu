# WP28 — outage recovery and active hypothesis management

Date: 2026-07-18. Parent: `pf_only_rtk_scaleup_plan_2026_07_18.md`.

## Objective

Restore correct ambiguity-basin supply and survival after urban outage or
generation change, without weakening the truth-free FIX gate. Begin with
Nagoya run3/200, where WP27's frozen satellite selector sees a live sub-50 cm
candidate on only 13/200 epochs.

## First diagnostic increment

Separate proposal generation from selection and commit policy. Run a frozen
supply grid with:

- DDPR respawn trigger `0 m` (attempt every raw 1 Hz anchor);
- respawn top-K `64/128/256`;
- basin cap `128/256/512`;
- respawn ambiguity subset dimension `6/8`.

All arms keep DDPR sigma, birth mass, float KF, integer search, and output/FIX
policy unchanged. Truth may compute per-epoch oracle minimum candidate error
only after proposals and pruning. No grid result may become a run-specific
production constant.

## Measurements

- raw proposal count and respawn epochs;
- live-basin oracle sub-50 cm epochs;
- correct-candidate contiguous survival spans;
- supplied-then-pruned epochs, requiring proposal-stage oracle diagnostics;
- wall time and maximum live basins;
- operational trajectory/FIX neutrality.

## Gate

The first increment passes only if it identifies whether generation, ranking,
or pruning dominates the 13/200 recall. A proposal arm may advance only if it
increases candidate recall without changing output/FIX. Production recovery
still requires candidate recall `>=90%`, conditional survival p90 `>=5 s`, and
incorrect holdover zero on artificial and natural outages.
