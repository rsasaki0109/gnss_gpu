# Experiment Results

This directory is a generated-artifact area. It contains current comparison
outputs, old sweeps, diagnostics, plots, HTML reports, and scratch CSVs. File
presence alone does not mean a result is current.

## Current PPC Artifacts

| artifact | meaning |
|---|---|
| `phase71_osmroad_production_summary.csv` | current Phase71 production replay summary; official 86.205492% |
| `phase71_osmroad_block_other_runs_summary.csv` | blocked projection showing n/r2-only OSM candidate deployment |
| `phase70_osmroad_neutral_check_summary.csv` | six-run neutral check with OSM candidate present on all runs |
| `ppc_phase57_gap_nagoya_run2_internal_epochs.csv` | n/r2 Phase43 internal diagnostics used to materialize the Phase71 OSM candidate |
| `ppc_ctrbpf_fgo_phase43_prod_*_full_runs.csv` | Phase43 production baseline summaries used for Phase71 deltas |
| `ppc_phase71_osmroad_prod_*_full_runs.csv` | Phase71 per-run production replay summaries |
| `selector_ranker_predictions_v5_nlos.csv` | base v5_nlos ranker scores; large generated CSV, not suitable for casual diffs |

The short current interpretation is in
[`../../internal_docs/ppc_current_status.md`](../../internal_docs/ppc_current_status.md).

Large OSM-road materialized candidates and ranker overlay CSVs are generated
under `/tmp` by default by the Phase70/71 scripts. Do not commit those generated
overlay CSVs unless a task explicitly asks for archival.

## Important Directories

| directory | meaning |
|---|---|
| `libgnss_rtk_pos_v5/` | current `gnss_gpu` hybrid floor `.pos` files |
| `libgnss_spp_pos/` | SPP fallback `.pos` files used by hybrid scorer |
| `libgnss_diag_phase10/` | multiple `libgnss++` RTK diagnostic/candidate profiles |
| `libgnss_diag_phase19/` | gici-generated candidate outputs used as local candidate sources |
| `paper_assets/` | generated paper-facing summary figures and tables |

## Output Naming

- `*_runs.csv`: run-level summary.
- `*_summary.csv`: aggregate or compact summary.
- `*_epochs.csv`: per-epoch output.
- `*_internal_epochs.csv`: per-epoch internal state diagnostics.
- `*_labels.csv`: candidate-label or selector-label diagnostics.
- `*_spans.csv`: contiguous span-level summary.
- `.html`: visual report artifact.

## Maintenance Rules

- Do not assume untracked files are trash. Many are generated diagnostics from
  previous sweeps.
- If a generated result becomes important, add it to the current artifact table
  above or to the relevant topic note.
- Prefer adding a new result prefix over overwriting older sweep outputs.
- Cleanup should be explicit and scoped by prefix, date, or experiment family.
