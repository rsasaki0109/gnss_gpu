# Experiments Directory

This directory is intentionally large. It contains active experiment runners,
old sweeps, analysis scripts, artifact builders, and one-off probes. Do not
treat every script here as a stable API.

## Naming Conventions

- `exp_*.py`: experiment runners that execute a solver/evaluation pipeline.
- `sim_*.py`: offline simulation or replay over already materialized outputs.
- `analyze_*.py`: post-processing, diagnostics, or report extraction.
- `materialize_*.py`: scripts that generate candidate `.pos`/CSV artifacts.
- `build_*.py`: figure, dashboard, or site asset builders.
- `sweep_*.py`: parameter sweeps.

Reusable code should move to `python/gnss_gpu/` or `src/` only after it survives
fixed evaluation. Until then, keep variant-heavy code here.

## Current PPC Entry Points

| script | role |
|---|---|
| `exp_ppc_ctrbpf_fgo.py` | main PPC PF/RBPF + hybrid + RTKDiag/FGO experiment runner |
| `exp_ppc_libgnss_hybrid.py` | scores `libgnss++` RTK/SPP `.pos` outputs under PPC denominator |
| `analyze_ppc_internal_state.py` | summarizes internal PF/RTKDiag epoch diagnostics |
| `exp_ppc_realtime_fusion.py` | realtime-fusion experiments and per-epoch summaries |
| `sim_ppc_segment_ungated_replay.py` | offline selector/candidate replay for local windows |
| `sim_ppc_segment_candidate_audit.py` | candidate availability and segment audit tooling |
| `materialize_phase70_osm_road_centerline_candidate.py` | materializes the OSM road-centerline corrected candidate used by Phase71 |
| `build_phase70_osm_road_ranker_overlay.py` | appends high-score selector rows for triggered OSM-road epochs |
| `scripts_run_phase70_osmroad_neutral_check.sh` | six-run neutral check for all-run OSM candidate presence |
| `scripts_run_phase71_osmroad_production.sh` | current Phase71 production replay; regenerates large OSM artifacts under `/tmp` |

## Current Non-PPC Entry Points

| script | role |
|---|---|
| `build_githubio_summary.py` | generated visual snapshot under `docs/` |
| `build_paper_assets.py` | paper-facing summary tables and figures |
| `reproduce_gsdc2023_matlab_reference_final.py` | GSDC MATLAB/reference final-output parity |

## Results

Generated artifacts live under [`results/`](results/). That directory contains
many historical sweeps, so use [`results/README.md`](results/README.md) before
opening random CSVs.

## Policy

- Keep current, repeatable commands in docs; keep raw sweep noise out of the
  root README.
- When an experiment becomes current guidance, summarize it in
  [`../internal_docs/ppc_current_status.md`](../internal_docs/ppc_current_status.md)
  or [`../internal_docs/decisions.md`](../internal_docs/decisions.md).
- Do not delete or overwrite generated artifacts unless the cleanup target is
  explicit.
