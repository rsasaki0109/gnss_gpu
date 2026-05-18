# Internal Documentation Map

This directory contains working notes, decisions, and handoffs. It is not a
polished public documentation tree. Use this map to avoid treating every old
experiment note as current guidance.

## Start Here

- [`ppc_current_status.md`](ppc_current_status.md): current PPC/libgnss++/
  gnss_gpu status, important numbers, and next tasks.
- [`plan.md`](plan.md): long chronological PPC working log. Use it for
  provenance, not as the first document to read.
- [`decisions.md`](decisions.md): durable adoption/rejection decisions.
- [`experiments.md`](experiments.md): older experiment index and result notes.
- [`interfaces.md`](interfaces.md): retained API/interface notes.

## Topic Notes

- [`ctrbpf_fgo_ppc_design.md`](ctrbpf_fgo_ppc_design.md): CT-RBPF/FGO design
  direction.
- [`ppc2024_realtime_target.md`](ppc2024_realtime_target.md): PPC2024 realtime
  target framing.
- [`proper_rbpf_velocity_results.md`](proper_rbpf_velocity_results.md),
  [`rbpf_velocity_gated_results.md`](rbpf_velocity_gated_results.md), and
  [`rbpf_velocity_phase2_results.md`](rbpf_velocity_phase2_results.md): RBPF
  velocity experiment history.
- [`gsdc_dgnss_feasibility.md`](gsdc_dgnss_feasibility.md) and
  `gsdc2023_*`: GSDC-specific submission and feasibility notes.
- [`product_deliverable/README.md`](product_deliverable/README.md): product
  deliverable packet and related dashboard artifacts.

## Maintenance Rules

- Put the current state in `ppc_current_status.md`; keep `plan.md` as a
  chronological log.
- If a result becomes a durable conclusion, summarize it in `decisions.md`.
- If an artifact is generated, document the important output path in
  [`../experiments/results/README.md`](../experiments/results/README.md).
- Avoid adding long raw command transcripts to top-level docs. Keep them in the
  relevant experiment note or in `plan.md`.
