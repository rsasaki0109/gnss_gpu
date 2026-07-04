# Fable5 response — NLOS Wave 2 pivot (2026-07-04)

Model: `claude-fable-5-thinking-high`  
Request: `internal_docs/fable5_nlos_wave2_advice_request_2026_07_04.md`

## Executive verdict

Wave 1 closure and PR #117 merge were **correct**. The Wave 2 6-pool regression is **not evidence against the ranker hypothesis** — it is confounded by a weak 6-candidate pool plus missing production emit guards (`emit-max-diff-m 0.4`, `max-to-hybrid-m 0`), which forced rtkdiag to emit on 100% of epochs vs ~65% in Phase 33 production.

Do **not** invest days in the 50+ pool rebuild for PPC gain alone. Run two cheap gate experiments first. Treat any pool rebuild as primarily **production-replay restoration**; refreshed-mask validation is secondary.

## Q1–Q6 answers

| Q | Verdict |
|---|---------|
| Q1 Wave 1 closure | **CONFIRMED** — merge was correct |
| Q2 PF-layer retention | **KEEP in mainline, frozen** behind `--pf-nlos-preset` |
| Q3 Pivot path | **B** (gate experiments now), gate to **A** only if oracle shows headroom |
| Q4 Validation target | **NO** — +1.07 pp already banked; use parity replay (n/r2 ≈ 64.08% OFFICIAL full-run) |
| Q5 Minimum smoke | Config-parity re-smoke + oracle ceiling on n/r2 (no new RTK) |
| Q6 Do-not | **CONFIRMED** + no v6/v7 ranker expansion, no k<99 on n/r2, no w2pool overwriting production CSV names |

## Composer actions (ordered)

1. De-collide artifacts → `*_v5_nlos_w2pool.csv` for Wave 2 train outputs
2. Config-parity re-smoke → add Phase 33 emit guards to `wave2-smoke`
3. Oracle ceiling script → best-of-{hybrid, w2_*, pf_bridge} on n/r2 window

## PR #118

**Keep draft** until gate experiments complete. Merge as tooling + documented negative (like #117) if Wave 2 closes, or after parity replay if pool build proceeds.

## Risk flags

- Window smokes (epoch 1000–1200) ≠ full-run OFFICIAL metrics
- Mask provenance drift (geoid constant vs May originals)
- Phase 10/19 pool may not reproduce 64.08% exactly (±0.1 pp tolerance)
- **Active hazard:** wave2-train may have overwritten production `selector_ranker_predictions_v5_nlos.csv`
- Ranker layer saturated (Phases 34–36); expected new upside ≤ ~0.2 pp

## Gate experiment results (Composer, 2026-07-04)

Following Fable action items 1–3:

| Experiment | Result |
|------------|--------|
| Artifact de-collision | Wave 2 outputs renamed to `*_v5_nlos_w2pool.*` (production-named files were overwritten by wave2-train) |
| Config-parity re-smoke (Phase 33 emit guards) | nagoya/run2 honest **2.31%** unchanged; rtkdiag PU still **1191/1191** |
| Oracle ceiling (4853 hybrid-covered tows) | hybrid segment **14.77%**, oracle **14.77%**, **headroom 0.0 pp** (hybrid wins 4844/4853) |

**Gate verdict:** Option A (50+ pool rebuild) **not justified for PPC gain** from this 6-variant bootstrap. Pool has zero oracle headroom vs hybrid on nagoya/run2. Wave 2 closes as documented negative (like Wave 1).

**PR #118:** Ready to merge as tooling + gate evidence. Do not pursue ranker+NLOS on 6-pool further.
