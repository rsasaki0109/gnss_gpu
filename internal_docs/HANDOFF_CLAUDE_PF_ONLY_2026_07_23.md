# HANDOFF → Claude: PF-only RTK stretch campaign — 2026-07-23

This document supersedes the immediate-action section of
`internal_docs/HANDOFF_CODEX_PF_ONLY_2026_07_18.md`. That older handoff remains
useful for the WP21--WP23 architecture history, but production work has advanced
through WP138 and an in-progress WP139 evaluation.

## 0. User objective and non-negotiable invariants

Raise PPC Dataset matched-3D `<50cm_full` to:

- Tokyo run1: at least 81%
- Nagoya run1: at least 86%

while preserving all of the following:

- PF-only production; no runtime FGO.
- Identical truth-free production configuration and candidate selection.
- Full run denominator, never a favorable subset denominator.
- Declared FIX false rate at or below 1%.
- Measurable candidate-supply, posterior-selection, and outage-recovery gates.
- Hash-locked benchmarks and reproducible artifacts.
- Existing M4 artifacts are immutable baseline inputs.

Do not mark the campaign complete until both rate targets and all invariants are
proven from current artifacts. Current rates are still far below the targets.

## 1. Immediate first action

WP139 is complete and rejected; do not rerun or promote it. Begin the next
unassessed all-bad block as WP140:

- Tokyo run1 segment `11220--11275`
- global-scan selected stride phase: 3
- carrier rows: 256
- DDPR rows: 216
- current production sub-50 cm epochs: 0/55

Use WP138 production as the input trajectory. Acquire the fixed 17-sample GSI
cache, then evaluate carrier-reference ranks 0/1/2 at stride phase 3. Proceed to
grid/posterior work only if the frozen absolute gates pass. The broader ranking
is in §10.

## 2. Authoritative current production state

### Tokyo run1

- Production: WP138
- `<50cm_full`: **3521 / 11924 = 29.52868165045287%**
- FIX epochs: 0
- False FIX epochs: 0
- Declared false FIX rate: 0%
- Production trajectory:
  `results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv`
- Trajectory SHA-256:
  `9ABFD989B8CF9978E2A19D87644E898D521B411B9EBB04FAB073E6346AC335A8`
- Benchmark:
  `results/wp31/tokyo_run1_wp138_stability_cppr_full_benchmark.json`
- Benchmark SHA-256:
  `2EF62C743794F673ACAD4AC8A6C8717F055B0AB0B42FDDC8932DFE197EE4BDFD`

### Nagoya run1

- Production: WP100
- `<50cm_full`: **5274 / 7583 = 69.55030990373203%**
- FIX epochs: 0
- False FIX epochs: 0
- Declared false FIX rate: 0%
- Production trajectory:
  `results/wp31/nagoya_run1_wp100_constant_singlebasis_full_trajectory.csv`
- Trajectory SHA-256:
  `CB4F56596C817A22C1D6103222B68D3C9EB75F0FB79BBCC28E6FD541210CC9E8`
- Lock:
  `internal_docs/wp100_pf_constant_singlebasis_benchmark_2026_07_23.json`

### M4 immutable hashes

These must remain exact before every promotion:

- `internal_docs/wp30_m4_production_config.json`
  `66A5FF3F1919C4B0F9ED95A5EFA38865B518C9E03E6FD2652B7A0456A1F89486`
- `internal_docs/wp30_m4_tokyo_evidence_ledger.json`
  `9D756F447304C30B73694225F1CEEA1A82DE864F8D968D449928662582DF098C`

## 3. WP138: current best selector and promotion

WP138 recovered Tokyo `5005--5060`, adding 55 epochs with zero loss.

Candidate supply:

- Initial three-reference pools passed absolute gates but all candidates were
  0/55.
- A coarse grid (3 m radius, 1 m step, 49 cells) also supplied only zero-gain
  modes.
- The existing builder default fine grid (1.5 m radius, 0.5 m step, 49 cells)
  supplied three 55/55 modes.
- Audit fields were completely removed before production selection.

Frozen WP138 selector families:

1. rank-0-to-rank-2 refit disagreement
2. within-basis block spread
3. maximum cross-basis carrier RMS
4. CP/PR rank sum

Each family is dense-ranked. Acceptance requires:

- every winner family rank within the top 40% of eligible modes;
- runner margin at least 20%;
- all inherited WP131 absolute gates, including cross disagreement <=0.1 m,
  carrier RMS <=0.5 cycles, block spread <=0.5 m, checked pairs >=40, and bad
  CP/PR pair fraction <=0.05.

WP138 target candidate 26 had ranks `6/3/10/3` and a 36.36% margin. The unsafe
WP129 holdout tied at the top and exceeded the family-rank limit. Nagoya WP53
abstained because CP/PR evidence was absent.

Key implementation and locks:

- `experiments/select_wp138_stability_cppr_consensus.py`
- `experiments/promote_wp138_stability_cppr_consensus.py`
- `tests/test_wp138_stability_cppr_consensus.py`
- `internal_docs/wp138_tokyo_stability_cppr_validation_2026_07_23.json`
- `internal_docs/wp138_pf_tokyo_stability_cppr_benchmark_2026_07_23.md`
- `results/wp31/tokyo_run1_wp138_stability_cppr_consensus_5005_5060.json`
- `results/wp31/tokyo_run1_wp138_stability_cppr_promotion_5005_5060.json`

Do not weaken WP131, WP133, or WP138 thresholds. Add a separately named and
holdout-validated selector if a later block requires different evidence.

## 4. Completed WP139 posterior rejection: do not rerun

Target segment: Tokyo run1 `4950--5005`.

Global truth-free supply scan:

- selected stride phase: 1
- raw carrier rows: 284
- raw DDPR rows: 189
- current production sub-50 cm epochs in segment: 0/55

GSI height evidence:

- artifact:
  `results/wp31/tokyo_run1_wp139_gsi_multisample17_height_cache_4950_5005.json`
- SHA-256:
  `04ECF68758F5A15A7E070B1910B16B3B60FE5C0D9C0DE945400CAC5C0888090B`
- compatible samples: 17
- inlier samples: 16
- inlier spread: 0.4021060407802537 m
- Up prior center: +4.32964588892758 m

Initial carrier-reference results:

| Rank | Carrier rows | Hypotheses | Strict | Minimum DDPR RMS | Maximum strict audit | Diagnostic DDPR | Diagnostic ceiling |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 284 | 12 | 10 | 1.4010022440 m | 0/55 | 1.4024823784 m | 55/55 |
| 1 | 284 | 12 | 10 | 1.4033168767 m | 0/55 | 1.4009763439 m | 55/55 |
| 2 | 262 | 13 | 12 | 1.3999865598 m | 0/55 | 1.4008299743 m | 55/55 |

Artifacts and hashes:

- rank 0:
  `results/wp31/tokyo_run1_wp139_ref_rank0_phase1_constant_4950_5005_development.json`
  `3734862AAE17F24172DBC141DA1F8174950BF855D80268EFE3A2392D9AE04F49`
- rank 1:
  `results/wp31/tokyo_run1_wp139_ref_rank1_phase1_constant_4950_5005_development.json`
  `2A1FB28877DF14A04B6ECE31FB610DEF6876D80F138D996D55FDD6D9C5FD4C66`
- rank 2:
  `results/wp31/tokyo_run1_wp139_ref_rank2_phase1_constant_4950_5005_development.json`
  `C853B716D11F855375E9E2D25F44EA9A84620B326AE348E3D5476715F779406E`

Interpretation: this is a measured candidate-basin supply miss, not structural
DDPR or block instability. The fixed fine grid has now resolved that supply
miss:

- fine-grid artifact:
  `results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_seed_grid_4950_5005.json`
- fine-grid SHA-256:
  `08C61DF818BB4D27B6F2638EF864E627AFFE6D187430948EE770A65EDAFBE312`
- rank-1 fine-grid development fit:
  `results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_development.json`
- fine-grid fit SHA-256:
  `6A83CE0ADA7B85BF348A64836352BC9A5E1FC23163871CBBAD842600FFCFDC87`
- hypotheses: 61
- strict candidates: 59
- useful diagnostic candidates: 2
- candidate 25: 55/55 diagnostic ceiling
- candidate 32: 46/55 diagnostic ceiling

These audit counts were development diagnostics only and did not enter the
selector. The completed truth-free posterior result is:

- truth-free source SHA-256:
  `FBE17F36A224C412A37A20BC87E56EAA5F8223ABED94088D12A3C7DF29FAB146`
- truth-free pool SHA-256:
  `3AF1D809501985896FAD185CAD860BB9C2D1B90E846D4B70660A8058C6EC38B1`
- rank-0 truth-free refit SHA-256:
  `EC9C08A7EACF1A66CCA00105AA0774D8F2CF1C26999F2D1556934619C58676B1`
- rank-2 truth-free refit SHA-256:
  `00B2EA5C52B811C35B6B197DB24092888BCD06EDB044A8A2A39988A1F501A0B1`
- cross-basis consensus SHA-256:
  `E900197D15E7659CC0FA8189C2CFB7537877CE311E62D49E79B260A4C7941165`
- WP138 selector artifact SHA-256:
  `799B99CBBF91B4FB539C3057143EC5BB498C78053672F5424E3533C394D8BCE7`
- selected winner: candidate 55, diagnostic 0/55
- runner: candidate 24, diagnostic 0/55
- runner margin: 18.75%, below the fixed 20% minimum
- accepted: false

Candidate 25 (55/55 diagnostic) ranked poorly under independent stability
families and is not substituted after audit. The rejection lock is
`internal_docs/wp139_tokyo_stability_cppr_posterior_rejection_2026_07_23.json`.
WP138 production remains unchanged.

## 5. Important recent negative locks

These blocks were measured and should not be rediscovered:

- WP128 `1595--1650`: all hypotheses fail 0.5 m block-spread gate.
- WP129 `5225--5280`: posterior chooses unsafe zero-gain basin; canonical unsafe
  holdout for later selectors.
- WP130 `11660--11715`: original posterior failure, later recovered by WP131.
- WP132 `1650--1705`: supplied useful rank-2 basin, later recovered by WP133.
- WP134 `1485--1540`: structural DDPR floor 6.740--6.821 m.
- WP135 `1210--1265`: structural DDPR floor 9.949--10.002 m.
- WP136 `11330--11385`: scan phase is 3; structural DDPR floor 16.313--16.490 m.
  The phase-0 zero-observation diagnostics are explicitly excluded.

Locks:

- `internal_docs/wp128_tokyo_block_spread_rejection_2026_07_23.json`
- `internal_docs/wp129_tokyo_posterior_rejection_2026_07_23.json`
- `internal_docs/wp130_tokyo_posterior_rejection_2026_07_23.json`
- `internal_docs/wp134_tokyo_ddpr_structural_rejection_2026_07_23.json`
- `internal_docs/wp135_tokyo_ddpr_structural_rejection_2026_07_23.json`
- `internal_docs/wp136_tokyo_ddpr_structural_rejection_2026_07_23.json`

The narrative chain is:
`internal_docs/wp31_tokyo_static_outage_chain_2026_07_19.md`.

## 6. Truth discipline and artifact flow

Development fit artifacts may contain post-selection audit fields. They are not
valid selector inputs until sanitized.

Required flow:

```text
development fit
  -> sanitize_wp55_cppr_candidates.py
  -> truth-free source
  -> build_wp53_cross_basis_seed_pool.py
  -> rank-0/rank-2 cross refits
  -> sanitize both cross refits
  -> select_wp53_cross_basis_consensus.py
  -> select_wp138_stability_cppr_consensus.py
  -> hash-recomputing promoter with two failing holdouts
  -> full-denominator application
  -> truth-only audit after output positions are frozen
```

Never select a diagnostic candidate after viewing `audit_sub50cm_epochs`. Never
substitute the useful runner when a truth-free winner is unsafe. The WP137
development path explicitly rejected such a substitution before WP138 introduced
a separately defined and holdout-tested selector.

## 7. Environment, repository, and dirty-worktree notes

- CWD: `C:\Users\rsasa\Workspace\old\gnss_gpu`
- PPC data used by current commands:
  `E:/datasets/PPC-Dataset-data/tokyo/run1`
- Shell: PowerShell
- Current branch: `agent/wp23b-basin-ar`
- Observed HEAD: `27d0a920c11b31e2879db515bce08681d7a18f57`
- The worktree is intentionally very dirty with many prior campaign edits and
  untracked artifacts. Preserve all unrelated/user changes.
- Do not use `git reset --hard` or `git checkout --`.
- Do not push or open a PR unless the user explicitly asks.
- No runtime FGO is allowed. Post-selection truth audit is allowed only after
  production output is frozen.

README and campaign visualization:

- `README.md` currently reports Tokyo 29.53% and Nagoya 69.55%.
- `docs/assets/figures/pf_only_rtk_stretch.svg`
- current figure SHA-256:
  `81AD4FEA9B075A17EFDEA8547DA5D3422DE5C7B0284F7612132FDA25C0121A4E`
- external latest libgnss++ selected references are shown only as references:
  Tokyo 80.02%, Nagoya 85.84%. They are not PF-only production results.

## 8. Exact WP139 reproduction commands (completed; do not rerun by default)

Run from the repository root.

### 8.1 Build the fixed fine grid — completed; do not rerun

```powershell
python experiments/build_wp75_affine_horizontal_seed_grid.py `
  results/wp31/tokyo_run1_wp139_ref_rank1_phase1_constant_4950_5005_development.json `
  results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv `
  --output results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_seed_grid_4950_5005.json
```

The omitted radius/step flags intentionally use the frozen builder defaults:
1.5 m and 0.5 m, producing 49 cells.

### 8.2 Fit the fine grid under rank 1 — completed; do not rerun

```powershell
python experiments/refine_wp31_moving_block_ambiguity.py `
  --data-dir E:/datasets/PPC-Dataset-data/tokyo/run1 `
  --trajectory results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv `
  --start 4950 --end 5005 --stride 5 --stride-phase 1 `
  --external-seeds results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_seed_grid_4950_5005.json `
  --gsi-height-cache results/wp31/tokyo_run1_wp139_gsi_multisample17_height_cache_4950_5005.json `
  --carrier-reference-rank 1 `
  --output results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_development.json
```

Do not add `--enforce-final-up-prior` to this grid refit. The seeds are already
GSI-Up normalized; this matches the WP137 fine-grid procedure.

### 8.3 Sanitize and cross-refit useful supply — completed

Sanitize and build the pool:

```powershell
python experiments/sanitize_wp55_cppr_candidates.py `
  results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_development.json `
  --output results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_truthfree.json

python experiments/build_wp53_cross_basis_seed_pool.py `
  --source results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_truthfree.json `
  --reference-rank 1 `
  --output results/wp31/tokyo_run1_wp139_rank1_fine_cross_basis_seed_pool_4950_5005.json
```

Cross-refit the same pool under ranks 0 and 2:

```powershell
python experiments/refine_wp31_moving_block_ambiguity.py `
  --data-dir E:/datasets/PPC-Dataset-data/tokyo/run1 `
  --trajectory results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv `
  --start 4950 --end 5005 --stride 5 --stride-phase 1 `
  --external-seeds results/wp31/tokyo_run1_wp139_rank1_fine_cross_basis_seed_pool_4950_5005.json `
  --gsi-height-cache results/wp31/tokyo_run1_wp139_gsi_multisample17_height_cache_4950_5005.json `
  --carrier-reference-rank 0 `
  --output results/wp31/tokyo_run1_wp139_rank1pool_cross_rank0_4950_5005_development.json

python experiments/refine_wp31_moving_block_ambiguity.py `
  --data-dir E:/datasets/PPC-Dataset-data/tokyo/run1 `
  --trajectory results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv `
  --start 4950 --end 5005 --stride 5 --stride-phase 1 `
  --external-seeds results/wp31/tokyo_run1_wp139_rank1_fine_cross_basis_seed_pool_4950_5005.json `
  --gsi-height-cache results/wp31/tokyo_run1_wp139_gsi_multisample17_height_cache_4950_5005.json `
  --carrier-reference-rank 2 `
  --output results/wp31/tokyo_run1_wp139_rank1pool_cross_rank2_4950_5005_development.json
```

Sanitize both cross-refits:

```powershell
python experiments/sanitize_wp55_cppr_candidates.py `
  results/wp31/tokyo_run1_wp139_rank1pool_cross_rank0_4950_5005_development.json `
  --output results/wp31/tokyo_run1_wp139_rank1pool_cross_rank0_4950_5005_truthfree.json

python experiments/sanitize_wp55_cppr_candidates.py `
  results/wp31/tokyo_run1_wp139_rank1pool_cross_rank2_4950_5005_development.json `
  --output results/wp31/tokyo_run1_wp139_rank1pool_cross_rank2_4950_5005_truthfree.json
```

Then run:

```powershell
python experiments/select_wp53_cross_basis_consensus.py `
  --pool results/wp31/tokyo_run1_wp139_rank1_fine_cross_basis_seed_pool_4950_5005.json `
  --source results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_truthfree.json `
  --cross-rank0 results/wp31/tokyo_run1_wp139_rank1pool_cross_rank0_4950_5005_truthfree.json `
  --cross-rank2 results/wp31/tokyo_run1_wp139_rank1pool_cross_rank2_4950_5005_truthfree.json `
  --output results/wp31/tokyo_run1_wp139_rank1_cross_basis_consensus_4950_5005.json

python experiments/select_wp138_stability_cppr_consensus.py `
  --source results/wp31/tokyo_run1_wp139_ref_rank1_constant_fine_gridfit_4950_5005_truthfree.json `
  --cross-basis results/wp31/tokyo_run1_wp139_rank1_cross_basis_consensus_4950_5005.json `
  --output results/wp31/tokyo_run1_wp139_stability_cppr_consensus_4950_5005.json
```

If the selector accepts, audit the selected candidate in the development file
only after the selection is frozen. Then reuse these holdouts under the same
WP138 selector:

- unsafe Tokyo source:
  `results/wp31/tokyo_run1_wp129_ref_rank1_constant_gridfit_5225_5280_truthfree.json`
- unsafe Tokyo cross basis:
  `results/wp31/tokyo_run1_wp131_wp129_cross_basis_consensus_5225_5280.json`
- missing-evidence Nagoya source:
  `results/wp31/nagoya_run1_wp131_wp53_ref_rank1_1436_1656_truthfree.json`
- missing-evidence Nagoya cross basis:
  `results/wp31/nagoya_run1_wp53_cross_basis_consensus_1436_1656_validation.json`

Use `experiments/promote_wp138_stability_cppr_consensus.py` for hash-recomputing
promotion, then apply with `experiments/apply_wp42_moving_block_offset.py` using
the WP138 production trajectory as input. Require gained epochs >0, lost epochs
0, FIX/false FIX unchanged, and exact M4 hashes.

WP139 did not pass, so the promotion paragraph above is reproduction guidance
only.

### 8.4 Actual continuation: start WP140 at 11220--11275

```powershell
python experiments/acquire_wp50_gsi_moving_height_cache.py `
  results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv `
  --calibration-cache results/wp31/tokyo_run1_gsi_height_cache.json `
  --start 11220 --end 11275 --sample-count 17 `
  --output results/wp31/tokyo_run1_wp140_gsi_multisample17_height_cache_11220_11275.json
```

Then run `experiments/refine_wp31_moving_block_ambiguity.py` three times with:

- `--data-dir E:/datasets/PPC-Dataset-data/tokyo/run1`
- `--trajectory results/wp31/tokyo_run1_wp138_stability_cppr_full_trajectory.csv`
- `--start 11220 --end 11275 --stride 5 --stride-phase 3`
- the new WP140 GSI cache
- `--enforce-final-up-prior`
- carrier reference rank 0, 1, then 2

Name outputs
`tokyo_run1_wp140_ref_rank{0,1,2}_phase3_constant_11220_11275_development.json`.
Do not assume phase 0: the global scan explicitly selected phase 3.

## 9. Verification commands

Relevant selector tests and lint were green immediately before this handoff:

```powershell
python -m pytest `
  tests/test_wp53_cross_basis_consensus.py `
  tests/test_wp131_cross_basis_cppr_consensus.py `
  tests/test_wp133_cppr_anchor_consensus.py `
  tests/test_wp138_stability_cppr_consensus.py -q
```

Last result: `11 passed`.

Run Ruff on any modified selector/promoter/application files. Parse all new JSON
locks with `python -m json.tool`. Parse the SVG as XML after changing it.

Before every promotion, independently recompute:

```powershell
Get-FileHash internal_docs/wp30_m4_production_config.json -Algorithm SHA256
Get-FileHash internal_docs/wp30_m4_tokyo_evidence_ledger.json -Algorithm SHA256
```

## 10. Broader next-block order

After WP139, rerank unassessed all-bad 55-epoch blocks against the newest
production trajectory using
`results/wp31/tokyo_run1_wp106_global_evidence_supply_scan.json`. Always honor
each block's `selected_stride_phase`; do not assume phase 0.

The current ranking after excluding assessed blocks is:

1. `11220--11275` (WP140, phase 3, carrier 256, DDPR 216)
2. `5115--5170` (phase 1, carrier 275, DDPR 197)
3. `2530--2585` (phase 0, carrier 239, DDPR 231)
4. `5060--5115` (phase 1, carrier 275, DDPR 192)
5. `4785--4840` (phase 1, carrier 266, DDPR 194)
6. `11165--11220` (phase 3, carrier 263, DDPR 195)
7. `3025--3080` (phase 0, carrier 233, DDPR 225)
8. `7095--7150` (phase 1, carrier 266, DDPR 180)

Recompute this order after every promotion because a promoted segment changes
which blocks remain all-bad. Continue measured negative locks for structural
DDPR/block failures, and reserve new selector design for blocks with demonstrated
useful supply but failed truth-free posterior selection.
