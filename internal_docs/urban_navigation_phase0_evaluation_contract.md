# Urban-navigation Phase 0 evaluation contract

Phase 0 freezes how experiments are judged before estimator work continues.
The contract is intentionally fail-closed: missing evidence is a failed gate,
not permission to promote.

## Dataset roles

The machine-readable split is
`configs/evaluation/urban_campaign_splits_v1.json`.

- PPC: Nagoya run1 is development, run2 validation, run3 final holdout.
  Tokyo run1 is final-holdout-only because no second durable Tokyo PPC run is
  currently available.
- UrbanNav external: Tokyo Odaiba is development, Tokyo Shinjuku validation,
  and Hong Kong 2019-04-28 is the final external negative control.
- Final-holdout truth may be read only after the candidate and its truth-free
  acceptance decision are frozen.

These campaigns are reported separately. They must not be pooled into one
headline number.

## Mandatory negative holdouts

Every successor selector must supply complete evidence for all four:

| ID | Dataset / epoch | Failure class | Required result |
|---|---|---|---|
| `nagoya_wp53` | Nagoya run1, 1436–1656 | missing independent evidence | abstain |
| `tokyo_wp129` | Tokyo run1, 5225–5280 | wrong basin identity | reject |
| `tokyo_wp156` | Tokyo run1, 10890–10945 | zero-gain unsafe acceptance | reject |
| `tokyo_wp168` | Tokyo run1, 1320–1375 | screened zero-gain unsafe acceptance | reject |

The registry pins the lock schema and SHA-256 of each historical record. M4
production config and ledger hashes are pinned by the same contract.

## Failure taxonomy

`gnss_gpu.evaluation_contract.FailureCategory` defines:

- observation NLOS/multipath;
- basin identity;
- offset/drift model;
- evidence thinning;
- outage/reacquisition;
- misleading map constraint;
- missing evidence;
- unsafe acceptance;
- runtime/resource;
- data integrity;
- unknown.

New campaigns report counts for every category. `unknown` is allowed for
triage but is not a substitute for root-cause classification at a promotion
review.

## Common KPI and promotion gates

The normalized candidate summary records total and sub-50 cm epochs,
false-FIX, gained/lost epochs, P50/P95 error, longest contiguous failure,
P50/P95 epoch latency, and peak GPU memory.

Promotion requires:

1. no truth in production input;
2. gained epochs greater than zero and lost epochs equal to zero;
3. false-FIX equal to zero;
4. all four negative holdouts complete and safely rejected/abstained;
5. exact historical lock and M4 hashes;
6. a valid reproducibility manifest with every input hash;
7. every Phase 0 KPI present.

Run:

```text
python experiments/evaluate_urban_campaign.py \
  --input path/to/campaign_input.json \
  --output path/to/evaluation_result.json
```

Exit code 0 means every gate passed. Exit code 2 means fail-closed. The result
contains each gate and reason so CI and later promotion tooling can consume it.

The manifest captures input paths, sizes and SHA-256, canonical config hash,
exact command, Python/platform details, Git commit, and tracked-worktree state.
It does not include wall-clock time, so identical inputs/config produce a
stable content hash.
