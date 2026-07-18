# WP27 Report — multi-pivot absolute evidence

## Verdict

**Absolute selection and causal integration pass; production/FIX gate not yet passed.**
Multi-pivot DDPR breaks enough of the coherent-translation ambiguity to improve
diagnostic selection from 0/200 to 91/200 epochs when paired with TDCP holdover.
The signal is sparse and not yet calibrated for safe output.

## Delivered

- pivot-invariant, robust, multi-constellation DDPR candidate scores;
- synthetic coverage for pivot, biased-satellite, multi-system, and support
  edge cases;
- reproducible DD-anchor and DD-plus-TDCP replay evaluators;
- a per-epoch best-configuration selection trace;
- an opt-in online diagnostic arm with separately audited DDPR/TDCP sources;
- an executable control-versus-arm neutrality audit;
- an ablation separating clean integrity evidence from the contaminated
  original basin likelihood.

No FGO is used and the production FIX path is unchanged.

## Evidence

On Tokyo run3/200, 40 epochs have raw DDPR anchors and 33 of those contain a
live sub-50 cm basin. The robust multi-pivot score selects 19/33. With TDCP
holdover, the best truth-free-input configuration selects sub-50 cm candidates
on 91/200 epochs, has 0.549 m median error, and improves on single MAP in
180/200 epochs. Single MAP selects 0/200 and has 1.583 m median error.

The selected trajectory contains correct runs of 33 and 30 epochs, but also an
initial wrong run of 75 epochs. Posterior gamma remains deliberately diffuse
and never crosses 0.99. Consequently, this is strong evidence for the WP27
direction but insufficient evidence for a FIX declaration or headline
full-denominator coverage claim.

The causal online arm reproduces 91/200 using 40 raw DDPR anchors and 199 TDCP
intervals. Its operational trajectory is bit-identical to the disabled control
(`cb606f...ce73a7`), all eight audited operational fields have zero mismatch,
the evidence ledger has zero beta errors, and online/replay commit decisions
match at every epoch. This proves integration neutrality, not production
accuracy: the selected integrity candidate is still diagnostic-only.

Artifacts:

- `csv/wp27_multipivot_run3_200_sweep.json`;
- `csv/wp27_integrity_tdcp_run3_200_sweep.json`;
- `csv/wp27_integrity_tdcp_run3_200_best.json`;
- `csv/wp27_integrity_tdcp_run3_200_best_selections.csv`;
- `csv/wp27_online_integrity_run3_200_summary.json`;
- `csv/wp27_online_run3_200_neutrality.json`;
- `csv/wp27_online_integrity_run3_200_evidence.csv`.

Reproduce the enabled run and neutrality audit:

```powershell
python experiments/exp_wp23b_basin_ar.py --run tokyo/run3 --max-epochs 200 `
  --enable-ddpr-respawn --ddpr-respawn-top-k 64 `
  --enable-integrity-lineage --out-diagnostics <epochs.csv> `
  --out-summary <summary.json> --out-trajectory <trajectory.csv> `
  --out-evidence <evidence.csv>

python experiments/audit_wp27_online_arm.py `
  --control-diagnostics <control-epochs.csv> `
  --integrity-diagnostics <epochs.csv> `
  --control-trajectory <control-trajectory.csv> `
  --integrity-trajectory <trajectory.csv> --out-summary <audit.json>
```
