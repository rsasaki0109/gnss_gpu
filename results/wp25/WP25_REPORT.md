# WP25 Report — temporal ambiguity lineage, increment 1

## Verdict

**Structural partial pass; production gate fail.** Multi-epoch assignment
filtering and deterministic replay work, and top-64 supplies many correct live
basins. The present transition evidence does not reliably select them and its
gamma is inversely calibrated. It is therefore diagnostic-only.

## What was learned

The important result is not the seven to ten recovered positions alone. On the
top-64 run, a sub-50 cm candidate was alive on 165/200 epochs while the standard
MAP selected it zero times. This proves that candidate availability is no
longer the dominant limitation in this window.

The current temporal model uses assignment persistence and the motion of each
basin's own navigation conditional. It selects a correct live basin on at most
10 epochs in the replay sweep, worsens median error to about 4.95 m, and every
`gamma>0.99` decision is wrong. That motion is conditioned by the same
DDPR/DDCP stream that created the basins, so it is not the independent evidence
needed to reject coherent urban multipath.

## Safety decision

- `--enable-temporal-lineage` remains off by default;
- temporal gamma is not connected to FIX or emitted position;
- DDPR sigma is not tightened;
- the trusted path remains exactly neutral over Tokyo run2/200, with 14 correct
  and zero false FIX epochs.

## Reproduction

The truth-free basin trace is
`csv/wp25_temporal_respawn64_run3_200_basins.csv.gz`. Replay:

```powershell
python experiments/eval_wp25_temporal_lineage.py `
  --trace results/wp25/csv/wp25_temporal_respawn64_run3_200_basins.csv.gz `
  --run tokyo/run3 `
  --out-summary results/wp25/csv/wp25_temporal_respawn64_run3_200_sweep.json
```

The trace contains assignments, per-epoch likelihoods, posterior weights,
positions, velocities, birth epochs, and estimator lineage IDs, but no truth or
error field. The evaluator joins reference data only after candidate selection.

## Next gate

WP26 must introduce slip-aware TDCP plus Doppler/IMU relative innovation as a
separately audited transition source. It passes only if correct-live-basin
selection and calibration improve on held-out data without increasing false
FIX. Retuning the current self-conditioned motion model is not justified.
