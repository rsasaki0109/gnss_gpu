# Decision Log

This file records current README-facing design decisions. Exploratory PPC and
GSDC chronology remains under `internal_docs/`.

## D-001: Keep the PLATEAU NLOS Mask CSV Contract

Status: adopted

Decision:

Use the CSV contract below as the shared handoff between ray-traced geometry and
downstream estimators:

```text
tow,epoch_idx,prn,is_los
```

`is_los=0` marks NLOS. Consumers may also use diagnostic columns such as
`nlos_expected_bias_m`, but the first four columns are the stable minimal
contract.

Evidence:

`experiments/run_plateau_nlos_demo_suite.py` exports the mask once and replays it
through SPP, PF, and local-FGO consumers.

| Estimator | Baseline RMS | Mask-soft RMS | RMS gain |
|---|---:|---:|---:|
| SPP | 11.85 m | 4.07 m | 65.6% |
| PF | 11.18 m | 1.40 m | 87.4% |
| FGO | 8.10 m | 0.38 m | 95.4% |

Reasoning:

- The same mask path improves three estimator families, so the interface is not
  overfit to one solver.
- The CSV is simple enough for FGO/SPP/PF replay scripts and compatible with
  existing experiment mask tooling.
- Keeping geometry export separate from solver replay makes the demo auditable:
  downstream scripts can be tested without rerunning ray tracing.

Consequence:

- New NLOS-aware downstream experiments should accept this mask contract before
  adding richer map-specific inputs.
- Richer columns are allowed as diagnostics, but the minimal contract should
  remain readable by consumers that only need LOS/NLOS classification.
