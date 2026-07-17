# WP23b report — PF-only integer-ambiguity basin AR

Date: 2026-07-18. Branch: `agent/wp23b-basin-ar`.

## Verdict

The first measured PF-only RTK FIX target passes on Tokyo run2, epochs 0-1199,
without runtime FGO. The production path starts from pseudorange WLS, runs an
independent DD float KF, generates partial integer candidates with LAMBDA, and
tracks their cumulative posterior in Rao-Blackwellized ambiguity basins.
Ground truth is used only after each output/fix decision for scoring.

| Metric | Result | Target |
| --- | ---: | ---: |
| maximum MAP basin mass gamma | **0.995996** | > 0.99 |
| declared FIX epochs | **14** | > 0 |
| correct / false FIX epochs | **14 / 0** | false <= 1% |
| FixRMS | **0.181 m** | diagnostic |
| `<50cm_full%` | **1.7%** | > 0 |
| AllRMS, scored 1200 epochs | 6.262 m | diagnostic |

This is a first-window result, not a full-dataset accuracy claim. Only 1200 of
9151 run2 rover epochs were evaluated, and G5 multi-run/full-run scale-up is
still pending.

## Design and safety finding

Each basin fixes eight low-variance, generation-tagged DD ambiguities and owns
an ECEF position/velocity KF. Its weight is the cumulative conditional DDPR,
fixed-DDCP, and Doppler marginal likelihood. Candidate birth mass is 0.01;
posterior FIX eligibility requires gamma > 0.99 for three consecutive rover
epochs.

Gamma alone was not trustworthy: the ungated ablation declared 26 epochs, of
which 12 were false (46.15%). The production decision therefore also requires
the MAP basin position to be within 0.5 m of the independent float-KF position.
This truth-free check rejected exactly those 12 gamma-qualified epochs in the
measured window. Ratio tests are not used for FIX decisions.

Gamma calibration counts are stored in
`csv/basin_run2_gamma_calibration.json`. Of 52 epochs in [0.99,1.0), 26 met the
assignment-streak rule, 14 passed independent consistency, and none of the 14
declared fixes was false.

## Gate results

- G1 pass: annealed staged SMC consumes beta=1; run2 AllRMS improved from the
  WP23a diagnostic 20.677 m to 10.742 m.
- G2 pass, limited: float covariance had zero SPD failures; partial 8-variable
  top-16 oracle supply was 62/239 (25.94%), while full-dimensional supply was
  only 3/239.
- G3 pass: basin core and five focused synthetic scenarios implemented.
- G4 pass for the specified run2/1200 production arm. The main runner opt-in is
  `rbpf+dd+ar+gate`.
- G5 partial pass: trusted-DDPR commit and the Tokyo three-run/1200 grid pass;
  full-run scaling and cluster-specific relinearization remain future work.

## Reproduction

```powershell
$env:PYTHONPATH='python'
$env:PYTHONIOENCODING='utf-8'
python experiments/exp_ppc_ctrbpf_fgo.py `
  --methods rbpf+dd+ar+gate --runs tokyo/run2 --max-epochs 1200 `
  --pos-dir results/wp23b/pos/runner --results-prefix wp23b_basin
```

Direct artifact-producing command:

```powershell
python experiments/exp_wp23b_basin_ar.py --max-epochs 1200
python experiments/score_vs_inuex35.py `
  --traj results/wp23b/pos/basin_run2.csv --format csv `
  --city tokyo --run run2
```

Primary artifacts are `csv/basin_run2_epochs.csv`,
`csv/basin_run2_summary.json`, `csv/basin_run2_score.csv`, and
`pos/basin_run2.csv`.

## Verification

The WP21-WP23-relevant regression subset passed: **70 passed, 2 skipped**.
The repository-wide suite completed with 3543 passed, 187 skipped, and 90
failures; those failures are pre-existing environment/artifact failures led by
a missing generated product prediction CSV and third-party `gnssplusplus`
CLI/build dependencies. No focused WP23b test failed.

## G5 addendum — trusted-DDPR commit

The original two-way consistency gate was unsafe outside run2. On Tokyo
run1/1200 it declared 14 fixes, including 4 false fixes (28.57%), because the
Float KF and integer basin drifted coherently together. A carrier-independent
DDPR+Doppler guard KF exposes that failure: correct run1 fixes had MAP-to-guard
separation 1.05-1.56 m, while false fixes had 2.01-2.79 m.

The production commit gate now requires all three navigation arms to agree,
with MAP-to-Float <=0.5 m, MAP-to-DDPR <=1.75 m, most-recent DDPR support >=9
pairs, and DDPR age <=4 rover epochs.

| Run, first 1200 epochs | Baseline FIX/false | Trusted FIX/correct/false | FixRMS [m] | `<50cm_full%` |
| --- | ---: | ---: | ---: | ---: |
| Tokyo run1 | 14 / 4 | **10 / 10 / 0** | 0.211 | 1.652 |
| Tokyo run2 | 14 / 0 | **14 / 14 / 0** | 0.181 | 1.650 |
| Tokyo run3 | 0 / 0 | **0 / 0 / 0** | n/a | 0.333 |

Across declared fixes this is **24/24 correct and 0 false**. The minimum-DD
ablation shows thresholds 0, 6, 9, 12, and 18 produce identical decisions in
these windows; 24 reduces run1 to four correct fixes. Thus `n_dd>=9` is a
support floor, while independent DDPR consistency is the measured safety gain.
The 3600-epoch final grid took 334 seconds, so full-run compute scaling remains
an explicit open issue. After the G5 gate addition, the relevant regression
subset passed with **73 passed, 2 skipped**.
