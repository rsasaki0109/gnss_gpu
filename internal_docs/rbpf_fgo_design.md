# RB-FGO-PF: Rao-Blackwellized Particle Filter over Integer-Ambiguity Basins with Factor-Graph Conditionals

Status: DESIGN (2026-07-10). Milestone 2 of the FGO-first roadmap. Author: campaign session.
Prereqs it builds on: the WP13 standalone (graph construction, held-N folding), WP15 GPU batch-LAMBDA (in flight), `particle_filter_device.py` / `dd_likelihood.py` / `particle_ffbsi.py`.

## 1. Why this factorization (grounded in campaign evidence)

Every hard failure of the 19-WP campaign was a **multimodality** failure handled by a
single-hypothesis pipeline:

- **Wrong-fix lock-in** (WP13a/c): a wrong integer basin, once committed+held, is
  self-consistent; ratio tests cannot flag it (wrong-hold streaks pass ratio≥50, WP13o).
- **Residual blindness** (WP13i): multipath corrupts PR and CP together → no residual
  statistic separates good from wrong fixes at commit time.
- **Wrong-basin re-entry** (WP13s): after an outage the float re-enters with 0.5–1.5 m
  error; a gate-free accept locks the wrong basin; tc/ survives only because its float
  re-enters the right basin. Post-tunnel: ours 19.8 % re-fix vs tc/ 66.9 % (WP13r).
- tc/'s answer is **hundreds of hand-tuned knobs** (gates, FSMs, probation ladders) that
  *avoid* committing to the wrong mode. The Bayesian answer is to **carry the modes**.

Conditioned on the integer ambiguities, the RTK/INS problem is (locally) linear-Gaussian.
That is the textbook Rao-Blackwell split:

- **Sampled (discrete):** d_k = (H_k, r_k) — the held-integer assignment
  H_k : (sat,freq) → ℤ over the currently supported DD set, plus regime flags r_k
  (hold-survive vs reset branches). A particle is a *basin lineage* d_{1:k}.
- **Marginalized (Gaussian):** X_k — fixed-lag NavState window (pose, vel, bias) plus
  the float (un-held) ambiguities. Solved per particle by a **fixed-lag factor graph**
  (not an EKF): IMU preintegration + DD-PR + DD-CP(with λN) + NHC/ZUPT/Doppler, with
  H_k folded as constants (exactly WP13s's `_make_ddcp_factor_with_held_n`).

p(d_{1:k}, X_k | y_{1:k}) = p(d_{1:k} | y_{1:k}) · N(X_k; m_k(d), P_k(d))

The discrete space is SMALL (dozens–hundreds of live basins), so N ≈ 64–512 particles —
not the 100K–1M of position-space PFs. RB weights are low-variance by construction.

## 2. Per-epoch cycle

Inputs at epoch k: IMU preintegration Δ_k (shared — independent of d), epoch DD set.

1. **Shared linearization.** Build the epoch's factors ONCE at a reference point x̄
   (previous MAP basin's conditional mean) → normal equations (Λ_k, η_k) with the
   ambiguity block explicit. *(Approximation A1: shared linearization across basins;
   see §5 risk 1.)*
2. **Candidate generation (GPU).** From the reference float + joint marginal Q_N
   (position×ambiguity — the WP13i lesson), run **batch-LAMBDA** for the top-K integer
   candidates (K ≈ 8–32) with their residuals. Add structural branches: slip-release
   candidates, hold-survive vs reset (the WP13q/r probation ladder becomes a *prior over
   branches*, not a hard FSM).
3. **Hypothesis moves (GPU batched).** For each particle i and compatible move m
   (keep H, adopt candidate c, release subset s): conditioning the shared normal
   equations on m's integers changes ONLY the right-hand side (η − Λ_{·N} N). One
   shared factorization + **batched triangular solves** (cuSolver) evaluate all
   (particle × move) conditionals in one launch. A particle's persistent state is just
   (integer vector, weight, small cached mean) — the graph and Schur complement are shared.
4. **Weight update.** log w_i += log p(y_k | d_i', ·) = −½‖r_wht‖² − ½log|2πS| from the
   conditioned system (batched). This is the signal single-hypothesis gates never had:
   a wrong basin that passes any single-epoch ratio bar is killed by **cumulative
   marginal likelihood** over the streak.
   Practical guard: temper the DD-CP term (σ=3 mm makes likelihoods peaky) with an
   annealing exponent β∈[0.3,1] or a Huber-ized likelihood; sweep on run2-3000ep.
5. **Resample + move.** Systematic resampling when ESS < N/2, **stratified by basin
   identity** (never collapse basin diversity prematurely). MCMC rejuvenation: ±1 moves
   on the weakest ambiguity (by conditional variance), batch-evaluated on GPU.
6. **Output / fix decision.** Basin posterior mass γ = Σ_{i ∈ MAP basin} w_i.
   Report FIX iff γ > γ* (0.99) — a *calibrated* fix probability replacing every ratio
   gate; report the MAP basin's conditional mean (scored family) and optionally the
   mixture mean (robust family). smode 4/5 accordingly; honest coverage as always.
7. **Window management.** Marginalize old states in the shared graph; per-particle only
   the integer-dependent RHS statistics survive. On slip/outage per particle: ambiguity
   leaves H (float) or branches per step 2.

## 3. Outage re-entry — the killer app

On CP-resume after an outage: spawn hypotheses = {DDPR-only position prior} ×
{top-K LAMBDA candidates under that prior}. The right and wrong basins BOTH live;
cumulative likelihood resolves them within tens of epochs. This replaces the entire
recovery FSM zoo (recov_cp_hold / sanity_pose_replace / ddpr_recover ladders) with one
mechanism, and directly targets the two measured gaps: post-tunnel re-fix (19.8 % → aim
tc/'s 66.9 %) and the raw_nb==0 mega-gaps.

## 4. Offline smoothing (unique asset)

PPC scoring is offline. Run **FFBSi** (`particle_ffbsi.py`) backward over basin lineages
→ a mixture smoother over (d_{1:K}, X): late evidence retroactively re-weights early
basin choices. Expected to lift `<50cm_full%` beyond the filter output; inuex35 has no
equivalent (1 s fixed-lag only — their structural ceiling, per the benchmark doc).

## 5. Risks and mitigations

1. **Shared-linearization bias** when live basins imply >~1 m position spread →
   cluster basins (small K_c) and refresh linearization per cluster; measure the error
   on run2 first (A1 ablation).
2. **Basin explosion** → dedup hypotheses by H over the active window; stratified
   resampling; cap N with birth prioritized by candidate residual.
3. **Peaky CP likelihood degeneracy** → tempering/robustification (step 4), standard.
4. **Cost** — per epoch: 1 linearization + 1 batch-LAMBDA + O(N·M) batched RHS solves.
   With N≤512, window ≤ 10 states: well within one GPU launch budget; the CPU prototype
   (N≤64) must stay ≥ ~2 ep/s to iterate.

## 6. Validation plan (campaign discipline: full-run confirmation, purity gates)

- **Stage 0 (CPU prototype, N≤64):** run2-3000ep. Gates: (a) false-fix at γ>0.99 ≤
  WP13r's per-run bests; (b) fix rate ≥ WP13r; (c) the known wrong-hold streak epochs
  (WP13o's ratio≥50 cluster) show basin-mass ambiguity instead of confident wrongness.
- **Stage 1:** run1 post-tunnel block (tow 188137 + ~5,000 ep): re-fix % vs 19.8/66.9.
  Canyon (188990–189070) purity as always.
- **Stage 2 (GPU, full 3 runs):** vs Python bests 36.3/51.1/63.3, vs libgnss++
  56.8/80.5/72.8, vs inuex35 56.7/69.9/67.9. run3 judgments on FULL runs only.
- **Ablations:** RB-PF vs plain PF; A1 on/off; K, N, β sweeps; FFBSi on/off.

## 7. Relation to existing code

| Piece | Source |
|---|---|
| Conditional FGO + held-N folding | `repro_tc_fgo/gtsam_rtk_standalone.py` (WP13s `TC_LITERAL` machinery) |
| Batch-LAMBDA GPU | WP15 `src/ar/lambda_batch.cu` (+ extend to top-K output) |
| Batched conditional solves | new kernel, cuSolver batched TRSM/POTRS pattern |
| Weights / resampling | `python/gnss_gpu/particle_filter_device.py`, `dd_likelihood.py` |
| Backward smoothing | `python/gnss_gpu/particle_ffbsi.py` |
