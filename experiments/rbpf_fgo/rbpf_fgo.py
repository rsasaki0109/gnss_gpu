"""WP16 (TASK_M22): RB-FGO-PF Stage-0 CPU prototype.

Rao-Blackwellized particle filter over integer-ambiguity BASINS with the
WP13 standalone fixed-lag FGO as the (shared) Gaussian conditional --
see `gnss_gpu/internal_docs/rbpf_fgo_design.md` (the spec) and
TASK_M22.md (this milestone's gates).

Architecture (Stage-0 simplifications are declared in WP16_REPORT.md):

- The SHARED filter is the existing `GtsamRtkTc` (WP13r Q4 stack) run
  with its fix-and-hold machinery neutralized (HOLD_RATIO ~ inf): the
  graph stays a PURE-FLOAT fixed-lag FGO, so the joint marginal it
  exports each epoch (`_write_back_tc`: nav.x / nav.P over position x
  SD-ambiguities) is uncontaminated float evidence. This realizes the
  design's step-1 "shared linearization" at the float mean (declared
  approximation A1': the design linearizes at the previous MAP basin's
  conditional mean; Stage 0 has no per-basin graph feedback).

- A PARTICLE is a basin lineage: per-(sys,freq) group of relative
  integers {sat -> z} (one gauge DOF per group, canonicalized to the
  lowest sat id), plus a cumulative log-weight. DD(ref,j) = z[ref]-z[j]
  for any pair inside the group, so reference-satellite changes cost
  nothing (re-anchoring is exact).

- CANDIDATES come from cssrlib mlambda top-K (`ncands=K, parmode=1`,
  full ILS) on the DD float (y, Qb) sliced from the shared posterior --
  the same joint (position x ambiguity) marginal `_do_ar` uses. The
  interface (`_lambda_topk`) takes (y, Qb, K) and returns integer
  candidate columns, so WP15's GPU batch-LAMBDA can swap in later.

- The per-particle CONDITIONAL solve is exact Gaussian conditioning of
  the shared joint posterior on the particle's held DD integers (the
  dense-CPU equivalent of the design's step-3 "conditioning the shared
  normal equations changes only the RHS"): fixed position
  x|z = x_float - Qab_H Qb_HH^-1 (y_H - z_H).

- The WEIGHT increment (design step 4) is a genuine per-epoch predictive
  likelihood, NOT the posterior z-score: this epoch's raw DD-PR and
  DD-CP innovations (recomputed from the stashed epoch observations at
  the float antenna position) are scored under each hypothesis with the
  3-DOF position offset marginalized against the FGO position marginal:

    CP row in H      : v = lam*(b_meas - z_dd),   sigma = s_cp/sqrt(beta)
    CP row not in H  : v = lam*(b_meas - y_dd),
                       sigma = sqrt(s_cp^2 + lam^2 (Qb_dd + 1/12))/sqrt(beta)
    PR row (always)  : v = dd_pr - dd_range(x),   sigma = s_pr
    logL = log Integral N(v; J dx, S) N(dx; 0, Ppos) d dx   (closed form)

  The lam^2/12 lattice floor on un-held rows makes the empty/partial
  hypothesis the honest composite "some other integer" alternative, so
  the basin posterior mass gamma stays calibrated (without it the empty
  particle scores as well as the true basin once the float converges and
  gamma could never reach the fix gate). beta in [0.3,1] tempers the
  peaky CP likelihood (design step 4 practical guard).

- MOVES per particle per epoch (design step 2/3 structural branches):
  survive (keep H, maintenance drops only), extend (conditional-rounding
  bootstrap of the un-held support given H), adopt-k (jump to LAMBDA
  candidate k = hold-reset branch), release (drop to empty = float
  lineage). The proposal is FULLY ADAPTED: each move's tempered
  likelihood is evaluated, the particle samples a move proportional to
  prior x likelihood and its weight gains log sum_m prior_m L_m -- the
  optimal proposal for a finite move set (low-variance RB weights).

- RESAMPLING: systematic, particles sorted by basin id (stratified-by-
  basin: each basin keeps floor/ceil of proportional share), when
  ESS < N/2.

- FIX decision (design step 6): gamma = MAP basin's posterior mass;
  report FIX iff gamma > gamma* (0.99) and the basin covers >= min_nb
  DD pairs this epoch. Output = MAP basin's conditional mean.

Slip/reset coupling: every `_release_ambiguity` call in the shared
filter (slip / MW / CMC / outage / FDE / reset ladder) drops that
(sat,f) from every particle's basin the same epoch; a full graph re-seed
(`_seed_phase2_graph`) clears all basins. Sats unseen for > maxout
epochs age out of basins (tc/'s outage clear_hold analogue).
"""

from __future__ import annotations

import math
import os
import time
from collections import OrderedDict

import numpy as np

from cssrlib.gnss import sat2prn, uGNSS, geodist, time2gpst
from cssrlib.mlambda import LambdaError

from gtsam_rtk_standalone import (
    GtsamRtkTc, _wp13m_finite_guarded_mlambda, _get_wavelengths, build_pim)

_LOG2PI = math.log(2.0 * math.pi)
# WP44: agreement-mass thresholds of the per-epoch `nb_at` diagnostic
NB_AT_THRESH = (0.5, 0.8, 0.9, 0.95, 0.98, 0.99)


# --------------------------------------------------------------------------
# config
# --------------------------------------------------------------------------

def _env_f(name, default):
    return float(os.environ.get(name, default))


def _env_i(name, default):
    return int(float(os.environ.get(name, default)))


class RbpfConfig:
    """All knobs env-overridable (RBPF_*); defaults = the WP16 spec."""

    def __init__(self):
        self.n_particles = _env_i('RBPF_N', 64)
        self.topk = _env_i('RBPF_K', 12)
        self.beta = _env_f('RBPF_BETA', 0.5)
        self.gamma_fix = _env_f('RBPF_GAMMA', 0.99)
        self.min_fix_nb = _env_i('RBPF_MIN_FIX_NB', 5)
        self.ess_frac = _env_f('RBPF_ESS_FRAC', 0.5)
        self.maxout = _env_i('RBPF_MAXOUT', 5)      # tc/ slip_reset_maxout
        self.seed = _env_i('RBPF_SEED', 20260710)
        # move priors (renormalized over the moves actually available):
        # survive / extend (conditional-rounding growth) / adopt-consensus
        # (top-K agreement subset -- the partial-AR analogue; full-vector
        # ILS candidates are almost always partially wrong at nb 30-45 on
        # urban data, so a subset entry move is what lets a basin lineage
        # get a foothold) / adopt-full-k / shrink-worst (release subset,
        # design step-3 move class) / release (full float).
        self.p_survive = _env_f('RBPF_P_SURVIVE', 0.62)
        self.p_extend = _env_f('RBPF_P_EXTEND', 0.12)
        self.p_adopt_cons = _env_f('RBPF_P_ADOPT_CONS', 0.12)
        self.p_adopt = _env_f('RBPF_P_ADOPT', 0.08)
        self.p_shrink = _env_f('RBPF_P_SHRINK', 0.04)
        self.p_release = _env_f('RBPF_P_RELEASE', 0.02)
        # Robust likelihood (design step-4 practical guard). Two layers,
        # both measured necessary on the run2 smoke (the population
        # otherwise collapses to the empty basin):
        # 1) IRLS-Huber position solve (huber_c): one multipath row must
        #    not drag the 3-DOF position offset.
        # 2) Per-row two-component mixture noise: (1-eps) N(0,sigma^2) +
        #    eps N(0,sigma_mp^2). During multipath bursts the float sits
        #    0.1-0.5 cycles off the lattice; at sigma_cp ~ 4mm that costs
        #    a held hypothesis 15-90 nats/row even Huber-capped, so every
        #    basin died vs the marginalized (empty) hypothesis. The
        #    mixture bounds a burst row's penalty at ~ln(1/eps) while a
        #    clean row keeps its ~2.5 nat advantage -- basins survive
        #    bursts (mass splits) and win clean stretches, which is the
        #    multimodality behavior gate 3 asks to demonstrate.
        self.huber_c = _env_f('RBPF_HUBER_C', 3.0)
        self.mix_eps = _env_f('RBPF_MIX_EPS', 0.10)
        self.mix_sigma_cp = _env_f('RBPF_MIX_SIGMA_CP', 0.05)   # m
        self.mix_sigma_pr = _env_f('RBPF_MIX_SIGMA_PR', 10.0)   # m
        # Single-epoch DD-CP innovation noise floor [m]: the varerr sigma
        # (~3.6mm) is the FACTOR sigma; measured single-epoch innovations
        # b-y run at ~6mm median (residual atmosphere/small multipath),
        # so without a floor the clean-row advantage of a correct basin
        # flips sign epoch-to-epoch and basins cannot accumulate.
        self.sig_cp_floor = _env_f('RBPF_SIG_CP_FLOOR', 0.008)
        # Per-(sys,freq)-group common-offset nuisance [m] for held CP
        # rows: reference-satellite carrier multipath shifts EVERY DD row
        # of the group coherently (~1-3cm here); without this nuisance a
        # CORRECT basin pays ~2-4 nats/row and can never out-score the
        # flat-lattice hypothesis (measured: ll_cons 5-40 nats below
        # ll_empty on the very stretch WP13r fixes 57-69%). A wrong
        # common INTEGER (19cm) stays ~6-9 sigma under this prior.
        self.sig_group = _env_f('RBPF_SIG_GRP', 0.02)
        # Position-prior inflation [m] for the per-epoch LS: the graph's
        # position covariance is overconfident (claims cm while the float
        # errs by dm), which pinned the LS offset and made even the
        # GROUND-TRUTH basin score 15-200 nats below empty (measured,
        # RBPF_DBG4): every held row carried the un-absorbable float
        # position error. The epoch's own PR + held-CP rows determine the
        # 3-DOF offset; the prior only has to keep far-away wrong basins
        # from teleporting the position.
        self.sig_pos_floor = _env_f('RBPF_SIG_POS_FLOOR', 1.0)
        # Per-epoch burst regime (prior p_burst): "this epoch's CP is
        # untrustworthy" -- all CP rows fall back to the flat-lattice
        # density for EVERY hypothesis, so during a full multipath burst
        # the empty hypothesis stops out-scoring held basins by
        # ~1.6 nats/row/epoch and basins coast on the survive prior
        # (N persists through bursts unless a slip fires -- and slips
        # drop the component explicitly). Between-basin mass is frozen
        # during bursts: exactly the split-mass multimodality behavior
        # gate 3 requires.
        self.p_burst = _env_f('RBPF_P_BURST', 0.10)
        # Un-held CP rows: 1 = flat-over-lattice marginal (density 1/lam
        # per metre, no position info). The earlier Gaussian plug-in
        # (score vs the POST-FIT float y) let the empty hypothesis cheat:
        # y had already absorbed this epoch's CP, so its "residual" was a
        # post-fit ~0 while held basins faced genuine innovations -- the
        # population collapsed to the empty basin even on clean
        # stretches (measured, main_r2 first attempt). 0 = the old
        # Gaussian marginal (kept for A/B).
        self.marg_uniform = _env_i('RBPF_MARG_UNIFORM', 1)
        # numerical guards
        self.qb_jitter = _env_f('RBPF_QB_JITTER', 1e-8)
        # ------------------------------------------------------------------
        # WP17 Stage-1 knobs (all default OFF = WP16 Stage-0 bit-identical)
        # ------------------------------------------------------------------
        # 1) Basin feedback (TASK_M23 item 1): fold gamma-gated consensus
        #    DD integers back into the shared graph as conditioned holds
        #    (the WP13r COND_HOLD machinery; needs COND_HOLD=1 in the
        #    shared-filter env so the held mirror/_ar_eligible exclusion
        #    are active). A pair is committed after its per-DD agreement
        #    mass has been > fb_gamma for fb_m CONSECUTIVE epochs;
        #    release is driven by the PF itself: agreement mass on the
        #    HELD integer dropping below fb_rel_mass for fb_rel_m
        #    consecutive epochs (no heuristic ladders), or the (sat,f)
        #    ageing out of the PF support (maxout).
        self.fb_enable = _env_i('RBPF_FB', 0)
        self.fb_m = _env_i('RBPF_FB_M', 3)
        self.fb_gamma = _env_f('RBPF_FB_GAMMA', self.gamma_fix)
        self.fb_rel_mass = _env_f('RBPF_FB_REL_MASS', 0.5)
        self.fb_rel_m = _env_i('RBPF_FB_REL_M', 3)
        # min pairs agreeing this epoch before ANY commit is considered
        # (keeps tiny partial sets from pinning the graph).
        self.fb_min_nb = _env_i('RBPF_FB_MIN_NB', self.min_fix_nb)
        # 2) Post-outage re-entry spawning (design section 3): after
        #    spawn_outage consecutive epochs without a usable CP problem,
        #    the next spawn_epochs epochs add 'spawn' moves built from a
        #    DDPR-only position prior x top-K LAMBDA candidates under
        #    that prior (the float posterior, wrong after re-entry, is
        #    bypassed for candidate generation).
        self.spawn_enable = _env_i('RBPF_SPAWN', 0)
        self.spawn_outage = _env_i('RBPF_SPAWN_OUTAGE', 15)
        self.spawn_epochs = _env_i('RBPF_SPAWN_EPOCHS', 25)
        self.spawn_sig_pos = _env_f('RBPF_SPAWN_SIG_POS', 30.0)
        self.spawn_min_cp = _env_i('RBPF_SPAWN_MIN_CP', 5)
        self.p_spawn = _env_f('RBPF_P_SPAWN', 0.12)
        # extra spawn triggers (the run1 post-tunnel data shows the
        # re-entry regime is NOT a clean no-CP outage: 24-39 s tow gaps
        # interleave with 4-9-pair epochs while the float is metres
        # wrong for minutes):
        #  - a raw data/time gap > spawn_gap_s arms a full spawn window;
        #  - a rolling "desperation" spawn while the PF has been fixless
        #    for >= spawn_after consecutive stepped epochs (one spawn
        #    per epoch until a basin takes hold; ~1 extra mlambda/epoch).
        self.spawn_gap_s = _env_f('RBPF_SPAWN_GAP_S', 2.0)
        self.spawn_after = _env_i('RBPF_SPAWN_AFTER', 25)
        # Only spawn when the DDPR-only position DISAGREES with the
        # float by more than this [m] -- the actual re-entry signature
        # (float metres wrong, PR still sane). Without it the
        # desperation trigger spawned DDPR-anchored candidates inside
        # the canyon (float 0.4-0.9 m, DDPR multipath-corrupted) and
        # produced two wrong-basin fix clusters (12 false fixes in the
        # canyon window, tow 189003-189014, full_r1 pre-gate run). When
        # float and DDPR agree, posterior-anchored candidates already
        # cover the same basins and spawning adds only risk.
        self.spawn_min_dpos = _env_f('RBPF_SPAWN_MIN_DPOS', 2.0)
        # Fix-output DDPR cross-check (WP17): reject a Q=4 report whose
        # LS position sits > fix_max_dpr metres from THIS epoch's
        # DDPR-only robust position (0 = off). Catches single-epoch
        # LS blow-ups (measured: one 7.08 m fix at tow 177263 on run2
        # with the float at 0.12 m and correct neighbors) WITHOUT
        # re-anchoring on the float — after an outage the float is
        # metres wrong while a correct spawned fix is NEAR the DDPR
        # position, so a float-distance guard would kill re-entry and
        # this one does not. Same spirit as WP13i's ambiguity-
        # independent DD-PR cross-validation, applied to the PF output.
        self.fix_max_dpr = _env_f('RBPF_FIX_MAX_DPR', 5.0)
        # 3) WP15 GPU batch-LAMBDA drop-in at _lambda_topk (RBPF_CUDA=1;
        #    bit-identical kernel, CPU fallback on any error).
        self.use_cuda = _env_i('RBPF_CUDA', 0)
        # ------------------------------------------------------------------
        # WP18 Stage-2 knobs (all default OFF = WP17 Stage-1 bit-identical)
        # ------------------------------------------------------------------
        # (b) 7.08m single-epoch LS blow-up (WP17 caveat 2): OUTPUT-layer
        #     conditioning guard. The blow-up epoch's geometry was
        #     degenerate (the DDPR cross-check shared the same rows and
        #     landed near the same wrong point, evading the position
        #     comparison), so the guard is on the LS NORMAL MATRIX
        #     instead: reject a Q=4 report whose data-only position
        #     information (Schur complement over the group nuisances,
        #     prior removed) has a min eigenvalue < fix_min_info [1/m^2].
        #     Reporting only -- basin/feedback state untouched.
        self.fix_min_info = _env_f('RBPF_FIX_MIN_INFO', 0.0)
        # (a) coherent-shift wrong-basin lock-in (WP17 caveat 1): the
        #     position-marginalized likelihood keeps a coherently-shifted
        #     basin+held-mirror competitive, so the mass-drop release
        #     never fires. The DDPR-only position (code rows, ambiguity-
        #     independent) does NOT share the shift -- two uses:
        #     - commit gate: skip committing NEW holds on an epoch whose
        #       reported position sits > fb_commit_max_dpr [m] from the
        #       DDPR position (streaks keep accruing; commit happens on
        #       the first clean epoch). "Holds committed during a float
        #       excursion" was the lock-in entry mechanism.
        #     - hold release: reported position (fix output when fixing,
        #       else the held-pinned float) > fb_ddpr_rel_d [m] from the
        #       DDPR position for fb_ddpr_rel_m CONSECUTIVE trustworthy-
        #       DDPR epochs releases ALL conditioned holds (each release
        #       also drops that (sat,f) from every particle) and arms the
        #       re-entry spawner. 0 = off.
        self.fb_commit_max_dpr = _env_f('RBPF_FB_COMMIT_MAX_DPR', 0.0)
        self.fb_ddpr_rel_d = _env_f('RBPF_FB_DDPR_REL_D', 0.0)
        self.fb_ddpr_rel_m = _env_i('RBPF_FB_DDPR_REL_M', 5)
        # DDPR trustworthiness for the (a) signals: sqrt(max eigenvalue
        # of the DDPR LS posterior covariance) must be < ddpr_qual_max
        # [m] for the epoch to count toward the release streak / gate
        # (a multipath-corrupted or degenerate DDPR must not release
        # good holds -- the canyon regime).
        self.ddpr_qual_max = _env_f('RBPF_DDPR_QUAL_MAX', 1.0)
        # (a) challenger spawns: after chal_m CONSECUTIVE reported-fix
        #     epochs whose trusted-DDPR distance exceeds chal_d [m],
        #     force a spawn (DDPR-anchored candidates) with the
        #     spawn_min_dpos gate bypassed -- during the measured lock
        #     the float-vs-DDPR distance sat at ~1.0 m (med), UNDER the
        #     2 m re-entry gate, so the ordinary spawner never anchored
        #     candidates there; a reported fix also puts the spawner to
        #     sleep, which the challenger deliberately ignores. The
        #     candidates enter as normal spawn moves and must WIN the
        #     tempered likelihood to matter (non-destructive). 0 = off.
        self.chal_d = _env_f('RBPF_CHAL_D', 0.0)
        self.chal_m = _env_i('RBPF_CHAL_M', 3)
        # (b) three-way consistency vote (float / DDPR / fix): reject a
        #     Q=4 report when float and DDPR disagree by > fix_vote_dd
        #     [m] (geometry corrupt -- the 7.08 m epoch measured 10.9)
        #     AND the fix agrees with NEITHER (fix-vs-DDPR >
        #     fix_vote_dpr [m]); a correct post-outage re-entry fix is
        #     NEAR the DDPR position, so it survives (measured: 0 good
        #     fixes hit on run1-8500 incl. post-tunnel + run2-6400 at
        #     6.0/3.5). 0 = off.
        self.fix_vote_dd = _env_f('RBPF_FIX_VOTE_DD', 0.0)
        self.fix_vote_dpr = _env_f('RBPF_FIX_VOTE_DPR', 3.5)
        # WP19 experimental A1-off approximation: retain the previous
        # conditional position of the highest-mass basins and use it as
        # that basin's prior centre on the next epoch. Default-off keeps
        # the WP18 ship path bit-identical.
        self.cluster_relin = _env_i('RBPF_CLUSTER_RELIN', 0)
        self.cluster_relin_top = _env_i('RBPF_CLUSTER_RELIN_TOP', 2)
        self.cluster_parent_min = _env_i('RBPF_CLUSTER_PARENT_MIN', 3)
        self.cluster_parent_frac = _env_f('RBPF_CLUSTER_PARENT_FRAC', 0.8)
        self.cluster_sig_floor = _env_f('RBPF_CLUSTER_SIG_FLOOR', 0.5)
        # Rejuvenate a conditioned graph when its winning basin loses
        # enough independent DD support for several epochs. This is a
        # structural integrity signal (not a position/truth threshold).
        self.fb_support_min = _env_i('RBPF_FB_SUPPORT_MIN', 0)
        self.fb_support_m = _env_i('RBPF_FB_SUPPORT_M', 3)
        # Dual-evidence graph mutation: shadow posterior components may
        # commit only when the independently generated shared-graph top-K
        # candidates unanimously support the same DD integer.
        self.fb_dual_enable = _env_i('RBPF_FB_DUAL', 0)
        self.fb_dual_min = _env_i('RBPF_FB_DUAL_MIN', 12)
        self.fb_witness_pos = _env_i('RBPF_FB_WITNESS_POS', 0)
        self.fb_witness_chi = _env_f('RBPF_FB_WITNESS_CHI', 3.0)
        self.fb_witness_sig_floor = _env_f('RBPF_FB_WITNESS_SIG_FLOOR', 0.25)
        # Output-only leave-one-DD-out carrier consistency diagnostic.
        # Each fixed DD is predicted from a conditional solution using the
        # other fixed DDs; no score is fed into the posterior or graph.
        self.loo_enable = _env_i('RBPF_LOO', 0)
        # Output-only delayed stability of the ambiguity-induced position
        # correction.  At 5 Hz, 25/100 stepped epochs are about 5/20 s.
        self.delayed_enable = _env_i('RBPF_DELAYED', 0)
        self.delayed_short = _env_i('RBPF_DELAYED_SHORT', 25)
        self.delayed_long = _env_i('RBPF_DELAYED_LONG', 100)
        # Experimental PF rejuvenation: a delayed-drift alarm forces
        # DDPR-anchored challenger candidates for this many future epochs.
        # Zero preserves the diagnostic-only WP32 behavior.
        self.delayed_rejuv_epochs = _env_i('RBPF_DELAYED_REJUV', 0)
        self.delayed_rejuv_thresh = _env_f('RBPF_DELAYED_REJUV_D', 0.5)
        self.delayed_rejuv_edge = _env_i('RBPF_DELAYED_REJUV_EDGE', 0)
        self.delayed_rejuv_reset = _env_f('RBPF_DELAYED_REJUV_RESET', 0.4)
        self.delayed_rejuv_cooldown = _env_i('RBPF_DELAYED_REJUV_COOLDOWN', 100)
        # 0 disables this gate.  A positive value requires the DDPR anchor
        # covariance quality to be below the limit before mutating the PF.
        self.delayed_rejuv_qual_max = _env_f('RBPF_DELAYED_REJUV_QUAL_MAX', 0.0)
        # Stage edge-triggered challengers outside the production population
        # and promote only after cumulative future predictive evidence wins.
        self.shadow_epochs = _env_i('RBPF_SHADOW_EPOCHS', 0)
        self.shadow_margin = _env_f('RBPF_SHADOW_MARGIN', 0.0)
        # ------------------------------------------------------------------
        # WP43 per-particle position tracking (default OFF = WP18 ship path
        # bit-identical). Each particle lineage carries its own Gaussian
        # belief of the antenna position, predicted between epochs with
        # the shared graph's IMU motion increment and updated with the
        # epoch's DD rows under the particle's own integers. (The WP19
        # cluster prior lacked the motion prediction: it centred the next
        # epoch on the previous absolute position.)
        #  2 = shadow (shipped with RBPF_TRACK_OUT=2): beliefs are updated
        #      under the integers the WP18 moves picked; weights untouched.
        #  1 = experimental, not adopted: the weight increment becomes the
        #      predictive likelihood against the belief, so every particle
        #      x move needs its own solve (one batched `_loglik_batch`
        #      call). Raised the fix rate but locked wrong basins (WP43).
        self.track = _env_i('RBPF_TRACK', 0)
        # random-walk sigma [m per graph epoch] of the IMU-predicted
        # motion increment (float-velocity error over 0.2 s)
        self.track_q = _env_f('RBPF_TRACK_Q', 0.02)
        # floor [m] on the tracked posterior sigma (unmodelled multipath)
        self.track_floor = _env_f('RBPF_TRACK_FLOOR', 0.01)
        # adopt / spawn moves: 1 = keep the lineage's position belief as
        # the prior of the new integers; 0 = start from the loose prior
        self.track_inherit = _env_i('RBPF_TRACK_INHERIT', 1)
        # empty (float) particles: 1 = track a code-only belief; 0 = drop
        # their belief so they stay loose explorers
        self.track_empty = _env_i('RBPF_TRACK_EMPTY', 0)
        # drop every belief after a data gap longer than this [s]
        self.track_gap_s = _env_f('RBPF_TRACK_GAP_S', 2.0)
        # WP45 (RBPF_TRACK=1 only): for this long after a belief reset
        # (PIM break / data gap, e.g. a tunnel) every move is scored
        # against the loose float prior, as with RBPF_TRACK=2, while the
        # beliefs keep updating. Without it the first basin adopted after
        # the reset builds a cm belief and out-scores a later fresh
        # adoption by ~3 ln(1 m / 1 cm) nats per epoch (WP43: a 0.72 m
        # wrong basin locked after the run1 tunnel). 0 = off.
        self.track_probation_s = _env_f('RBPF_TRACK_PROBATION_S', 0.0)
        # Reported position: 0 = WP18 (FIX: epoch-local LS, FLOAT: shared
        # float); 1 = FIX only, mass-weighted tracked belief of the
        # particles carrying every reported integer; 2 = every epoch whose
        # MAP basin is non-empty reports that basin's mass-weighted
        # tracked belief (FIX/FLOAT labels unchanged). Output only: the
        # weights, holds and FIX decisions never see it.
        self.track_out = _env_i('RBPF_TRACK_OUT', 0)
        # extra shadow-tracking variants run side by side with the main
        # one (RBPF_TRACK=2 only; weights are untouched so they share one
        # filter): "q:floor:empty,q:floor:empty,..." -> per-epoch MAP
        # beliefs recorded for each (variant 0 = the main knobs above)
        # ------------------------------------------------------------------
        # WP44 basin persistence (default OFF = WP43 bit-identical). The PF
        # used to drop a (sat,f) from every basin whenever the shared graph
        # re-initialised its float ambiguity (graph re-seed after solve
        # failures, outage age-out after 5 graph epochs, any fresh key).
        # None of those is a carrier slip: with the receiver still locked,
        # the DD integer is unchanged and the per-epoch likelihood (raw DD
        # carrier vs the held integer) keeps testing it. With RBPF_KEEP=1
        # only releases whose reason is in keep_drop (slip / MW / CMC /
        # FDE-on-carrier) drop the integer, a graph re-seed keeps the
        # basins, and an unseen (sat,f) ages out after keep_maxout_s
        # seconds instead of `maxout` graph epochs.
        self.keep = _env_i('RBPF_KEEP', 0)
        self.keep_maxout_s = _env_f('RBPF_KEEP_MAXOUT_S', 10.0)
        self.keep_drop = {
            x for x in os.environ.get(
                'RBPF_KEEP_DROP', 'slip,mw,cmc,fde_cp,fde_cp_ref').split(',')
            if x}
        self.track_variants = [(self.track_q, self.track_floor,
                                self.track_empty)]
        for part in os.environ.get('RBPF_TRACK_VARIANTS', '').split(','):
            if part.strip():
                q_, f_, e_ = part.split(':')
                self.track_variants.append((float(q_), float(f_), int(e_)))
        # WP45 diagnostic (default off): record every phase-observed DD
        # row of the epoch (all non-GLONASS satellites the receiver tracks,
        # not only the shared filter's ambiguity states) with the MAP
        # basin's integer and agreement mass, for offline FIX-validation
        # studies (independent-satellite / ambiguity-function checks).
        self.dump_rows = _env_i('RBPF_DUMP_ROWS', 0)
        # WP45 ambiguity-function label feature: share of the epoch's DD
        # carrier rows within afv_thr cycle of an integer at the reported
        # position (out['afv_n'], out['afv_frac']; nothing in the filter
        # reads it -- the FIX-label rule does). 0 = not computed.
        self.afv = _env_i('RBPF_AFV', 0)
        self.afv_thr = _env_f('RBPF_AFV_THR', 0.15)


# --------------------------------------------------------------------------
# basin representation
#
# H = dict[(sys_id, f)] -> dict[sat] -> int  (relative integers, one gauge
# DOF per group; DD(ref, j) = z[ref] - z[j]).
# --------------------------------------------------------------------------

def _canon_basin_id(H):
    """Gauge-invariant, hashable basin identity."""
    out = []
    for g in sorted(H.keys()):
        zmap = H[g]
        if not zmap:
            continue
        base = zmap[min(zmap.keys())]
        out.append((g, tuple(sorted((s, z - base) for s, z in zmap.items()))))
    return tuple(out)


def _copy_H(H):
    return {g: dict(zmap) for g, zmap in H.items() if zmap}


class Particle:
    __slots__ = ('H', 'logw', 'last_move', 'state', 'vstates')

    def __init__(self, H=None, logw=0.0, state=None, vstates=None):
        self.H = H if H is not None else {}
        self.logw = float(logw)
        self.last_move = 'init'
        # WP43: None or an immutable TrackState (shared by resampled copies)
        self.state = state
        # WP43: beliefs of the extra shadow variants (index v-1)
        self.vstates = list(vstates) if vstates else []


class TrackState:
    """Position belief of one particle lineage (WP43): antenna ECEF mean
    and covariance at graph epoch `ep`, with the shared IMU odometer
    reading `odo` at that epoch (prediction = x + odo_now - odo)."""
    __slots__ = ('P', 'ep', 'odo', 'x')

    def __init__(self, x, P, odo, ep):
        self.x = x
        self.P = P
        self.odo = odo
        self.ep = ep


# --------------------------------------------------------------------------
# per-epoch problem extracted from the shared filter
# --------------------------------------------------------------------------

class EpochProblem:
    """Everything the PF needs for one epoch, sliced from the shared
    filter right after `_write_back_tc` + `_do_ar` ran."""

    def __init__(self):
        self.ok = False
        self.reason = ''
        self.pairs = []          # [(group=(sys,f), ref_sat, sat_j)]
        self._pair_runs = None
        self.y = None            # DD float (cycles), posterior
        self.Qb = None           # DD covariance (cycles^2), posterior
        self.Qab3 = None         # (3, nb) position x DD cross-cov (m*cyc)
        self.pos = None          # float antenna ECEF (3,)
        self.Ppos = None         # position cov ECEF (3,3)
        # measurement rows (this epoch's actual observations)
        self.rows_J = None       # (m,3) d(dd_range)/dx
        self.rows_v0 = None      # (m,) innovation at float pos (H-independent)
        self.rows_sig = None     # (m,) base sigma
        self.rows_iscp = None    # (m,) bool
        self.rows_pair = None    # (m,) index into pairs, -1 for PR rows
        self.rows_lam = None     # (m,) wavelength (cp rows)
        self.rows_bmeas = None   # (m,) single-epoch DD float amb (cp rows)
        self.cands = None        # (nb, K) integer candidate columns
        self.cand_sqnorm = None
        self.witness_cands = None  # witness top-K reprojected to these pairs
        self._ddpr_cache = False  # memoized _ddpr_offset result (WP18)

    def set_pairs(self, pairs):
        """Freeze this epoch's row layout, without retaining hypothesis values."""
        self.pairs = tuple((g, ref, j) for g, ref, j in pairs)
        runs = []
        for pi, (g, ref, j) in enumerate(self.pairs):
            if not runs or g != runs[-1][0] or ref != runs[-1][1]:
                runs.append((g, ref, []))
            runs[-1][2].append((pi, j))
        self._pair_runs = (self.pairs, tuple((g, ref, tuple(rows)) for g, ref, rows in runs))


# --------------------------------------------------------------------------
# the filter
# --------------------------------------------------------------------------

class RbpfFgo:

    def __init__(self, cfg: RbpfConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.particles = [Particle() for _ in range(cfg.n_particles)]
        self.prev_ref = {}        # group -> ref sat (continuity)
        self.last_seen = {}       # (sat, f) -> epoch2 last eligible
        self.n_steps = 0
        self.n_meas_updates = 0
        self.n_resamples = 0
        self.n_fix = 0
        self.n_skipped = 0
        self.step_seconds = 0.0
        self.last = None          # per-epoch output dict
        # WP17 item 1: basin-feedback (COND_HOLD) state
        self.fb_streak = {}       # (g, ref, j, z) -> consecutive gated epochs
        self.fb_bad = {}          # (sat, f) -> consecutive low-mass epochs
        self.fb_unseen_pending = set()   # (sat,f) aged out while held
        self.fb_commit_count = 0
        # WP52: holds taken from the label layer's integer carry
        self.carry_protect = set()
        self.carry_fb_commits = 0
        self.carry_fb_epochs = 0
        self.carry_fb_skipped = 0
        self.fb_release_count = 0
        self.fb_release_mass = 0
        self.fb_release_unseen = 0
        # WP17 item 3: post-outage re-entry spawning state
        self.no_cp_streak = 0
        self.spawn_remaining = 0
        self.spawn_events = 0
        self.fixless = 0
        self._last_tow = None
        # WP18 Stage-2: failure-mode guards state/counters
        self.ddpr_bad_streak = 0     # (a) consecutive trusted-DDPR bad epochs
        self.fb_release_ddpr = 0     # (a) release-all events fired
        self.fb_commit_gated = 0     # (a) epochs whose commit pass was gated
        self.fix_info_rejects = 0    # (b) Q=4 reports rejected on min_info
        self.chal_streak = 0         # (a) consecutive bad reported-fix epochs
        self.chal_spawn_events = 0   # (a) forced challenger spawns
        self.fix_vote_rejects = 0    # (b) Q=4 reports rejected by the vote
        self.cluster_states = {}     # basin id -> {H, pos, P}
        self._conditional_cache = {}
        self.fb_support_bad_streak = 0
        self.fb_release_support = 0
        self.fb_dual_gated = 0
        self.fb_witness_gated = 0
        self.delayed_history = []  # (epoch2, fixed-minus-float ECEF vector)
        self.delayed_rejuv_remaining = 0
        self.delayed_rejuv_events = 0
        self.delayed_rejuv_armed = True
        self.delayed_rejuv_last_ep = -10 ** 9
        self.shadow_pending = False
        self.shadow_challengers = []  # {H, score, age}
        self.shadow_promotions = 0
        self.track_evals = 0         # WP43 batched (particle x move) solves
        # WP17 item 4: WP15 GPU batch-LAMBDA drop-in (CPU fallback kept)
        self._gpu_mlb = None
        self.gpu_topk_calls = 0
        self.gpu_topk_fallbacks = 0
        if cfg.use_cuda:
            try:
                import sys
                from pathlib import Path
                _gp = os.environ.get(
                    'GNSS_GPU_PY',
                    str(Path(__file__).resolve().parent.parent
                        / 'gnss_gpu' / 'python'))
                if _gp not in sys.path:
                    sys.path.insert(0, _gp)
                from gnss_gpu.lambda_batch import (
                    mlambda_batch as _mlb, HAS_LAMBDA_BATCH as _has)
                if not _has:
                    raise ImportError(
                        '_gnss_gpu_lambda_batch native module missing')
                _mlb([np.array([0.1, 0.2])], [np.eye(2) * 0.1],
                     ncands=2, parmode=1)   # one-shot CUDA probe
                self._gpu_mlb = _mlb
                print('RBPF: RBPF_CUDA active (gnss_gpu.lambda_batch)')
            except Exception as _e:
                print(f'RBPF: RBPF_CUDA=1 but GPU path unavailable '
                      f'({_e}); using CPU mlambda')

    # -- maintenance ------------------------------------------------------

    def _maintenance(self, rtk, support_sf, ep, tow=None):
        """Drop released / re-seeded / long-unseen (sat,f) from every
        particle's basin; update last-seen."""
        released = getattr(rtk, '_rbpf_released', None) or set()
        if self.cfg.keep:
            return self._maintenance_keep(rtk, support_sf, released, tow)
        if getattr(rtk, '_rbpf_graph_reseeded', False):
            rtk._rbpf_graph_reseeded = False
            for p in self.particles:
                p.H = {}
                p.state = None
                p.vstates = []
            self.last_seen.clear()
            released = set()
            # WP17: a graph re-seed wipes the feedback state too (the
            # base class already cleared rtk._committed) and arms the
            # re-entry spawner (the fresh float re-enters like a
            # post-outage float).
            self.fb_streak.clear()
            self.fb_bad.clear()
            self.cluster_states.clear()
            self.fb_support_bad_streak = 0
            if self.cfg.spawn_enable:
                self.no_cp_streak = max(self.no_cp_streak,
                                        self.cfg.spawn_outage)
        fresh = {k for k, e0 in rtk._amb_init_epoch.items() if e0 == ep}
        for sf in support_sf:
            self.last_seen[sf] = ep
        dead = set(released) | set(fresh)
        for (s, f), e_seen in list(self.last_seen.items()):
            if ep - e_seen > self.cfg.maxout:
                dead.add((s, f))
                del self.last_seen[(s, f)]
                # WP17: a held (sat,f) that ages out of the PF support
                # must release its conditioned hold too (tc/'s outage
                # clear_hold analogue) -- queued for the next feedback
                # pass (last_seen alone would be refreshed again by the
                # time the sat re-enters).
                if self.cfg.fb_enable:
                    self.fb_unseen_pending.add((int(s), int(f)))
        if dead:
            for p in self.particles:
                for (s, f) in dead:
                    sys_id, _ = sat2prn(int(s))
                    g = (int(sys_id), int(f))
                    zmap = p.H.get(g)
                    if zmap is not None:
                        zmap.pop(int(s), None)
                        if len(zmap) < 2:
                            p.H.pop(g, None)
        if hasattr(rtk, '_rbpf_released'):
            rtk._rbpf_released = set()
            rtk._rbpf_release_reasons = {}

    def _maintenance_keep(self, rtk, support_sf, released, tow):
        """WP44 RBPF_KEEP=1 maintenance (see RbpfConfig.keep)."""
        cfg = self.cfg
        reasons = getattr(rtk, '_rbpf_release_reasons', None) or {}
        if getattr(rtk, '_rbpf_graph_reseeded', False):
            rtk._rbpf_graph_reseeded = False
            # the base class cleared rtk._committed: restart the feedback
            # streaks, keep the integers
            self.fb_streak.clear()
            self.fb_bad.clear()
            self.cluster_states.clear()
            self.fb_support_bad_streak = 0
            if cfg.spawn_enable:
                self.no_cp_streak = max(self.no_cp_streak, cfg.spawn_outage)
        dead = {k for k in released
                if any(r in cfg.keep_drop for r in reasons.get(k, ('',)))}
        if tow is not None:
            for sf in support_sf:
                self.last_seen[sf] = tow
            for (s, f), t_seen in list(self.last_seen.items()):
                if tow - t_seen > cfg.keep_maxout_s:
                    dead.add((s, f))
                    del self.last_seen[(s, f)]
                    if cfg.fb_enable:
                        self.fb_unseen_pending.add((int(s), int(f)))
        if dead:
            for p in self.particles:
                for (s, f) in dead:
                    sys_id, _ = sat2prn(int(s))
                    g = (int(sys_id), int(f))
                    zmap = p.H.get(g)
                    if zmap is not None:
                        zmap.pop(int(s), None)
                        if len(zmap) < 2:
                            p.H.pop(g, None)
        if hasattr(rtk, '_rbpf_released'):
            rtk._rbpf_released = set()
            rtk._rbpf_release_reasons = {}

    # -- problem extraction -------------------------------------------------

    def _extract(self, rtk, ctx):
        """Build the epoch problem from the shared filter state + the
        stashed epoch context (obs/geometry from `_build_dd_factors_arm`)."""
        prob = EpochProblem()
        nav = rtk.nav
        na = nav.na
        sat = ctx['sat']
        el = ctx['el']
        iu = ctx['iu']
        ir = ctx['ir']
        rs = ctx['rs']
        rsb = ctx['rsb']
        obs_sd = ctx['obs_sd']

        if rtk.last_joint_P is None:
            prob.reason = 'no_joint_marginal'
            return prob

        pos = np.asarray(nav.x[0:3], dtype=float).copy()
        Ppos = np.asarray(nav.P[0:3, 0:3], dtype=float).copy()
        if not (np.all(np.isfinite(pos)) and np.all(np.isfinite(Ppos))):
            prob.reason = 'nonfinite_pos'
            return prob
        # guard: degenerate position covariance
        if not np.all(np.linalg.eigvalsh(0.5 * (Ppos + Ppos.T)) > 0):
            Ppos = Ppos + np.eye(3) * 1e-4
        # relax the (overconfident) graph position marginal for the
        # per-epoch likelihood LS -- see RbpfConfig.sig_pos_floor
        Ppos = Ppos + np.eye(3) * self.cfg.sig_pos_floor ** 2

        # eligible support: today's sats with vsat==1 on f, non-GLO
        idx_of_sat = {int(sat[i]): i for i in range(len(sat))}
        groups = {}
        for i in range(len(sat)):
            s = int(sat[i])
            sys_id, _ = sat2prn(s)
            if sys_id == uGNSS.GLO:
                continue  # FDMA DD ambiguities are non-integer
            for f in range(nav.nf):
                if nav.vsat[s - 1, f] != 1:
                    continue
                xN = nav.x[rtk.IB(s, f, na)]
                if not np.isfinite(xN):
                    continue
                groups.setdefault((int(sys_id), int(f)), []).append(s)

        pairs = []
        for g in sorted(groups.keys()):
            mem = groups[g]
            if len(mem) < 2:
                continue
            ref = self.prev_ref.get(g)
            if ref not in mem:
                ref = mem[int(np.argmax([el[idx_of_sat[s]] for s in mem]))]
            self.prev_ref[g] = ref
            for j in mem:
                if j != ref:
                    pairs.append((g, ref, j))
        if len(pairs) < 1:
            prob.reason = 'no_pairs'
            return prob

        nb = len(pairs)
        ib0 = np.array([rtk.IB(r, g[1], na) for (g, r, j) in pairs])
        ib1 = np.array([rtk.IB(j, g[1], na) for (g, r, j) in pairs])
        P = nav.P
        y = nav.x[ib0] - nav.x[ib1]
        DP = P[np.ix_(ib0, ib0)] - P[np.ix_(ib0, ib1)] \
            - P[np.ix_(ib1, ib0)] + P[np.ix_(ib1, ib1)]
        Qb = 0.5 * (DP + DP.T)
        Qab3 = P[0:3, :][:, ib0] - P[0:3, :][:, ib1]
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Qb))
                and np.all(np.isfinite(Qab3))):
            prob.reason = 'nonfinite_marginal'
            return prob
        Qb = Qb + np.eye(nb) * self.cfg.qb_jitter

        # measurement rows from this epoch's actual observations
        rows_J, rows_v0, rows_sig, rows_iscp, rows_pair = [], [], [], [], []
        rows_lam, rows_bmeas = [], []
        rb = np.array(nav.rb, dtype=float)
        lam_cache = {}
        for pi, (g, ref, j) in enumerate(pairs):
            sysf = g
            f = g[1]
            ri = idx_of_sat.get(ref)
            ji = idx_of_sat.get(j)
            if ri is None or ji is None:
                continue
            if obs_sd.P.shape[1] <= f:
                continue
            if obs_sd.P[ri, f] == 0 or obs_sd.P[ji, f] == 0:
                continue
            rs_ref = rs[iu[ri], :3]
            rs_j = rs[iu[ji], :3]
            rsb_ref = rsb[ir[ri], :3]
            rsb_j = rsb[ir[ji], :3]
            r_ref, e_ref = geodist(rs_ref, pos)
            r_j, e_j = geodist(rs_j, pos)
            rb_ref, _ = geodist(rsb_ref, rb)
            rb_j, _ = geodist(rsb_j, rb)
            ddrange = (r_ref - rb_ref) - (r_j - rb_j)
            # d(ddrange)/dx: d|rs-x|/dx = -e  (e = unit rover->sat)
            Jrow = -np.asarray(e_ref, dtype=float) + np.asarray(e_j, dtype=float)

            dd_pr = float(obs_sd.P[ri, f] - obs_sd.P[ji, f])
            sig_pr = (rtk._varerr_dd_sigma(1, el[ri], el[ji])
                      if rtk.varerr_enable else rtk.sigma_pr * np.sqrt(2.0))
            rows_J.append(Jrow)
            rows_v0.append(dd_pr - ddrange)
            rows_sig.append(sig_pr)
            rows_iscp.append(False)
            rows_pair.append(-1)
            rows_lam.append(0.0)
            rows_bmeas.append(0.0)

            if obs_sd.L.shape[1] <= f:
                continue
            if obs_sd.L[ri, f] == 0 or obs_sd.L[ji, f] == 0:
                continue
            if sysf not in lam_cache:
                lam_cache[sysf] = _get_wavelengths(nav, obs_sd, ref)
            lams = lam_cache[sysf]
            if f >= len(lams) or not np.isfinite(lams[f]) or lams[f] <= 0:
                continue
            lam = float(lams[f])
            dd_cp = float(obs_sd.L[ri, f] - obs_sd.L[ji, f]) * lam
            b_meas = (dd_cp - ddrange) / lam
            sig_cp = (rtk._varerr_dd_sigma(0, el[ri], el[ji])
                      if rtk.varerr_enable else rtk.sigma_cp * np.sqrt(2.0))
            sig_cp = max(sig_cp, self.cfg.sig_cp_floor)
            rows_J.append(Jrow)
            rows_v0.append(0.0)          # filled per-hypothesis
            rows_sig.append(sig_cp)
            rows_iscp.append(True)
            rows_pair.append(pi)
            rows_lam.append(lam)
            rows_bmeas.append(b_meas)

        if not rows_J:
            prob.reason = 'no_rows'
            return prob

        prob.set_pairs(pairs)
        prob.y = np.asarray(y, dtype=float)
        prob.Qb = Qb
        prob.Qab3 = np.asarray(Qab3, dtype=float)
        prob.pos = pos
        prob.Ppos = Ppos
        prob.rows_J = np.asarray(rows_J, dtype=float)
        prob.rows_v0 = np.asarray(rows_v0, dtype=float)
        prob.rows_sig = np.asarray(rows_sig, dtype=float)
        prob.rows_iscp = np.asarray(rows_iscp, dtype=bool)
        prob.rows_pair = np.asarray(rows_pair, dtype=int)
        prob.rows_lam = np.asarray(rows_lam, dtype=float)
        prob.rows_bmeas = np.asarray(rows_bmeas, dtype=float)

        # top-K LAMBDA candidates from the posterior joint (GPU batch-
        # LAMBDA swap-in point: same (y, Qb, K) -> columns contract).
        prob.cands, prob.cand_sqnorm = self._lambda_topk(
            prob.y, prob.Qb, self.cfg.topk)
        prob.ok = True
        return prob

    def _attach_witness(self, prob, witness_prob):
        """Reproject witness top-K basins onto production DD pairs."""
        if (witness_prob is None or not witness_prob.ok
                or witness_prob.cands is None):
            return
        cols = []
        for k in range(witness_prob.cands.shape[1]):
            Hw = self._adopt(witness_prob, k)
            cols.append(self._pair_dd(Hw, prob))
        if cols:
            prob.witness_cands = np.column_stack(cols)

    def _lambda_topk(self, y, Qb, K):
        """mlambda top-K (full ILS). Returns (nb,K') int array or
        (None, None). WP17 item 4: with RBPF_CUDA=1 this dispatches to
        the WP15 gnss_gpu.lambda_batch kernel (bit-identical transcription
        of cssrlib mlambda, WP15 parity report: 0 mismatches on 34,569
        real inputs); ANY GPU-side problem falls back to the CPU path."""
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(Qb))):
            return None, None      # same reject as the WP13m guard
        if self._gpu_mlb is not None:
            try:
                res = self._gpu_mlb(
                    [np.asarray(y, dtype=float)],
                    [np.asarray(Qb, dtype=float)],
                    ncands=int(K), parmode=1)[0]
                if res.status == 0:
                    self.gpu_topk_calls += 1
                    return self._topk_postprocess(res.afix, res.s)
                if res.status == 1:
                    # non-PD covariance: cssrlib raises LambdaError here
                    self.gpu_topk_calls += 1
                    return None, None
                # status 2 = unsupported dims -> CPU fallback
                self.gpu_topk_fallbacks += 1
            except Exception:
                self.gpu_topk_fallbacks += 1
        try:
            afix, sqnorm, _, _ = _wp13m_finite_guarded_mlambda(
                y, Qb, ncands=int(K), parmode=1)
        except (LambdaError, np.linalg.LinAlgError, ValueError):
            return None, None
        return self._topk_postprocess(afix, sqnorm)

    @staticmethod
    def _topk_postprocess(afix, sqnorm):
        cands = np.asarray(afix)
        if cands.ndim == 1:
            cands = cands[:, None]
        sq = np.asarray(sqnorm, dtype=float).reshape(-1)
        keep = sq < 1e17          # mlambda pre-fills 1e18 for unfilled slots
        if not np.any(keep):
            return None, None
        return np.rint(cands[:, keep]).astype(int), sq[keep]

    # -- hypothesis scoring -------------------------------------------------

    def _pair_dd(self, H, prob):
        """(nb,) DD integers for pairs covered by H (nan = not covered)."""
        out = np.full(len(prob.pairs), np.nan)
        prepared = getattr(prob, '_pair_runs', None)
        if prepared is not None and prepared[0] is prob.pairs:
            for g, ref, rows in prepared[1]:
                zmap = H.get(g)
                zr = zmap.get(ref) if zmap is not None else None
                if zr is None:
                    continue
                for pi, j in rows:
                    zj = zmap.get(j)
                    if zj is not None:
                        out[pi] = zr - zj
            return out
        for pi, (g, ref, j) in enumerate(prob.pairs):
            zmap = H.get(g)
            if zmap is None:
                continue
            zr = zmap.get(ref)
            zj = zmap.get(j)
            if zr is not None and zj is not None:
                out[pi] = zr - zj
        return out

    @staticmethod
    def _basin_parent_score(child, parent):
        """Return (agree, compared) over gauge-invariant common-sat DDs."""
        agree = compared = 0
        for g in set(child).intersection(parent):
            sats = sorted(set(child[g]).intersection(parent[g]))
            if len(sats) < 2:
                continue
            ref = sats[0]
            for sat in sats[1:]:
                compared += 1
                if (child[g][ref] - child[g][sat]
                        == parent[g][ref] - parent[g][sat]):
                    agree += 1
        return agree, compared

    def _cluster_state(self, H):
        """Exact state, else an unambiguous high-overlap parent state."""
        if not self.cfg.cluster_relin or not H:
            return None
        # Require enough independent DD support for *all* shadow-state
        # inheritance, including an exact basin-id match. Previously the
        # threshold applied only while searching a changed-id parent, so a
        # low-support wrong basin could retain its exact shadow position
        # indefinitely (WP22 run2 tow 178106-178143).
        support = sum(max(len(zmap) - 1, 0) for zmap in H.values())
        if support < self.cfg.cluster_parent_min:
            return None
        bid = _canon_basin_id(H)
        if bid in self.cluster_states:
            return self.cluster_states[bid]
        ranked = []
        for state in self.cluster_states.values():
            agree, compared = self._basin_parent_score(H, state['H'])
            if (compared >= self.cfg.cluster_parent_min
                    and agree / compared >= self.cfg.cluster_parent_frac):
                ranked.append((agree, compared, state))
        if not ranked:
            return None
        ranked.sort(key=lambda x: (x[0], x[1]), reverse=True)
        if len(ranked) > 1 and ranked[0][:2] == ranked[1][:2]:
            return None
        return ranked[0][2]

    def _likelihood_rows(self, prob, dd):
        """Prepare residuals and scales, preserving lattice-sum row order."""
        cfg = self.cfg
        m = len(prob.rows_v0)
        v = prob.rows_v0.copy()
        sig = prob.rows_sig.copy()
        iscp = prob.rows_iscp
        pi = prob.rows_pair
        lam = prob.rows_lam
        sb = math.sqrt(cfg.beta)
        uniform = np.zeros(m, dtype=bool)   # flat-over-lattice rows
        const_ll = 0.0
        if cfg.marg_uniform:
            cp = np.flatnonzero(iscp)
            z = dd[pi[cp]]
            held = np.isfinite(z)
            rows = cp[held]
            v[rows] = lam[rows] * (prob.rows_bmeas[rows] - z[held])
            sig[rows] = sig[rows] / sb
            for r in cp[~held]:
                uniform[r] = True
                const_ll += -math.log(max(lam[r], 1e-3))
            return v, sig, uniform, const_ll
        qdiag = np.diag(prob.Qb)
        for r in range(m):
            if not iscp[r]:
                continue
            p = pi[r]
            z = dd[p]
            if np.isfinite(z):
                v[r] = lam[r] * (prob.rows_bmeas[r] - z)
                sig[r] = sig[r] / sb
            elif cfg.marg_uniform:
                # un-held ambiguity, flat prior over the integer lattice:
                # density 1/lam per metre, carries no position info
                uniform[r] = True
                const_ll += -math.log(max(lam[r], 1e-3))
            else:
                v[r] = lam[r] * (prob.rows_bmeas[r] - prob.y[p])
                sig[r] = math.sqrt(
                    sig[r] ** 2 + lam[r] ** 2 * (max(qdiag[p], 0.0) + 1.0 / 12.0)) / sb
        return v, sig, uniform, const_ll

    def _nuisance_columns(self, prob, iscp, pi_kept, sig):
        """Carrier nuisance columns in first retained group occurrence order."""
        cp = np.flatnonzero(iscp)
        groups = {}
        ids = np.empty(len(cp), dtype=int)
        for i, r in enumerate(cp):
            g = prob.pairs[pi_kept[r]][0]
            ids[i] = groups.setdefault(g, len(groups))
        G = len(groups)
        if not G:
            return None, 0
        C = np.zeros((len(sig), G))
        if sig.dtype == np.float64:
            C[cp, ids] = 1.0 / sig[cp]
        else:
            # Keep scalar promotion/rounding for other numeric dtypes.
            for r, gi in zip(cp, ids):
                C[r, gi] = 1.0 / sig[r]
        return C, G

    def _loglik(self, H, prob, cache):
        """Tempered per-epoch marginal likelihood of this epoch's DD rows
        under hypothesis H (position marginalized against Ppos)."""
        key = _canon_basin_id(H)
        hit = cache.get(key)
        if hit is not None:
            return hit
        cfg = self.cfg
        dd = self._pair_dd(H, prob)
        v, sig, uniform, const_ll = self._likelihood_rows(prob, dd)
        iscp, pi = prob.rows_iscp, prob.rows_pair
        if np.any(uniform):
            keep = ~uniform
            v, sig = v[keep], sig[keep]
            J = prob.rows_J[keep]
            iscp = iscp[keep]
            pi_kept = pi[keep]
        else:
            J = prob.rows_J
            pi_kept = pi
        centre = np.zeros(3)
        state = self._cluster_state(H)
        Pprior = prob.Ppos
        if state is not None:
            centre = np.asarray(state['pos'], dtype=float) - prob.pos
            Pprior = np.asarray(state['P'], dtype=float)
            if not np.all(np.isfinite(centre)):
                centre = np.zeros(3)
                Pprior = prob.Ppos
        # Basin-specific shadow conditional: residuals are linearized
        # around the basin's previous conditional position and the
        # Gaussian offset prior is centred there. `sol` below is the
        # correction from this shadow centre.
        v = v - J @ centre
        A = J / sig[:, None]
        b = v / sig
        # per-(sys,f)-group common-offset nuisance columns on HELD CP
        # rows (reference-sat carrier multipath model, see RbpfConfig)
        C, G = self._nuisance_columns(prob, iscp, pi_kept, sig)
        if G:
            A = np.hstack([A, C])
        try:
            Pinv3 = np.linalg.inv(Pprior)
            nd = 3 + G
            Pinv = np.zeros((nd, nd))
            Pinv[0:3, 0:3] = Pinv3
            if G:
                Pinv[3:, 3:] = np.eye(G) / (cfg.sig_group ** 2)
            Lam = Pinv + A.T @ A
            m1 = A.T @ b
            sol = np.linalg.solve(Lam, m1)
            c = cfg.huber_c
            if c > 0:
                # IRLS-Huber offset solve (robust point estimate)
                for _ in range(2):
                    r = b - A @ sol
                    wgt = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
                    sw = np.sqrt(wgt)
                    Aw = A * sw[:, None]
                    Lam = Pinv + Aw.T @ Aw
                    sol = np.linalg.solve(Lam, Aw.T @ (b * sw))
            sign, logdet_lam = np.linalg.slogdet(Lam)
            sign2, logdet_p3 = np.linalg.slogdet(Pprior)
            if sign <= 0 or sign2 <= 0:
                raise np.linalg.LinAlgError
            logdet_p = logdet_p3 + 2.0 * G * math.log(cfg.sig_group)
            # per-row mixture data loglik at the robust offset
            # (Laplace-style approximation of the marginal -- declared)
            r = b - A @ sol
            s_out = np.maximum(
                np.where(iscp, cfg.mix_sigma_cp, cfg.mix_sigma_pr), sig)
            eps = cfg.mix_eps
            ll_base = math.log(1.0 - eps) - 0.5 * r * r - np.log(sig)
            ll_out = (math.log(eps) - 0.5 * (r * sig / s_out) ** 2
                      - np.log(s_out))
            row_ll = np.logaddexp(ll_base, ll_out) - 0.5 * _LOG2PI
            ll = (float(np.sum(row_ll)) + const_ll
                  - 0.5 * float(sol @ (Pinv @ sol))
                  - 0.5 * logdet_lam - 0.5 * logdet_p)
        except np.linalg.LinAlgError:
            ll = -1e12
        cache[key] = ll
        return ll

    def _batch_rows(self, prob):
        """Hypothesis-independent row layout for `_loglik_batch`
        (memoized on the problem): the stacked design (m, 3+G) with one
        nuisance column per (sys,f) group carrying CP rows."""
        got = getattr(prob, '_batch_rows', None)
        if got is not None:
            return got
        iscp = prob.rows_iscp
        groups = {}
        grp = np.full(len(iscp), -1, dtype=int)
        for r in np.where(iscp)[0]:
            g = prob.pairs[prob.rows_pair[r]][0]
            grp[r] = groups.setdefault(g, len(groups))
        G = len(groups)
        A0 = np.zeros((len(iscp), 3 + G))
        A0[:, 0:3] = prob.rows_J
        for r in np.where(grp >= 0)[0]:
            A0[r, 3 + grp[r]] = 1.0
        prob._batch_rows = (A0, G)
        return prob._batch_rows

    def _loglik_batch(self, prob, dds, centres, Pps):
        """Batched `_loglik` (marg_uniform=1 model) over B hypotheses.

        dds (B, nb) held DD integers (nan = un-held); centres (B, 3) prior
        centre offsets from the float position; Pps (B, 3, 3) prior
        covariances. Returns (ll (B,), sol (B, 3) posterior offset from
        the float position, P3 (B, 3, 3) posterior position covariance).

        Every hypothesis shares one (m, 3+G) design: un-held CP rows are
        masked to zero (their flat-lattice density is the constant term)
        and a group column with no held row only meets its own prior,
        whose log-determinant cancels -- so the result equals the scalar
        `_loglik` up to rounding. This is the kernel a GPU port replaces.
        """
        cfg = self.cfg
        A0, G = self._batch_rows(prob)
        B = dds.shape[0]
        n = A0.shape[1]
        iscp = prob.rows_iscp
        pi = np.where(iscp, prob.rows_pair, 0)
        z = dds[:, pi]                                     # (B, m)
        held = iscp[None, :] & np.isfinite(z)
        mask = (~iscp)[None, :] | held
        sb = math.sqrt(cfg.beta)
        lam = prob.rows_lam
        v = np.where(held, lam[None, :] * (prob.rows_bmeas[None, :]
                                           - np.where(held, z, 0.0)),
                     prob.rows_v0[None, :])
        sig = np.where(held, prob.rows_sig[None, :] / sb,
                       prob.rows_sig[None, :])
        unif = iscp[None, :] & ~held
        const_ll = -np.sum(np.where(unif, np.log(np.maximum(lam, 1e-3))[None, :],
                                    0.0), axis=1)
        v = v - centres @ prob.rows_J.T
        wrow = mask / sig                                  # (B, m)
        A = A0[None, :, :] * wrow[:, :, None]
        b = v * wrow
        Pinv = np.zeros((B, n, n))
        ll = np.full(B, -1e12)
        sol = np.zeros((B, n))
        P3 = np.repeat(np.eye(3)[None], B, axis=0)
        try:
            Pinv[:, 0:3, 0:3] = np.linalg.inv(Pps)
        except np.linalg.LinAlgError:
            return ll, sol[:, 0:3], P3
        if G:
            idx = np.arange(3, n)
            Pinv[:, idx, idx] = 1.0 / (cfg.sig_group ** 2)
        Lam = Pinv + np.einsum('bmi,bmj->bij', A, A)
        try:
            s = np.linalg.solve(Lam, np.einsum('bmi,bm->bi', A, b)[..., None])[..., 0]
            c = cfg.huber_c
            if c > 0:
                for _ in range(2):
                    r = b - np.einsum('bmi,bi->bm', A, s)
                    sw = np.sqrt(np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12)))
                    Aw = A * sw[:, :, None]
                    Lam = Pinv + np.einsum('bmi,bmj->bij', Aw, Aw)
                    s = np.linalg.solve(
                        Lam, np.einsum('bmi,bm->bi', Aw, b * sw)[..., None])[..., 0]
        except np.linalg.LinAlgError:
            return ll, sol[:, 0:3], P3
        sign, logdet_lam = np.linalg.slogdet(Lam)
        sign2, logdet_p3 = np.linalg.slogdet(Pps)
        logdet_p = logdet_p3 + 2.0 * G * math.log(cfg.sig_group)
        r = b - np.einsum('bmi,bi->bm', A, s)
        sig_safe = np.where(mask, sig, 1.0)
        s_out = np.maximum(np.where(iscp, cfg.mix_sigma_cp, cfg.mix_sigma_pr)[None, :],
                           sig_safe)
        eps = cfg.mix_eps
        ll_base = math.log(1.0 - eps) - 0.5 * r * r - np.log(sig_safe)
        ll_out = math.log(eps) - 0.5 * (r * sig_safe / s_out) ** 2 - np.log(s_out)
        row_ll = np.logaddexp(ll_base, ll_out) - 0.5 * _LOG2PI
        prior = np.einsum('bi,bij,bj->b', s, Pinv, s)
        out = (np.sum(np.where(mask, row_ll, 0.0), axis=1) + const_ll
               - 0.5 * prior - 0.5 * logdet_lam - 0.5 * logdet_p)
        good = (sign > 0) & (sign2 > 0) & np.all(np.isfinite(s), axis=1)
        ll = np.where(good, out, -1e12)
        try:
            P3 = np.linalg.inv(Lam)[:, 0:3, 0:3]
        except np.linalg.LinAlgError:
            pass
        P3 = 0.5 * (P3 + np.swapaxes(P3, 1, 2))
        return ll, s[:, 0:3] + centres, P3

    # -- WP43 per-particle position tracking ------------------------------------

    def _track_reset(self, ctx):
        """True when every belief must be dropped this epoch: the runner
        flagged a PIM break / missing odometer, or the data gap since the
        last tracked epoch exceeds track_gap_s."""
        reset = bool(ctx.get('track_reset')) or ctx.get('odo') is None
        try:
            _, tow = time2gpst(ctx['obs'].t)
        except (KeyError, AttributeError, TypeError, ValueError):
            return reset
        last = getattr(self, '_track_last_tow', None)
        if last is not None and tow - last > self.cfg.track_gap_s:
            reset = True
        self._track_last_tow = float(tow)
        return reset

    def _track_predict(self, st, odo, ep, q=None):
        """Predict a TrackState to graph epoch `ep` -> (x, P) or None."""
        if st is None:
            return None
        q = self.cfg.track_q if q is None else q
        n = max(int(ep - st.ep), 1)
        x = st.x + (odo - st.odo)
        P = st.P + np.eye(3) * (q ** 2) * n
        return x, P

    # -- moves ---------------------------------------------------------------

    def _move_pair(self, H, prob):
        """Build both structural moves from one particle's current DDs.

        H and prob are read-only in both helpers. Keep the shared array
        local to this call so maintenance or the next epoch cannot reuse it.
        """
        dd = self._pair_dd(H, prob)
        return self._extend(H, prob, dd), self._shrink_worst(H, prob, dd)

    def _extend(self, H, prob, dd=None):
        """Conditional-rounding bootstrap: extend H over the un-held part
        of today's support, conditioning the posterior float on H."""
        if dd is None:
            dd = self._pair_dd(H, prob)
        held = np.isfinite(dd)
        idxH = np.flatnonzero(held)
        if len(idxH) == 0 or len(idxH) == len(dd):
            return None
        idxC = [pi for pi in np.flatnonzero(~held)
                if H.get(prob.pairs[pi][0], {}).get(prob.pairs[pi][1]) is not None]
        if len(idxC) == 0:
            return None
        try:
            QHH = prob.Qb[np.ix_(idxH, idxH)]
            QCH = prob.Qb[np.ix_(idxC, idxH)]
            r = np.linalg.solve(QHH, prob.y[idxH] - dd[idxH])
            b_cond = prob.y[idxC] - QCH @ r
        except np.linalg.LinAlgError:
            return None
        H2 = _copy_H(H)
        for k, pi in enumerate(idxC):
            g, ref, j = prob.pairs[pi]
            zmap = H2.setdefault(g, {})
            zr = zmap.get(ref)
            if zr is None:
                continue
            zmap[j] = int(zr - int(np.rint(b_cond[k])))
        return H2

    @staticmethod
    def _adopt(prob, k, subset=None):
        """Candidate column k -> basin (optionally restricted to a pair
        subset, e.g. the top-K consensus set = partial-AR analogue)."""
        H = {}
        z = prob.cands[:, k]
        for pi, (g, ref, j) in enumerate(prob.pairs):
            if subset is not None and pi not in subset:
                continue
            zmap = H.setdefault(g, {})
            zmap.setdefault(ref, 0)
            zmap[j] = int(zmap[ref] - int(z[pi]))
        return {g: zm for g, zm in H.items() if len(zm) >= 2}

    @staticmethod
    def _consensus_subset(prob):
        """Pairs on which ALL top-K ILS candidates agree (the classic
        partial-fix heuristic: disagreement marks the uncertain
        components)."""
        if prob.cands is None or prob.cands.shape[1] < 2:
            return None
        agree = np.all(prob.cands == prob.cands[:, :1], axis=1)
        idx = set(np.where(agree)[0].tolist())
        return idx if len(idx) >= 2 else None

    @staticmethod
    def _adopt_map(prob, z_by_pair):
        """{pair index -> DD integer} -> basin (same convention as
        `_adopt`: DD(ref,j) = z[ref] - z[j])."""
        H = {}
        for pi, z in z_by_pair.items():
            g, ref, j = prob.pairs[pi]
            zmap = H.setdefault(g, {})
            zmap.setdefault(ref, 0)
            zmap[j] = int(zmap[ref] - int(z))
        return {g: zm for g, zm in H.items() if len(zm) >= 2}

    def _ddpr_offset(self, prob):
        """DDPR-only robust position offset from the float (IRLS-Huber
        LS on this epoch's raw DD-PR rows, wide prior). Returns
        (sol3, P3) or None. Shared by the spawner (candidate anchor)
        and the fix-output cross-check. WP18: memoized per epoch (pure
        function of the extracted problem; the result is now also used
        every epoch for the coherent-shift diagnostics)."""
        if prob._ddpr_cache is not False:
            return prob._ddpr_cache
        prob._ddpr_cache = self._ddpr_offset_impl(prob)
        return prob._ddpr_cache

    def _ddpr_offset_impl(self, prob):
        cfg = self.cfg
        pr = ~prob.rows_iscp
        if int(pr.sum()) < 3:
            return None
        J = prob.rows_J[pr]
        v = prob.rows_v0[pr]
        sig = prob.rows_sig[pr]
        A = J / sig[:, None]
        b = v / sig
        Pinv = np.eye(3) / (cfg.spawn_sig_pos ** 2)
        try:
            Lam = Pinv + A.T @ A
            sol = np.linalg.solve(Lam, A.T @ b)
            c = cfg.huber_c
            if c > 0:
                for _ in range(2):
                    r = b - A @ sol
                    wgt = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
                    sw = np.sqrt(wgt)
                    Aw = A * sw[:, None]
                    Lam = Pinv + Aw.T @ Aw
                    sol = np.linalg.solve(Lam, Aw.T @ (b * sw))
            Pp = np.linalg.inv(Lam)
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(sol)):
            return None
        return sol, Pp

    def _spawn_candidates(self, prob, force=False):
        """WP17 item 3 (design section 3, outage re-entry): hypotheses =
        {DDPR-only position prior} x {top-K LAMBDA candidates under that
        prior}. The float posterior re-enters an outage 0.5-1.5 m wrong
        (the WP13s wrong-basin re-entry failure), so candidate generation
        here deliberately bypasses it: position comes from THIS epoch's
        raw DD-PR rows alone (robust LS, wide prior), and the DD-CP
        single-epoch floats are re-anchored at that position before the
        integer search. Returns a list of candidate basins (full vectors
        + their top-K consensus subset)."""
        cfg = self.cfg
        iscp = prob.rows_iscp
        got = self._ddpr_offset(prob)
        if got is None:
            return []
        sol, Pp = got
        if (not force and cfg.spawn_min_dpos > 0
                and float(np.linalg.norm(sol[:3])) < cfg.spawn_min_dpos):
            return []   # float and DDPR agree: not a re-entry regime
            # (force=True: WP18 challenger spawns -- during a coherent-
            # shift lock the float-vs-DDPR distance is ~1 m, under this
            # gate, yet the DDPR anchor is the only signal that does
            # not share the shift.)
        # one CP row per pair, single-epoch float re-anchored at the
        # DDPR position: b(x+dx) = b_meas - (J dx)/lam
        pair_row = {}
        for r in range(len(iscp)):
            if iscp[r]:
                pair_row.setdefault(int(prob.rows_pair[r]), r)
        pis = sorted(pair_row)
        if len(pis) < cfg.spawn_min_cp:
            return []
        rows = [pair_row[pi] for pi in pis]
        G = np.array([prob.rows_J[r] / prob.rows_lam[r] for r in rows])
        yv = np.array([prob.rows_bmeas[r]
                       - float(prob.rows_J[r] @ sol) / prob.rows_lam[r]
                       for r in rows])
        sig_r = np.array([prob.rows_sig[r] / prob.rows_lam[r] for r in rows])
        Q = G @ Pp @ G.T + np.diag(sig_r ** 2)
        Q = 0.5 * (Q + Q.T) + np.eye(len(pis)) * cfg.qb_jitter
        cands, _ = self._lambda_topk(yv, Q, cfg.topk)
        if cands is None:
            return []
        out = []
        seen = set()
        # consensus subset of the spawn candidates first (partial entry
        # move -- full vectors are usually partially wrong on urban data)
        if cands.shape[1] >= 2:
            agree = np.all(cands == cands[:, :1], axis=1)
            if int(agree.sum()) >= 2:
                H = self._adopt_map(prob, {
                    pis[k]: int(cands[k, 0])
                    for k in range(len(pis)) if agree[k]})
                if H:
                    out.append(H)
                    seen.add(_canon_basin_id(H))
        for ci in range(cands.shape[1]):
            H = self._adopt_map(prob, {
                pis[k]: int(cands[k, ci]) for k in range(len(pis))})
            if H:
                bid = _canon_basin_id(H)
                if bid not in seen:
                    seen.add(bid)
                    out.append(H)
        return out

    def _shrink_worst(self, H, prob, dd=None):
        """Release-subset move: drop the held sat whose CP row misfits
        worst this epoch (design step-3 'release subset s' class)."""
        if dd is None:
            dd = self._pair_dd(H, prob)
        rows = np.flatnonzero(prob.rows_iscp)
        pairs = prob.rows_pair[rows]
        valid = np.isfinite(dd[pairs])
        rows, pairs = rows[valid], pairs[valid]
        if len(rows) == 0:
            return None
        wr = np.abs(prob.rows_lam[rows] * (prob.rows_bmeas[rows] - dd[pairs])) \
            / np.maximum(prob.rows_sig[rows], np.float64(1e-6))
        # The scalar strict comparison skips NaNs and keeps the first tie,
        # including the first +inf. argmax has the same tie order.
        wr = np.where(np.isnan(wr), -1.0, wr)
        worst = int(np.argmax(wr))
        if wr[worst] < 0:
            return None
        worst_pi = pairs[worst]
        g, ref, j = prob.pairs[worst_pi]
        H2 = _copy_H(H)
        zmap = H2.get(g)
        if zmap is None or j not in zmap:
            return None
        zmap.pop(j, None)
        if len(zmap) < 2:
            H2.pop(g, None)
        if not H2:
            return None
        return H2

    # -- resampling ------------------------------------------------------------

    def _maybe_resample(self, logw):
        n = len(logw)
        w = np.exp(logw - np.max(logw))
        w /= w.sum()
        ess = 1.0 / float(np.sum(w ** 2))
        if ess >= self.cfg.ess_frac * n:
            return None, ess
        # stratified-by-basin: sort particles by basin id, systematic sample
        order = sorted(range(n),
                       key=lambda i: (_canon_basin_id(self.particles[i].H), i))
        ws = w[order]
        c = np.cumsum(ws)
        u0 = self.rng.random() / n
        picks = []
        j = 0
        for i in range(n):
            u = u0 + i / n
            while c[j] < u and j < n - 1:
                j += 1
            picks.append(order[j])
        self.n_resamples += 1
        return picks, ess

    # -- WP17 item 1: basin feedback (COND_HOLD loop) ---------------------------

    def _feedback(self, rtk, prob, w, dd_map, used_pairs,
                  rep_dpr=float('nan'), ddpr_qual=float('nan'),
                  witness_chi=float('nan'), dd_all=None):
        """Close the loop: fold gamma-gated consensus DD integers back
        into the shared graph as conditioned holds (the WP13r COND_HOLD
        machinery: `rtk._committed` + `_inject_hold` one-time tight
        prior + the `_write_back_tc` held mirror; requires COND_HOLD=1
        in the shared-filter env), and RELEASE holds when the PF's own
        posterior withdraws its mass (per-pair agreement mass on the
        HELD integer < fb_rel_mass for fb_rel_m consecutive epochs), or
        when the (sat,f) ages out of the PF support -- no ratio/probation
        ladders anywhere.

        Gauge note: holds are SD values; only DDs are observable, so the
        per-group gauge is anchored on the current float estimate of the
        first committed sat of the group (later commits reuse committed
        values, keeping every held DD exactly integer)."""
        cfg = self.cfg
        committed = rtk._committed

        # ---- support-collapse rejuvenation. A coherent wrong parent can
        # remain self-consistent, so posterior mass does not withdraw and
        # the ordinary mass release never fires. If the MAP basin no longer
        # has enough independent DD constraints to justify carrying its
        # conditioned graph state, regenerate every held ambiguity and
        # re-enter through the existing spawner.
        map_support = (int(np.isfinite(dd_map).sum())
                       if dd_map is not None else 0)
        if cfg.fb_support_min > 0 and committed:
            if map_support < cfg.fb_support_min:
                self.fb_support_bad_streak += 1
            else:
                self.fb_support_bad_streak = 0
            if self.fb_support_bad_streak >= cfg.fb_support_m:
                for key in list(committed.keys()):
                    rtk._release_ambiguity(key[0], key[1],
                                           reason='rbpf_support')
                    self.fb_release_count += 1
                self.fb_release_support += 1
                self.fb_bad.clear()
                self.fb_streak.clear()
                self.cluster_states.clear()
                self.fb_support_bad_streak = 0
                if cfg.spawn_enable:
                    self.spawn_remaining = max(self.spawn_remaining,
                                               cfg.spawn_epochs)
                    self.spawn_events += 1
        elif not committed:
            self.fb_support_bad_streak = 0

        # ---- WP18 failure mode (a): DDPR-informed hold release. The
        # code-only DDPR position does not share a coherent CP shift, so
        # a persistent reported-vs-DDPR disagreement while holds are in
        # place marks a wrong-basin lock the mass-drop release cannot
        # see (the whole posterior is in the shifted basin). Releasing
        # ALL conditioned holds also drops each (sat,f) from every
        # particle (next maintenance pass), so the PF re-enters from
        # fresh posterior/spawn candidates.
        ddpr_ok = (np.isfinite(rep_dpr) and np.isfinite(ddpr_qual)
                   and ddpr_qual < cfg.ddpr_qual_max)
        if cfg.fb_ddpr_rel_d > 0:
            if not committed:
                self.ddpr_bad_streak = 0
            elif ddpr_ok:
                if rep_dpr > cfg.fb_ddpr_rel_d:
                    self.ddpr_bad_streak += 1
                else:
                    self.ddpr_bad_streak = 0
            # untrusted-DDPR epochs freeze the streak (canyon regime:
            # a multipath-corrupted DDPR must not release good holds)
            if committed and self.ddpr_bad_streak >= cfg.fb_ddpr_rel_m:
                for key in list(committed.keys()):
                    rtk._release_ambiguity(key[0], key[1],
                                           reason='rbpf_ddpr')
                    self.fb_release_count += 1
                self.fb_release_ddpr += 1
                self.fb_bad.clear()
                self.fb_streak.clear()
                self.ddpr_bad_streak = 0
                if cfg.spawn_enable:
                    # the float was pinned by the wrong holds; re-enter
                    # like a post-outage float
                    self.spawn_remaining = max(self.spawn_remaining,
                                               cfg.spawn_epochs)
                    self.spawn_events += 1

        # The caller may supply its component-agreement DDs: particle H
        # stays unchanged until the maintenance/resampling passes.
        if dd_all is None:
            dd_all = [self._pair_dd(p.H, prob) for p in self.particles]

        # Match integers in bulk; keep the original ordered weight additions.
        mass_dd = None
        if (committed and isinstance(dd_all, (list, tuple)) and dd_all
                and len(dd_all) == len(self.particles)
                and all(isinstance(d, np.ndarray) and d.dtype == np.float64
                        and d.shape == (len(prob.pairs),) for d in dd_all)):
            mass_dd = np.trunc(np.asarray(dd_all))

        # ---- release pass: per held pair, posterior mass on the HELD DD
        group_held = {}
        group_bad = {}
        held_pairs = []
        for pi, (g, ref, j) in enumerate(prob.pairs):
            f = g[1]
            vr = committed.get((int(ref), f))
            vj = committed.get((int(j), f))
            if vr is None or vj is None:
                continue
            z_held = int(np.rint(vr - vj))
            mass = 0.0
            if mass_dd is not None and abs(z_held) <= 2**53:
                for i_ in np.flatnonzero(mass_dd[:, pi] == z_held):
                    mass += float(w[i_])
            else:
                for i_ in range(len(self.particles)):
                    di = dd_all[i_][pi]
                    if np.isfinite(di) and int(di) == z_held:
                        mass += float(w[i_])
            held_pairs.append((g, ref, j, mass))
            group_held[g] = group_held.get(g, 0) + 1
            if mass < cfg.fb_rel_mass:
                group_bad[g] = group_bad.get(g, 0) + 1
        bad_keys = set()
        for (g, ref, j, mass) in held_pairs:
            f = g[1]
            if mass < cfg.fb_rel_mass:
                bad_keys.add((int(j), f))
                # majority of the group's held pairs bad -> blame the
                # reference (common-sat corruption / gauge broken)
                if group_bad.get(g, 0) >= max(2, group_held[g] // 2 + 1):
                    bad_keys.add((int(ref), f))
        for key in list(committed.keys()):
            if key in bad_keys:
                self.fb_bad[key] = self.fb_bad.get(key, 0) + 1
            elif key in self.fb_bad and key not in bad_keys:
                # only clear when the pair was measurable this epoch;
                # an unseen pair keeps its streak frozen
                seen_now = any((int(jj), gg[1]) == key or (int(rr), gg[1]) == key
                               for (gg, rr, jj, _m) in held_pairs)
                if seen_now:
                    self.fb_bad.pop(key, None)
        for key in list(committed.keys()):
            s, f = int(key[0]), int(key[1])
            unseen = (s, f) in self.fb_unseen_pending
            if not unseen and (s, f) in self.carry_protect:
                # WP52: the carry still fits this integer on the raw
                # carrier; the PF's mass dip alone does not release it
                continue
            if unseen or self.fb_bad.get((s, f), 0) >= cfg.fb_rel_m:
                rtk._release_ambiguity(
                    s, f, reason='rbpf_unseen' if unseen else 'rbpf_mass')
                self.fb_bad.pop((s, f), None)
                self.fb_release_count += 1
                if unseen:
                    self.fb_release_unseen += 1
                else:
                    self.fb_release_mass += 1
        self.fb_unseen_pending &= set(committed.keys())

        # ---- commit pass: gamma-gated consensus pairs, M consecutive epochs
        commit_pairs = list(used_pairs)
        if cfg.fb_dual_enable:
            dual_pairs = []
            cands = (prob.witness_cands if prob.witness_cands is not None
                     else prob.cands)
            if (dd_map is not None and cands is not None
                    and cands.ndim == 2 and cands.shape[0] == len(prob.pairs)):
                for pi in used_pairs:
                    shared = cands[pi]
                    if (len(shared) > 0
                            and np.all(shared == shared[0])
                            and int(shared[0]) == int(dd_map[pi])):
                        dual_pairs.append(pi)
            commit_pairs = dual_pairs
            if len(commit_pairs) < cfg.fb_dual_min:
                self.fb_dual_gated += 1

        new_streak = {}
        commit_min = (cfg.fb_dual_min if cfg.fb_dual_enable
                      else cfg.fb_min_nb)
        if dd_map is not None and len(commit_pairs) >= commit_min:
            for pi in commit_pairs:
                g, ref, j = prob.pairs[pi]
                kkey = (g, int(ref), int(j), int(dd_map[pi]))
                new_streak[kkey] = self.fb_streak.get(kkey, 0) + 1
        self.fb_streak = new_streak
        # The same support certificate gates new graph mutation. Waiting
        # until a low-support parent is already conditioned is too late:
        # its old position/IMU states persist for the fixed lag even after
        # ambiguity regeneration. Streak evidence is retained, but no new
        # tight hold enters the graph until support recovers.
        if (cfg.fb_support_min > 0
                and map_support < cfg.fb_support_min):
            return len(committed)
        if (cfg.fb_witness_pos and
                (not np.isfinite(witness_chi)
                 or witness_chi > cfg.fb_witness_chi)):
            self.fb_witness_gated += 1
            return len(committed)
        # WP18 (a): commit gate -- do not fold NEW holds into the graph
        # on an epoch whose reported position disagrees with a trusted
        # DDPR position (the lock-in entry mechanism was holds committed
        # during a float excursion in self-consistent multipath).
        # Streaks keep accruing; the commit fires on the first clean
        # epoch instead.
        if (cfg.fb_commit_max_dpr > 0 and ddpr_ok
                and rep_dpr > cfg.fb_commit_max_dpr):
            self.fb_commit_gated += 1
            return len(committed)
        newly = []
        for (g, ref, j, z), n in self.fb_streak.items():
            if n < cfg.fb_m:
                continue
            f = g[1]
            kr, kj = (int(ref), f), (int(j), f)
            if kr in committed and kj in committed:
                continue
            if (ref, f) not in rtk.amb_keys or (j, f) not in rtk.amb_keys:
                continue
            if kr in committed:
                vref = float(committed[kr])
            elif kj in committed:
                vref = float(committed[kj]) + z
            else:
                vref = float(rtk.nav.x[rtk.IB(int(ref), f, rtk.nav.na)])
                if not np.isfinite(vref):
                    continue
            vj = vref - z
            for key, val in ((kr, vref), (kj, vj)):
                if key not in committed:
                    committed[key] = float(val)
                    newly.append(key)
        if newly:
            xa = np.asarray(rtk.nav.x, dtype=float).copy()
            for key in newly:
                xa[rtk.IB(key[0], key[1], rtk.nav.na)] = committed[key]
            rtk._inject_hold(xa, newly)
            self.fb_commit_count += len(newly)
        return len(committed)

    def carry_feedback(self, rtk, state):
        """WP52 (RBPF_CARRY_FB): hold the label layer's carried integer set
        in the shared graph, and keep the PF's mass release off those
        pairs while the carry lasts. `state` is IntegerCarry.state,
        {(sys, f): {sat: m}} with DD(ref, j) = m[ref] - m[j], the graph's
        SD_ref - SD_j convention; None ends the protection.

        Per group, the gauge is a pair the graph already holds (all held
        pairs of the group must agree with the carry, else the group is
        left to the PF), or the float SD of its first satellite."""
        if not state:
            self.carry_protect = set()
            return 0
        self.carry_fb_epochs += 1
        committed = rtk._committed
        protect = set()
        newly = []
        for (_sys, f), m in state.items():
            f = int(f)
            keys = [(int(sv), f) for sv in m if (int(sv), f) in rtk.amb_keys]
            if len(keys) < 2:
                continue
            held = [k for k in keys if k in committed]
            if held:
                a = held[0]
                va = float(committed[a])
                if any(abs((va - float(committed[k])) - (m[a[0]] - m[k[0]])) > 0.5
                       for k in held[1:]):
                    self.carry_fb_skipped += 1
                    continue
            else:
                a = keys[0]
                va = float(rtk.nav.x[rtk.IB(a[0], f, rtk.nav.na)])
                if not np.isfinite(va):
                    continue
            for k in keys:
                protect.add(k)
                if k not in committed:
                    committed[k] = va - (m[a[0]] - m[k[0]])
                    newly.append(k)
        self.carry_protect = protect
        if newly:
            xa = np.asarray(rtk.nav.x, dtype=float).copy()
            for k in newly:
                xa[rtk.IB(k[0], k[1], rtk.nav.na)] = committed[k]
            rtk._inject_hold(xa, newly)
            self.carry_fb_commits += len(newly)
        return len(newly)

    # -- one epoch ---------------------------------------------------------------

    def step(self, rtk, ctx):
        t0 = time.perf_counter()
        cfg = self.cfg
        ep = rtk.epoch2
        self.n_steps += 1
        self._conditional_cache = {}

        prob = self._extract(rtk, ctx)
        self._last_prob = prob
        self._attach_witness(prob, getattr(rtk, '_rbpf_witness_prob', None))
        # maintenance AFTER extraction support known (uses today's support)
        support_sf = set()
        if prob.ok:
            for (g, ref, j) in prob.pairs:
                support_sf.add((ref, g[1]))
                support_sf.add((j, g[1]))
        _tow_m = None
        if cfg.keep:
            try:
                _, _tow_m = time2gpst(ctx['obs'].t)
            except (KeyError, AttributeError, TypeError, ValueError):
                _tow_m = None
        self._maintenance(rtk, support_sf, ep, _tow_m)

        out = {'ep': ep, 'ok': prob.ok, 'reason': prob.reason,
               'fix': False, 'gamma': 0.0, 'nb': 0, 'ess': float('nan'),
               'n_basins': 0, 'pos': None, 'npairs': 0,
               'second_gamma': 0.0, 'top': [], 'nheld': 0, 'spawn': 0,
               # WP18 per-epoch diagnostics (nan when not computable)
               'ddpr_dpos': float('nan'), 'ddpr_qual': float('nan'),
               'fix_dpr': float('nan'), 'min_info': float('nan'),
               'rep_dpr': float('nan'), 'loo_max_m': float('nan'),
               'loo_med_m': float('nan'), 'loo_n': 0,
               'delay_5s_m': float('nan'), 'delay_20s_m': float('nan'),
               'delay_corr_cos': float('nan'),
               'delay_cross_cos': float('nan'),
               'delayed_rejuv_trigger': 0, 'shadow_promote': 0,
               'shadow_n': len(self.shadow_challengers)}

        # WP17 item 3: outage / re-entry bookkeeping. A "usable" epoch
        # has a problem with >= spawn_min_cp CP-carrying pairs; after
        # spawn_outage consecutive unusable epochs, the next
        # spawn_epochs usable epochs add DDPR-prior spawn moves.
        n_cp = 0
        if prob.ok:
            n_cp = len({int(p) for p in prob.rows_pair[prob.rows_iscp]})
        spawn_H = []
        if cfg.spawn_enable:
            # raw data/time gap (tunnel-style dropout: the runner never
            # steps the PF on missing epochs, so count wall time too)
            try:
                _, _tow_now = time2gpst(ctx['obs'].t)
                if (self._last_tow is not None
                        and _tow_now - self._last_tow > cfg.spawn_gap_s):
                    self.no_cp_streak = max(self.no_cp_streak,
                                            cfg.spawn_outage)
                self._last_tow = float(_tow_now)
            except Exception:
                pass
            usable = prob.ok and n_cp >= cfg.spawn_min_cp
            if not usable:
                self.no_cp_streak += 1
                self.fixless += 1
            else:
                if self.no_cp_streak >= cfg.spawn_outage:
                    self.spawn_remaining = cfg.spawn_epochs
                    self.spawn_events += 1
                self.no_cp_streak = 0
                if (self.spawn_remaining <= 0
                        and self.fixless >= cfg.spawn_after):
                    self.spawn_remaining = 1   # rolling desperation spawn
                    if self.fixless == cfg.spawn_after:
                        self.spawn_events += 1
                if self.spawn_remaining > 0:
                    self.spawn_remaining -= 1
                    spawn_H = self._spawn_candidates(prob)
                elif (self.cfg.chal_d > 0
                      and self.chal_streak >= self.cfg.chal_m):
                    # WP18 (a): challenger spawn -- persistent trusted
                    # fix-vs-DDPR disagreement; bypass the sleep AND the
                    # min_dpos gate (see RbpfConfig.chal_d)
                    spawn_H = self._spawn_candidates(prob, force=True)
                    if spawn_H:
                        self.chal_spawn_events += 1
                if self.delayed_rejuv_remaining > 0:
                    forced = self._spawn_candidates(prob, force=True)
                    if forced:
                        spawn_H = forced
                    self.delayed_rejuv_remaining -= 1
        out['spawn'] = len(spawn_H)

        if not prob.ok:
            self.n_skipped += 1
            self.last = out
            self.step_seconds += time.perf_counter() - t0
            return out
        out['npairs'] = len(prob.pairs)
        self.n_meas_updates += 1

        # WP18: per-epoch DDPR position + quality (memoized; the
        # ambiguity-independent code-only signal used by failure-mode
        # (a) commit gating / hold release and the spawner)
        ddpr_got = self._ddpr_offset(prob)
        ddpr_pos = None
        if ddpr_got is not None:
            ddpr_pos = prob.pos + ddpr_got[0][:3]
            out['ddpr_dpos'] = float(np.linalg.norm(ddpr_got[0][:3]))
            try:
                Pp = ddpr_got[1]
                out['ddpr_qual'] = math.sqrt(max(float(np.max(
                    np.linalg.eigvalsh(0.5 * (Pp + Pp.T)))), 0.0))
            except np.linalg.LinAlgError:
                pass

        if os.environ.get('RBPF_DBG'):
            cp = prob.rows_iscp
            if np.any(cp):
                pi_cp = prob.rows_pair[cp]
                dby = prob.rows_bmeas[cp] - prob.y[pi_cp]
                print(f"RBPF_DBG ep={ep} npairs={len(prob.pairs)} "
                      f"ncp={int(cp.sum())} med|b-y|={np.median(np.abs(dby)):.3f} "
                      f"p90|b-y|={np.percentile(np.abs(dby),90):.3f} "
                      f"med_sig_cp={np.median(prob.rows_sig[cp]):.4f} "
                      f"med_sqrtQb={np.median(np.sqrt(np.maximum(np.diag(prob.Qb),0))):.3f} "
                      f"K={0 if prob.cands is None else prob.cands.shape[1]}")

        # ---- fully-adapted move + weight update
        cache = {}
        # per-epoch burst regime: the all-CP-flat score is exactly the
        # empty hypothesis's normal score (uniform rows + PR marginal)
        ll_burst = self._loglik({}, prob, cache)
        _lp_nb = math.log(max(1.0 - cfg.p_burst, 1e-12))
        _lp_b = math.log(max(cfg.p_burst, 1e-12)) + ll_burst

        def LL(H):
            return float(np.logaddexp(
                _lp_nb + self._loglik(H, prob, cache), _lp_b))

        # WP37: challengers observe future epochs without changing production
        # particles.  Compare each against the best currently represented
        # production hypothesis under exactly the same predictive likelihood.
        if cfg.shadow_epochs > 0:
            if self.shadow_pending:
                staged = self._spawn_candidates(prob, force=True)
                known = {_canon_basin_id(s['H']) for s in self.shadow_challengers}
                for Hs in staged:
                    bid = _canon_basin_id(Hs)
                    if bid not in known:
                        self.shadow_challengers.append(
                            {'H': _copy_H(Hs), 'score': 0.0, 'age': 0})
                        known.add(bid)
                self.shadow_pending = False
            if self.shadow_challengers:
                prod_ll = max(LL(p.H) for p in self.particles)
                survivors = []
                promoted = []
                for shadow in self.shadow_challengers:
                    shadow['score'] += LL(shadow['H']) - prod_ll
                    shadow['age'] += 1
                    if shadow['age'] >= cfg.shadow_epochs:
                        if shadow['score'] > cfg.shadow_margin:
                            promoted.append(shadow['H'])
                    else:
                        survivors.append(shadow)
                self.shadow_challengers = survivors
                if promoted:
                    spawn_H = promoted
                    self.shadow_promotions += len(promoted)
                    out['shadow_promote'] = len(promoted)
            out['shadow_n'] = len(self.shadow_challengers)

        K = prob.cands.shape[1] if prob.cands is not None else 0
        cons = self._consensus_subset(prob) if K else None
        H_cons = self._adopt(prob, 0, subset=cons) if cons else None
        cand_H = [self._adopt(prob, k) for k in range(K)]
        cand_H = [Hk for Hk in cand_H if Hk]

        if os.environ.get('RBPF_DBG3'):
            cp = prob.rows_iscp
            if np.any(cp):
                pis = prob.rows_pair[cp]
                dby = np.abs(prob.rows_bmeas[cp] - prob.y[pis])
                order = np.argsort(-dby)[:3]
                info = []
                for oi in order:
                    g, ref, j = prob.pairs[pis[oi]]
                    age_r = ep - rtk._amb_init_epoch.get((ref, g[1]), -999)
                    age_j = ep - rtk._amb_init_epoch.get((j, g[1]), -999)
                    info.append(f"(g={g} ref={ref}/age{age_r} j={j}/age{age_j} "
                                f"|b-y|={dby[oi]:.1f})")
                print(f"RBPF_DBG3 ep={ep} med|b-y|={np.median(dby):.3f} "
                      f"max3: {' '.join(info)}")
        if os.environ.get('RBPF_DBG4') and getattr(rtk, '_rbpf_truth_ecef', None) is not None:
            # ground-truth basin: integers from rounding the truth-anchored
            # single-epoch DD-CP float
            truth = np.asarray(rtk._rbpf_truth_ecef, dtype=float)
            dxt = truth - prob.pos
            H_true = {}
            zt = {}
            for ri in range(len(prob.rows_v0)):
                if not prob.rows_iscp[ri]:
                    continue
                p_i = prob.rows_pair[ri]
                b_true = prob.rows_bmeas[ri] - prob.rows_J[ri] @ dxt / prob.rows_lam[ri]
                zt[p_i] = int(np.rint(b_true))
            for p_i, zv in zt.items():
                g, ref, j = prob.pairs[p_i]
                zmap = H_true.setdefault(g, {})
                zmap.setdefault(ref, 0)
                zmap[j] = int(zmap[ref] - zv)
            ll_t = self._loglik(H_true, prob, cache) if H_true else float('nan')
            ll_e4 = self._loglik({}, prob, cache)
            ham1 = -1
            hammin = -1
            if prob.cands is not None and zt:
                z1 = prob.cands[:, 0]
                ham1 = sum(1 for p_i, zv in zt.items() if int(z1[p_i]) != zv)
                hammin = min(sum(1 for p_i, zv in zt.items()
                                 if int(prob.cands[p_i, k]) != zv)
                             for k in range(prob.cands.shape[1]))
            # best currently held basin (before this epoch's moves) and
            # how many truth pairs it gets right / wrong
            lw = np.array([p_.logw for p_ in self.particles])
            pbest = self.particles[int(np.argmax(lw))]
            ll_m = self._loglik(pbest.H, prob, cache) if pbest.H else ll_e4
            ddb = self._pair_dd(pbest.H, prob) if pbest.H else None
            nok = nbad = 0
            if ddb is not None:
                for p_i, zv in zt.items():
                    if np.isfinite(ddb[p_i]):
                        nok += int(int(ddb[p_i]) == zv)
                        nbad += int(int(ddb[p_i]) != zv)
            dflt = float(np.linalg.norm(dxt))
            print(f"RBPF_DBG4 ep={ep} ll_true={ll_t:.1f} ll_empty={ll_e4:.1f} "
                  f"gap={ll_t-ll_e4:.1f} ncp={len(zt)} ham_top1={ham1} "
                  f"ham_min={hammin} ll_best={ll_m:.1f} best_ok={nok} "
                  f"best_bad={nbad} flt_err={dflt:.2f}")
        if os.environ.get('RBPF_DBG2'):
            ll_e = self._loglik({}, prob, cache)
            ll_c = (self._loglik(H_cons, prob, cache)
                    if H_cons is not None else float('nan'))
            ll_1 = (self._loglik(cand_H[0], prob, cache)
                    if cand_H else float('nan'))
            ncons = len(cons) if cons else 0
            print(f"RBPF_DBG2 ep={ep} ll_empty={ll_e:.1f} "
                  f"ll_cons={ll_c:.1f} (ncons={ncons}) ll_top1={ll_1:.1f} "
                  f"npairs={len(prob.pairs)}")

        new_states = []
        if cfg.track == 1:
            new_states = self._moves_tracked(
                prob, ctx, ep, H_cons, cand_H, spawn_H, _lp_nb, _lp_b)
        shared_moves = []
        if cfg.track != 1 and self.particles:
            # These hypotheses and their float prior are shared by all
            # particles. Score once per epoch; retain the original move
            # order and keep particle-dependent survival/extension separate.
            ll_empty = float(np.logaddexp(_lp_nb + ll_burst, _lp_b))
            if H_cons is not None:
                shared_moves.append((cfg.p_adopt_cons, LL(H_cons),
                                     H_cons, 'adopt_cons'))
            for Hk in cand_H:
                shared_moves.append((cfg.p_adopt / len(cand_H), LL(Hk),
                                     Hk, 'adopt'))
            for Hs in spawn_H:
                shared_moves.append((cfg.p_spawn / len(spawn_H), LL(Hs),
                                     Hs, 'spawn'))
        move_logpri = {}
        for p in (self.particles if cfg.track != 1 else ()):
            moves = []           # (logprior, loglik, H, tag)
            prior_key = 0
            ll_s = LL(p.H)
            if p.H:
                prior_key = 1
                moves.append((cfg.p_survive, ll_s, p.H, 'survive'))
                H_ext, H_shr = self._move_pair(p.H, prob)
                if H_ext is not None:
                    prior_key |= 2
                    moves.append((cfg.p_extend, LL(H_ext), H_ext, 'extend'))
                if H_shr is not None:
                    prior_key |= 4
                    moves.append((cfg.p_shrink, LL(H_shr), H_shr, 'shrink'))
                moves.append((cfg.p_release, ll_empty, {}, 'release'))
            else:
                moves.append((cfg.p_survive + cfg.p_extend + cfg.p_shrink
                              + cfg.p_release, ll_s, p.H, 'stay'))
            moves.extend(shared_moves)
            logpri = move_logpri.get(prior_key)
            if logpri is None:
                pri = np.array([mv[0] for mv in moves], dtype=float)
                pri /= pri.sum()     # renormalize over available moves
                logpri = np.log(pri)
                move_logpri[prior_key] = logpri
            lam_arr = logpri + np.array([mv[1] for mv in moves])
            mx = float(np.max(lam_arr))
            wsum = np.exp(lam_arr - mx)
            tot = float(np.sum(wsum))
            p.logw += mx + math.log(tot)
            pick = int(self.rng.choice(len(moves), p=wsum / tot))
            new_states.append((moves[pick][2], moves[pick][3]))

        for p, (H, tag) in zip(self.particles, new_states):
            p.H = _copy_H(H)
            p.last_move = tag
        if cfg.track == 2:
            self._shadow_track(prob, ctx, ep)

        if os.environ.get('RBPF_DBG2'):
            from collections import Counter
            tags = Counter(t for _, t in new_states)
            szs = [sum(len(z) for z in H.values()) for H, _ in new_states]
            print(f"RBPF_DBG2b ep={ep} moves={dict(tags)} "
                  f"Hsize med={np.median(szs):.0f} max={max(szs)}")

        # normalize log-weights (shift only; relative masses preserved)
        logw = np.array([p.logw for p in self.particles])
        shift = float(np.max(logw))
        for p in self.particles:
            p.logw -= shift
        logw -= shift

        # ---- basin posterior
        w = np.exp(logw)
        w /= w.sum()
        masses = OrderedDict()
        basin_ids = []
        rep = {}                # basin id -> first representative H
        for i, p in enumerate(self.particles):
            bid = _canon_basin_id(p.H)
            basin_ids.append(bid)
            masses[bid] = masses.get(bid, 0.0) + float(w[i])
            rep.setdefault(bid, p.H)
        ranked = sorted(masses.items(), key=lambda kv: -kv[1])
        out['n_basins'] = len(ranked)

        top = []
        next_cluster_states = {}
        for bid, mass in ranked[:5]:
            H = rep[bid]
            xz, nb_used, _ = self._conditional_mean(H, prob)
            top.append({'mass': mass, 'nb': nb_used,
                        'empty': len(bid) == 0,
                        'pos': None if xz is None else xz.copy()})
            cached = self._conditional_cache.get(bid)
            if (cfg.cluster_relin and len(next_cluster_states) < cfg.cluster_relin_top
                    and xz is not None and len(bid) != 0):
                Pz = prob.Ppos if cached is None else cached[3]
                next_cluster_states[bid] = {
                    'H': _copy_H(H),
                    'pos': np.asarray(xz, dtype=float).copy(),
                    'P': np.asarray(Pz, dtype=float).copy(),
                }
        out['top'] = top

        gamma = ranked[0][1] if ranked else 0.0
        out['gamma'] = float(gamma)
        out['second_gamma'] = float(ranked[1][1]) if len(ranked) > 1 else 0.0
        map_H = rep[ranked[0][0]] if ranked else {}
        if cfg.track == 2 and len(cfg.track_variants) > 1:
            var_out = []
            for v in range(len(cfg.track_variants)):
                xs, ws = [], []
                for i_, p_ in enumerate(self.particles):
                    st_ = p_.state if v == 0 else (
                        p_.vstates[v - 1] if len(p_.vstates) >= v else None)
                    if (st_ is not None and ranked
                            and basin_ids[i_] == ranked[0][0]):
                        xs.append(st_.x)
                        ws.append(float(w[i_]))
                if xs and sum(ws) > 0:
                    var_out.append((np.average(np.asarray(xs), axis=0,
                                               weights=np.asarray(ws)),
                                    float(sum(ws))))
                else:
                    var_out.append((None, float('nan')))
            out['trk_var'] = var_out
        if cfg.track:
            # WP43: tracked beliefs of the MAP basin and of every non-empty
            # particle (mass-weighted), recorded for output studies
            # H is unchanged since the posterior grouping above.
            for key, sel in (('trk_map', lambda i_, p_: basin_ids[i_]
                              == ranked[0][0]),
                             ('trk_mix', lambda i_, p_: bool(p_.H))):
                xs, ws = [], []
                for i_, p_ in enumerate(self.particles):
                    if p_.state is not None and ranked and sel(i_, p_):
                        xs.append(p_.state.x)
                        ws.append(float(w[i_]))
                if xs and sum(ws) > 0:
                    out[key] = np.average(np.asarray(xs), axis=0,
                                          weights=np.asarray(ws))
                    out[key + '_mass'] = float(sum(ws))

        # ---- fix decision: PER-COMPONENT posterior agreement mass.
        # Basin-level mass splits among near-identical basins (a basin,
        # its subsets and its extensions are distinct ids), which makes
        # basin-gamma under-confident about the COMPONENTS they all agree
        # on (measured: MAP basin 97% correct already at basin-gamma
        # 0.5-0.9). The calibrated quantity for fixing component d is
        # sum of the mass of particles whose basin implies the same DD
        # integer; fix on the >gamma* subset (partial fix), which is the
        # posterior-mass version of partial AR. Empty/undefined
        # particles count AGAINST agreement.
        dd_all = None
        dd_map = self._pair_dd(map_H, prob) if map_H else None
        pair_mass = {}
        used_pairs = []
        gamma_min_used = 0.0
        if dd_map is not None:
            dd_all = [self._pair_dd(p.H, prob) for p in self.particles]
            pair_mass = {}
            for pi_ in np.where(np.isfinite(dd_map))[0]:
                am = 0.0
                for i_, p in enumerate(self.particles):
                    di = dd_all[i_][pi_]
                    if np.isfinite(di) and int(di) == int(dd_map[pi_]):
                        am += float(w[i_])
                pair_mass[int(pi_)] = am
            used_pairs = [pi_ for pi_, am in pair_mass.items()
                          if am > cfg.gamma_fix]
            if used_pairs:
                gamma_min_used = min(pair_mass[pi_] for pi_ in used_pairs)
        out['nb'] = int(len(used_pairs))
        out['gamma_pair'] = float(gamma_min_used)
        # WP44 diagnostic: MAP pairs whose agreement mass clears each of
        # NB_AT_THRESH (offline FIX-rule studies; nothing reads it)
        out['nb_at'] = ([int(sum(1 for am in pair_mass.values() if am > t))
                         for t in NB_AT_THRESH]
                        if dd_map is not None else [0] * len(NB_AT_THRESH))
        out['ddpr_xyz'] = ddpr_pos
        # WP44 diagnostic: the same counts for the best NON-EMPTY basin
        # with the mass renormalised over the non-empty particles, plus
        # that non-empty mass (the flat 'no integer' hypothesis can hold
        # most of the mass through a multipath burst while every integer
        # hypothesis agrees)
        ne_idx = [i_ for i_, p_ in enumerate(self.particles) if p_.H]
        ne_mass = float(sum(w[i_] for i_ in ne_idx))
        out['ne_mass'] = ne_mass
        out['nb_at_ne'] = [0] * len(NB_AT_THRESH)
        ne_ranked = [(bid, m_) for bid, m_ in ranked if len(bid) != 0]
        if ne_ranked and ne_mass > 0:
            # Hypotheses stay unchanged throughout component agreement.
            # Reuse this phase's DD arrays for the non-empty diagnostic.
            dd_ne = (dd_map if dd_map is not None and ne_ranked[0][0] == ranked[0][0]
                     else self._pair_dd(rep[ne_ranked[0][0]], prob))
            dd_l = {i_: (dd_all[i_] if dd_all is not None
                         else self._pair_dd(self.particles[i_].H, prob))
                    for i_ in ne_idx}
            masses_ne = []
            for pi_ in np.where(np.isfinite(dd_ne))[0]:
                am = sum(float(w[i_]) for i_ in ne_idx
                         if np.isfinite(dd_l[i_][pi_])
                         and int(dd_l[i_][pi_]) == int(dd_ne[pi_]))
                masses_ne.append(am / ne_mass)
            out['nb_at_ne'] = [int(sum(1 for am in masses_ne if am > t))
                               for t in NB_AT_THRESH]
        xz = None
        if len(used_pairs) >= cfg.min_fix_nb:
            # sub-basin over the agreed pairs only
            H_sub = {}
            for pi_ in used_pairs:
                g, ref, j = prob.pairs[pi_]
                zmap = H_sub.setdefault(g, {})
                zmap.setdefault(ref, 0)
                zmap[j] = int(zmap[ref] - int(dd_map[pi_]))
            xz, _, fix_info = self._conditional_mean(H_sub, prob)
            out['min_info'] = float(fix_info)
            if xz is not None and cfg.track and cfg.track_out == 1:
                # WP43: mass-weighted tracked belief of the particles that
                # carry every reported integer
                xs, ws = [], []
                for i_, p in enumerate(self.particles):
                    if p.state is None:
                        continue
                    di = dd_all[i_]
                    if all(np.isfinite(di[pi_])
                           and int(di[pi_]) == int(dd_map[pi_])
                           for pi_ in used_pairs):
                        xs.append(p.state.x)
                        ws.append(float(w[i_]))
                if xs and sum(ws) > 0.5:
                    xz = np.average(np.asarray(xs), axis=0,
                                    weights=np.asarray(ws))
            if cfg.loo_enable:
                loo = self._loo_consistency(prob, dd_map, used_pairs)
                if loo.size:
                    out['loo_max_m'] = float(np.max(loo))
                    out['loo_med_m'] = float(np.median(loo))
                    out['loo_n'] = int(loo.size)
            if xz is not None and ddpr_pos is not None:
                out['fix_dpr'] = float(np.linalg.norm(xz - ddpr_pos))
            # WP17: ambiguity-independent DDPR cross-check on the OUTPUT
            # (see RbpfConfig.fix_max_dpr) -- reject the report, leave
            # the posterior/feedback state untouched.
            if (xz is not None and cfg.fix_max_dpr > 0
                    and np.isfinite(out['fix_dpr'])
                    and out['fix_dpr'] > cfg.fix_max_dpr):
                out['fix_dpr_reject'] = out['fix_dpr']
                xz = None
            # WP18 failure mode (b): three-way consistency vote (float /
            # DDPR / fix; see RbpfConfig.fix_vote_dd). Output layer only.
            if (xz is not None and cfg.fix_vote_dd > 0
                    and np.isfinite(out['ddpr_dpos'])
                    and out['ddpr_dpos'] > cfg.fix_vote_dd
                    and np.isfinite(out['fix_dpr'])
                    and out['fix_dpr'] > cfg.fix_vote_dpr):
                out['fix_vote_reject'] = out['fix_dpr']
                self.fix_vote_rejects += 1
                xz = None
            # WP18 failure mode (b) diagnostic variant: conditioning
            # guard on the LS normal matrix (measured NON-separating on
            # run2 -- good fixes reach min_info 0.62 while the blow-up
            # sat at 14.6 -- kept implemented but default off).
            if (xz is not None and cfg.fix_min_info > 0
                    and np.isfinite(fix_info)
                    and fix_info < cfg.fix_min_info):
                out['fix_info_reject'] = float(fix_info)
                self.fix_info_rejects += 1
                xz = None
        if xz is not None:
            out['fix'] = True
            out['pos'] = xz
            if cfg.delayed_enable:
                correction = np.asarray(xz, dtype=float) - prob.pos
                deltas = {}
                for key, delay in (('delay_5s_m', cfg.delayed_short),
                                   ('delay_20s_m', cfg.delayed_long)):
                    target = ep - delay
                    prior = next((c for e, c in reversed(self.delayed_history)
                                  if e <= target), None)
                    if prior is not None:
                        delta = correction - prior
                        deltas[key] = delta
                        out[key] = float(np.linalg.norm(delta))
                d5 = deltas.get('delay_5s_m')
                d20 = deltas.get('delay_20s_m')
                cn = np.linalg.norm(correction)
                if d5 is not None and cn > 1e-12 and out['delay_5s_m'] > 1e-12:
                    out['delay_corr_cos'] = float(
                        np.dot(d5, correction) / (out['delay_5s_m'] * cn))
                if (d5 is not None and d20 is not None
                        and out['delay_5s_m'] > 1e-12
                        and out['delay_20s_m'] > 1e-12):
                    out['delay_cross_cos'] = float(
                        np.dot(d5, d20)
                        / (out['delay_5s_m'] * out['delay_20s_m']))
                self.delayed_history.append((ep, correction.copy()))
                keep_after = ep - max(cfg.delayed_long * 2, 250)
                self.delayed_history = [(e, c) for e, c in self.delayed_history
                                        if e >= keep_after]
                drift = out['delay_5s_m']
                if cfg.delayed_rejuv_epochs > 0 and np.isfinite(drift):
                    if cfg.delayed_rejuv_edge:
                        if drift < cfg.delayed_rejuv_reset:
                            self.delayed_rejuv_armed = True
                        if (drift > cfg.delayed_rejuv_thresh
                                and self.delayed_rejuv_armed
                                and ep - self.delayed_rejuv_last_ep
                                >= cfg.delayed_rejuv_cooldown
                                and (cfg.delayed_rejuv_qual_max <= 0
                                     or (np.isfinite(out['ddpr_qual'])
                                         and out['ddpr_qual']
                                         < cfg.delayed_rejuv_qual_max))):
                            if cfg.shadow_epochs > 0:
                                self.shadow_pending = True
                            else:
                                self.delayed_rejuv_remaining = max(
                                    self.delayed_rejuv_remaining,
                                    cfg.delayed_rejuv_epochs)
                            self.delayed_rejuv_events += 1
                            self.delayed_rejuv_last_ep = ep
                            self.delayed_rejuv_armed = False
                            out['delayed_rejuv_trigger'] = 1
                    elif drift > cfg.delayed_rejuv_thresh:
                        if self.delayed_rejuv_remaining == 0:
                            self.delayed_rejuv_events += 1
                        self.delayed_rejuv_remaining = max(
                            self.delayed_rejuv_remaining,
                            cfg.delayed_rejuv_epochs)
            self.n_fix += 1
            self.fixless = 0
            self.spawn_remaining = 0    # a basin took hold; spawner sleeps
            # WP18 (a): challenger streak -- consecutive REPORTED fixes
            # whose trusted-DDPR distance exceeds chal_d (untrusted-DDPR
            # epochs freeze the streak; float epochs freeze it too)
            if cfg.chal_d > 0 and np.isfinite(out['fix_dpr']) \
                    and np.isfinite(out['ddpr_qual']) \
                    and out['ddpr_qual'] < cfg.ddpr_qual_max:
                if out['fix_dpr'] > cfg.chal_d:
                    self.chal_streak += 1
                else:
                    self.chal_streak = 0
        else:
            out['pos'] = prob.pos.copy()
            self.fixless += 1

        # WP18 (a): this epoch's reported-position-vs-DDPR distance (the
        # coherent-shift discriminant: during a wrong-basin lock the fix
        # output AND the held-pinned float both sit ~1.5 m from the
        # code-only DDPR position, persistently)
        if ddpr_pos is not None:
            rep_pos = xz if xz is not None else prob.pos
            out['rep_dpr'] = float(np.linalg.norm(
                np.asarray(rep_pos, dtype=float) - ddpr_pos))

        out['witness_chi'] = float('nan')
        wpos = getattr(rtk, '_rbpf_witness_pos', None)
        wP = getattr(rtk, '_rbpf_witness_P', None)
        if wpos is not None and wP is not None:
            try:
                rep_pos = xz if xz is not None else prob.pos
                dv = np.asarray(rep_pos, dtype=float) - np.asarray(wpos, dtype=float)
                S = (np.asarray(wP, dtype=float)
                     + np.eye(3) * cfg.fb_witness_sig_floor ** 2)
                out['witness_chi'] = float(np.sqrt(max(
                    dv @ np.linalg.solve(0.5 * (S + S.T), dv), 0.0)))
            except (np.linalg.LinAlgError, ValueError):
                pass

        # ---- WP17 item 1: basin feedback into the shared graph
        if cfg.fb_enable:
            out['nheld'] = self._feedback(
                rtk, prob, w, dd_map, used_pairs,
                rep_dpr=out['rep_dpr'], ddpr_qual=out['ddpr_qual'],
                witness_chi=out['witness_chi'], dd_all=dd_all)

        # ---- resample
        picks, ess = self._maybe_resample(logw)
        out['ess'] = float(ess)
        if picks is not None:
            new = [Particle(_copy_H(self.particles[i].H), 0.0,
                            state=self.particles[i].state,
                            vstates=self.particles[i].vstates)
                   for i in picks]
            for np_, i in zip(new, picks):
                np_.last_move = self.particles[i].last_move
            self.particles = new

        # Publish shadow states only after every conditional evaluation
        # for this epoch, so all moves/fix decisions use the same prior.
        if cfg.cluster_relin:
            self.cluster_states = next_cluster_states

        # WP43 output: the MAP basin's tracked belief replaces the report
        # (after the feedback / guards consumed the WP18 position)
        if cfg.track and cfg.track_out == 2 and out.get('trk_map') is not None:
            out['pos_wp18'] = out['pos']
            out['pos'] = np.asarray(out['trk_map'], dtype=float).copy()

        if _env_i('RBPF_WL', 0):
            out.update(self._wl_fix(prob))

        rows_ = None
        carry_ = _env_i('RBPF_CARRY', 0)
        if cfg.dump_rows or cfg.afv or carry_:
            rows_ = self._dump_rows(rtk, ctx, prob, map_H, pair_mass)
        if carry_:
            # WP50: rows for the runner's integer carry (integer_carry.py)
            out['carry_rows'] = rows_
            out['rows_pos0'] = np.asarray(prob.pos, dtype=float).copy()
        if cfg.afv and rows_ is not None and len(rows_):
            x_ = np.asarray(out['pos'], dtype=float)
            b_ = rows_[:, 10] - rows_[:, 7:10] @ (x_ - prob.pos) / rows_[:, 6]
            out['afv_n'] = len(rows_)
            out['afv_frac'] = float(np.mean(np.abs(b_ - np.round(b_))
                                            < cfg.afv_thr))
        if cfg.dump_rows:
            out['rows'] = rows_
            out['fix_xyz'] = (np.asarray(xz, dtype=float).copy()
                              if xz is not None else None)
            out['rows_pos0'] = np.asarray(prob.pos, dtype=float).copy()

        self.last = out
        self.step_seconds += time.perf_counter() - t0
        return out

    def _dump_rows(self, rtk, ctx, prob, map_H, pair_mass):
        """WP45 diagnostic rows: one per phase-observed non-GLONASS DD
        (sys, f, ref = highest-elevation satellite of the group).

        b0 is the DD carrier minus the DD geometric range at the shared
        float position prob.pos, in cycles, so at any position x the
        ambiguity-free fractional residual is frac(b0 - J.(x - pos)/lam).
        held = MAP-basin DD integer (nan when the basin does not cover the
        pair), mass = its agreement mass (nan when the pair is not one of
        the PF's pairs, i.e. not a shared-filter ambiguity state)."""
        nav = rtk.nav
        sat = ctx['sat']
        el = ctx['el']
        iu, ir, rs, rsb = ctx['iu'], ctx['ir'], ctx['rs'], ctx['rsb']
        obs_sd = ctx['obs_sd']
        pos = np.asarray(prob.pos, dtype=float)
        rb = np.array(nav.rb, dtype=float)
        pair_idx = {(g, r, j): pi for pi, (g, r, j) in enumerate(prob.pairs)}
        groups = {}
        for i in range(len(sat)):
            s = int(sat[i])
            sys_id, _ = sat2prn(s)
            if sys_id == uGNSS.GLO:
                continue
            for f in range(min(nav.nf, obs_sd.L.shape[1])):
                if obs_sd.L[i, f] != 0 and obs_sd.P[i, f] != 0:
                    groups.setdefault((int(sys_id), int(f)), []).append(i)
        if os.environ.get('RBPF_DBG_SATS') and self.n_steps % 200 == 1:
            print('DBG_SATS nf', nav.nf, obs_sd.L.shape, [(sat2prn(int(s_))[0], sat2prn(int(s_))[1],
                                round(float(np.rad2deg(el[i_])), 1),
                                tuple(int(obs_sd.L[i_, f_] != 0)
                                      for f_ in range(obs_sd.L.shape[1])))
                               for i_, s_ in enumerate(sat)])
        rows = []
        for g in sorted(groups):
            mem = groups[g]
            if len(mem) < 2:
                continue
            ref_s = self.prev_ref.get(g)
            ri = next((i for i in mem if int(sat[i]) == ref_s), None)
            if ri is None:
                ri = mem[int(np.argmax([el[i] for i in mem]))]
            ref = int(sat[ri])
            lams = _get_wavelengths(nav, obs_sd, ref)
            f = g[1]
            if f >= len(lams) or not np.isfinite(lams[f]) or lams[f] <= 0:
                continue
            lam = float(lams[f])
            r_ref, e_ref = geodist(rs[iu[ri], :3], pos)
            rb_ref, _ = geodist(rsb[ir[ri], :3], rb)
            zmap = map_H.get(g) if map_H else None
            for ji in mem:
                if ji == ri:
                    continue
                j = int(sat[ji])
                r_j, e_j = geodist(rs[iu[ji], :3], pos)
                rb_j, _ = geodist(rsb[ir[ji], :3], rb)
                ddrange = (r_ref - rb_ref) - (r_j - rb_j)
                Jrow = -np.asarray(e_ref, dtype=float) + np.asarray(e_j, dtype=float)
                b0 = (float(obs_sd.L[ri, f] - obs_sd.L[ji, f]) * lam - ddrange) / lam
                ddpr = float(obs_sd.P[ri, f] - obs_sd.P[ji, f]) - ddrange
                held = np.nan
                if zmap is not None and ref in zmap and j in zmap:
                    held = float(zmap[ref] - zmap[j])
                pi = pair_idx.get((g, ref, j))
                mass = float(pair_mass.get(pi, np.nan)) if pi is not None else np.nan
                rows.append((g[0], f, ref, j, float(el[ri]), float(el[ji]), lam,
                             Jrow[0], Jrow[1], Jrow[2], b0, ddpr, held, mass,
                             1.0 if pi is not None else 0.0))
        return np.asarray(rows, dtype=float).reshape(-1, 15)

    def _moves_tracked(self, prob, ctx, ep, H_cons, cand_H, spawn_H,
                       lp_nb, lp_b):
        """WP43 fully-adapted move step with per-particle position beliefs.

        Same move set, priors and sampling as the WP18 loop in `step`,
        but each (particle, move) is scored against the particle's own
        predicted position belief (or the loose float prior when it has
        none), and the burst alternative uses that same prior. All the
        solves of the epoch go through one `_loglik_batch` call. Returns
        [(H, tag)] and sets each particle's new belief."""
        cfg = self.cfg
        odo = ctx.get('odo')
        reset = self._track_reset(ctx)
        if reset:
            for p in self.particles:
                p.state = None
            self._track_reset_tow = getattr(self, '_track_last_tow', None)
        if odo is None:
            odo = np.zeros(3)
        rt = getattr(self, '_track_reset_tow', None)
        probation = (cfg.track_probation_s > 0 and rt is not None
                     and getattr(self, '_track_last_tow', rt) - rt
                     < cfg.track_probation_s)

        nb = len(prob.pairs)
        nan_dd = np.full(nb, np.nan)
        evals_dd, evals_c, evals_P = [], [], []
        index = {}
        fresh_c = np.zeros(3)
        fresh_P = prob.Ppos

        def add(H, skey, pred):
            key = (_canon_basin_id(H) if H else (), skey)
            k = index.get(key)
            if k is not None:
                return k
            k = len(evals_dd)
            index[key] = k
            evals_dd.append(self._pair_dd(H, prob) if H else nan_dd)
            if pred is None:
                evals_c.append(fresh_c)
                evals_P.append(fresh_P)
            else:
                evals_c.append(pred[0] - prob.pos)
                evals_P.append(pred[1])
            return k

        plans = []
        for p in self.particles:
            pred = (None if probation
                    else self._track_predict(p.state, odo, ep))
            sk = id(p.state) if pred is not None else None
            fk, fp = (sk, pred)
            nk, npred = ((sk, pred) if cfg.track_inherit else (None, None))
            ek, epred = ((sk, pred) if cfg.track_empty else (None, None))
            moves = []   # (logprior, eval idx, burst idx, H, tag)

            def mv(prior, H, tag, skey, spred, moves=moves):
                moves.append((prior, add(H, skey, spred),
                              add({}, skey, spred), H, tag))
            if p.H:
                mv(cfg.p_survive, p.H, 'survive', fk, fp)
                H_ext, H_shr = self._move_pair(p.H, prob)
                if H_ext is not None:
                    mv(cfg.p_extend, H_ext, 'extend', fk, fp)
                if H_shr is not None:
                    mv(cfg.p_shrink, H_shr, 'shrink', fk, fp)
                mv(cfg.p_release, {}, 'release', ek, epred)
            else:
                mv(cfg.p_survive + cfg.p_extend + cfg.p_shrink
                   + cfg.p_release, p.H, 'stay', ek, epred)
            if H_cons is not None:
                mv(cfg.p_adopt_cons, H_cons, 'adopt_cons', nk, npred)
            for Hk in cand_H:
                mv(cfg.p_adopt / len(cand_H), Hk, 'adopt', nk, npred)
            for Hs in spawn_H:
                mv(cfg.p_spawn / len(spawn_H), Hs, 'spawn', nk, npred)
            plans.append(moves)

        ll, sol, P3 = self._loglik_batch(
            prob, np.asarray(evals_dd), np.asarray(evals_c),
            np.asarray(evals_P))
        self.track_evals += len(evals_dd)

        floor2 = np.eye(3) * cfg.track_floor ** 2
        made = {}
        new_states = []
        for p, moves in zip(self.particles, plans):
            pri = np.array([mv_[0] for mv_ in moves], dtype=float)
            pri /= pri.sum()
            a_nb = np.array([lp_nb + ll[mv_[1]] for mv_ in moves])
            a_b = np.array([lp_b + ll[mv_[2]] for mv_ in moves])
            lam_arr = np.log(pri) + np.logaddexp(a_nb, a_b)
            mx = float(np.max(lam_arr))
            wsum = np.exp(lam_arr - mx)
            tot = float(np.sum(wsum))
            p.logw += mx + math.log(tot)
            pick = int(self.rng.choice(len(moves), p=wsum / tot))
            _, e, eb, H, tag = moves[pick]
            if not H and not cfg.track_empty:
                p.state = None
            else:
                src = e if a_nb[pick] >= a_b[pick] else eb
                st = made.get(src)
                if st is None:
                    st = TrackState(prob.pos + sol[src], P3[src] + floor2,
                                    np.array(odo, dtype=float), ep)
                    made[src] = st
                p.state = st
            new_states.append((H, tag))
        return new_states

    def _shadow_track(self, prob, ctx, ep):
        """WP43 RBPF_TRACK=2: update every particle's position belief
        under the integers it now holds, without touching the weights
        (the WP18 move/weight path runs unchanged). Every configured
        variant is updated by the same batched call."""
        cfg = self.cfg
        odo = ctx.get('odo')
        reset = self._track_reset(ctx)
        if odo is None:
            odo = np.zeros(3)
        nV = len(cfg.track_variants)
        nb = len(prob.pairs)
        index, dds, cs, Ps, plan = {}, [], [], [], []
        for p in self.particles:
            if len(p.vstates) != nV - 1:
                p.vstates = [None] * (nV - 1)
            row = []
            for v, (q, _floor, empty) in enumerate(cfg.track_variants):
                st = p.state if v == 0 else p.vstates[v - 1]
                if reset or (not p.H and not empty):
                    st = None
                if not p.H and not empty:
                    row.append(None)
                    continue
                pred = self._track_predict(st, odo, ep, q)
                key = (v, _canon_basin_id(p.H),
                       id(st) if pred is not None else None)
                k = index.get(key)
                if k is None:
                    k = index[key] = len(dds)
                    dds.append(self._pair_dd(p.H, prob) if p.H
                               else np.full(nb, np.nan))
                    if pred is None:
                        cs.append(np.zeros(3))
                        Ps.append(prob.Ppos)
                    else:
                        cs.append(pred[0] - prob.pos)
                        Ps.append(pred[1])
                row.append(k)
            plan.append(row)
        if dds:
            _, sol, P3 = self._loglik_batch(prob, np.asarray(dds),
                                            np.asarray(cs), np.asarray(Ps))
            self.track_evals += len(dds)
        made = {}
        for p, row in zip(self.particles, plan):
            for v, k in enumerate(row):
                st = None
                if k is not None:
                    st = made.get(k)
                    if st is None:
                        fl = cfg.track_variants[v][1]
                        st = made[k] = TrackState(
                            prob.pos + sol[k], P3[k] + np.eye(3) * fl ** 2,
                            np.array(odo, dtype=float), ep)
                if v == 0:
                    p.state = st
                else:
                    p.vstates[v - 1] = st

    def _loo_consistency(self, prob, dd_map, used_pairs):
        """Absolute held-out DD-CP innovations in metres.

        For each fixed pair, solve position from all other fixed pairs.  A
        robust common offset is estimated only from training pairs in the
        same constellation/frequency group, then applied to the held row.
        This avoids certifying an ambiguity with the carrier row that created
        it while retaining the reference-satellite nuisance model.
        """
        scores = []
        used = [int(p) for p in used_pairs if np.isfinite(dd_map[int(p)])]
        cp_row = {int(prob.rows_pair[r]): r
                  for r in np.where(prob.rows_iscp)[0]}
        for held in used:
            if held not in cp_row:
                continue
            train = [p for p in used if p != held]
            if len(train) < self.cfg.min_fix_nb:
                continue
            H_train = {}
            for pi in train:
                g, ref, sat = prob.pairs[pi]
                zmap = H_train.setdefault(g, {})
                zmap.setdefault(ref, 0)
                zmap[sat] = int(zmap[ref] - int(dd_map[pi]))
            pos, _, _ = self._conditional_mean(H_train, prob)
            if pos is None:
                continue
            dpos = np.asarray(pos) - prob.pos
            group = prob.pairs[held][0]
            offsets = []
            for pi in train:
                if prob.pairs[pi][0] != group or pi not in cp_row:
                    continue
                r = cp_row[pi]
                raw = prob.rows_lam[r] * (prob.rows_bmeas[r] - dd_map[pi])
                offsets.append(float(raw - prob.rows_J[r] @ dpos))
            if not offsets:
                continue
            r = cp_row[held]
            raw = prob.rows_lam[r] * (prob.rows_bmeas[r] - dd_map[held])
            innovation = raw - prob.rows_J[r] @ dpos - np.median(offsets)
            if np.isfinite(innovation):
                scores.append(abs(float(innovation)))
        return np.asarray(scores, dtype=float)

    def _wl_fix(self, prob):
        """WP49 wide-lane diagnostic (RBPF_WL=1): DD wide-lane integers
        N_WL = N_f0 - N_f1 from this epoch's float DD ambiguities (the
        bands' references may differ: DD_f1(ra, j) = DD_f1(rb, j) -
        DD_f1(rb, ra)), LAMBDA ratio of the best two WL candidates, and the
        WL position from the epoch-local LS on the WL carrier rows
        lam_wl (b_f0 - b_f1 - N_WL) = J dx plus the DD-PR rows and the
        relaxed float prior. Nothing here feeds back into the PF."""
        res = {'wl_n': 0, 'wl_ratio': float('nan'), 'wl_rms': float('nan'),
               'wl_pos': None}
        if not prob.ok or prob.y is None:
            return res
        idx = {}
        refs = {}
        mem = {}
        for pi, (g, ref, j) in enumerate(prob.pairs):
            idx[(g[0], g[1], ref, j)] = pi
            refs[g] = ref
            mem.setdefault(g, set()).add(j)
        cp = {}
        for r in range(len(prob.rows_v0)):
            if prob.rows_iscp[r]:
                cp[int(prob.rows_pair[r])] = r
        T, rows = [], []
        nb = len(prob.pairs)
        for sys_id in sorted({g[0] for g in refs}):
            g0, g1 = (sys_id, 0), (sys_id, 1)
            if g0 not in refs or g1 not in refs:
                continue
            ra, rb = refs[g0], refs[g1]
            in1 = mem[g1] | {rb}
            if ra not in in1:
                continue
            for j in sorted(mem[g0]):
                if j not in in1:
                    continue
                p0 = idx[(sys_id, 0, ra, j)]
                p1j = idx.get((sys_id, 1, rb, j)) if j != rb else None
                p1a = idx.get((sys_id, 1, rb, ra)) if ra != rb else None
                t = np.zeros(nb)
                t[p0] += 1.0
                if p1j is not None:
                    t[p1j] -= 1.0
                if p1a is not None:
                    t[p1a] += 1.0
                T.append(t)
                # carrier rows of both bands (cycles), f1 rebuilt on ra
                r0 = cp.get(p0)
                r1j = cp.get(p1j) if p1j is not None else None
                r1a = cp.get(p1a) if p1a is not None else None
                ok = (r0 is not None and (p1j is None or r1j is not None)
                      and (p1a is None or r1a is not None))
                if ok:
                    b1 = 0.0
                    lam1 = None
                    sig1 = 0.0
                    if r1j is not None:
                        b1 += prob.rows_bmeas[r1j]
                        lam1 = prob.rows_lam[r1j]
                        sig1 = prob.rows_sig[r1j]
                    if r1a is not None:
                        b1 -= prob.rows_bmeas[r1a]
                        lam1 = prob.rows_lam[r1a]
                        sig1 = max(sig1, prob.rows_sig[r1a])
                    lam0 = prob.rows_lam[r0]
                    if lam1 is None or not (lam0 > 0 and lam1 > 0) or lam0 == lam1:
                        ok = False
                rows.append((prob.rows_bmeas[r0] - b1, lam0, lam1,
                             prob.rows_J[r0], prob.rows_sig[r0], sig1)
                            if ok else None)
        n = len(T)
        res['wl_n'] = n
        if n < 2:
            return res
        T = np.asarray(T)
        z = T @ prob.y
        Qz = T @ prob.Qb @ T.T
        Qz = 0.5 * (Qz + Qz.T)
        try:
            afix, sqnorm, _, _ = _wp13m_finite_guarded_mlambda(
                z, Qz, ncands=2, parmode=1)
        except (LambdaError, np.linalg.LinAlgError, ValueError):
            return res
        cands, sq = self._topk_postprocess(afix, sqnorm)
        if cands is None:
            return res
        nwl = cands[:, 0]
        res['wl_ratio'] = (float(sq[1] / max(sq[0], 1e-12))
                           if len(sq) > 1 else float('inf'))
        # WL carrier rows (metres) + DD-PR rows + relaxed float prior
        J, v, sg = [], [], []
        for k, rw in enumerate(rows):
            if rw is None:
                continue
            dbw, lam0, lam1, Jr, s0, s1 = rw
            lw = 1.0 / (1.0 / lam0 - 1.0 / lam1)
            J.append(Jr)
            v.append(lw * (dbw - nwl[k]))
            sg.append(abs(lw) * math.hypot(s0 / lam0, s1 / lam1))
        nwr = len(v)
        if nwr < 4:
            return res
        for r in range(len(prob.rows_v0)):
            if not prob.rows_iscp[r]:
                J.append(prob.rows_J[r])
                v.append(prob.rows_v0[r])
                sg.append(prob.rows_sig[r])
        J = np.asarray(J); v = np.asarray(v); sg = np.asarray(sg)
        A = J / sg[:, None]
        b = v / sg
        try:
            Pinv = np.linalg.inv(prob.Ppos)
            sol = np.linalg.solve(Pinv + A.T @ A, A.T @ b)
            c = self.cfg.huber_c
            if c > 0:
                for _ in range(2):
                    rr = b - A @ sol
                    sw = np.sqrt(np.minimum(1.0, c / np.maximum(np.abs(rr), 1e-12)))
                    Aw = A * sw[:, None]
                    sol = np.linalg.solve(Pinv + Aw.T @ Aw, Aw.T @ (b * sw))
        except np.linalg.LinAlgError:
            return res
        if not np.all(np.isfinite(sol)):
            return res
        rw_m = v[:nwr] - J[:nwr] @ sol
        res['wl_rms'] = float(np.sqrt(np.mean(rw_m ** 2)))
        res['wl_pos'] = prob.pos + sol
        res['wl_nrows'] = nwr
        return res

    def _conditional_mean(self, H, prob):
        """MAP-basin conditional position.

        Default: the epoch-local robust LS at the basin (held DD-CP rows
        at their tight sigma + DD-PR rows + relaxed position prior +
        group nuisances) -- the same model the weights use. The
        window-posterior Gaussian conditioning (x - Qab Qb^-1 (y-z)) is
        kept behind RBPF_COND_POSTERIOR=1: it inherits the graph
        covariance's overconfidence (measured: corrections under-scaled,
        conditional mean stuck at the ~35cm float error even on correct
        basins)."""
        if not H:
            return None, 0, float('nan')
        bid = _canon_basin_id(H)
        hit = self._conditional_cache.get(bid)
        if hit is not None:
            return hit[:3]
        dd = self._pair_dd(H, prob)
        idxH = np.where(np.isfinite(dd))[0]
        if len(idxH) == 0:
            return None, 0, float('nan')
        if _env_i('RBPF_COND_POSTERIOR', 0):
            try:
                QHH = prob.Qb[np.ix_(idxH, idxH)]
                r = np.linalg.solve(QHH, prob.y[idxH] - dd[idxH])
                xz = prob.pos - prob.Qab3[:, idxH] @ r
            except np.linalg.LinAlgError:
                return None, 0, float('nan')
            if not np.all(np.isfinite(xz)):
                return None, 0, float('nan')
            return xz, int(len(idxH)), float('nan')
        got = self._ls_offset(H, prob, dd)
        if got is None:
            return None, 0, float('nan')
        sol, min_info, P3 = got
        out = (prob.pos + sol[0:3], int(len(idxH)), min_info, P3)
        self._conditional_cache[bid] = out
        return out[:3]

    def _position_rows(self, prob, dd, sb):
        """Prepare position rows, preserving scalar fallback dtype promotion."""
        names = ('rows_v0', 'rows_sig', 'rows_J', 'rows_lam', 'rows_bmeas')
        if (isinstance(dd, np.ndarray) and dd.ndim == 1 and dd.dtype == np.float64
                and all(isinstance(getattr(prob, n), np.ndarray)
                        and getattr(prob, n).dtype == np.float64 for n in names)
                and isinstance(prob.rows_iscp, np.ndarray)
                and prob.rows_iscp.dtype == np.bool_
                and isinstance(prob.rows_pair, np.ndarray)
                and prob.rows_pair.dtype.kind in 'iu'):
            cp = np.flatnonzero(prob.rows_iscp)
            held = cp[np.isfinite(dd[prob.rows_pair[cp]])]
            keep = ~prob.rows_iscp
            keep[held] = True
            rows = np.flatnonzero(keep)
            if not len(rows):
                return None
            local = np.searchsorted(rows, held)
            v, sig = prob.rows_v0[rows], prob.rows_sig[rows]
            v[local] = prob.rows_lam[held] * (
                prob.rows_bmeas[held] - dd[prob.rows_pair[held]])
            sig[local] = sig[local] / sb
            grp = np.full(len(rows), -1, dtype=int)
            groups = {}
            for ri, r in zip(local, held):
                g = prob.pairs[prob.rows_pair[r]][0]
                grp[ri] = groups.setdefault(g, len(groups))
            return v, sig, prob.rows_J[rows], grp, len(groups)

        rows_v, rows_sig, rows_J, grp = [], [], [], []
        grp_index = {}
        for r in range(len(prob.rows_v0)):
            if prob.rows_iscp[r]:
                p = prob.rows_pair[r]
                z = dd[p]
                if not np.isfinite(z):
                    continue
                rows_v.append(prob.rows_lam[r] * (prob.rows_bmeas[r] - z))
                rows_sig.append(prob.rows_sig[r] / sb)
                g = prob.pairs[p][0]
                if g not in grp_index:
                    grp_index[g] = len(grp_index)
                grp.append(grp_index[g])
            else:
                rows_v.append(prob.rows_v0[r])
                rows_sig.append(prob.rows_sig[r])
                grp.append(-1)
            rows_J.append(prob.rows_J[r])
        if not rows_v:
            return None
        return (np.asarray(rows_v), np.asarray(rows_sig), np.asarray(rows_J),
                np.asarray(grp, dtype=int), len(grp_index))

    def _ls_offset(self, H, prob, dd):
        """Robust epoch-local LS offset for hypothesis H (same row model
        as _loglik; returns ((3+G,) solution, min_info) or None).

        WP18 (failure mode b): min_info = smallest eigenvalue [1/m^2] of
        the DATA-ONLY position information at the final Huber weights --
        the Schur complement of the group-nuisance block of Aw'Aw (group
        prior kept on the nuisances, POSITION prior removed). On the
        7.08m blow-up epoch the position was prior-dominated along one
        direction (shared degenerate geometry, which is also why the
        DDPR cross-check missed it), so this is the discriminant the
        output guard thresholds on."""
        cfg = self.cfg
        sb = math.sqrt(cfg.beta)
        prepared = self._position_rows(prob, dd, sb)
        if prepared is None:
            return None
        v, sig, J, grp, G = prepared
        centre = np.zeros(3)
        state = self._cluster_state(H)
        Pprior = prob.Ppos
        if state is not None:
            centre = np.asarray(state['pos'], dtype=float) - prob.pos
            Pprior = np.asarray(state['P'], dtype=float)
            if not np.all(np.isfinite(centre)):
                centre = np.zeros(3)
                Pprior = prob.Ppos
        v = v - J @ centre
        A = J / sig[:, None]
        if G:
            C = np.zeros((len(v), G))
            for ri, gi in enumerate(grp):
                if gi >= 0:
                    C[ri, gi] = 1.0 / sig[ri]
            A = np.hstack([A, C])
        b = v / sig
        Af = A
        try:
            Pinv = np.zeros((3 + G, 3 + G))
            Pinv[0:3, 0:3] = np.linalg.inv(Pprior)
            if G:
                Pinv[3:, 3:] = np.eye(G) / (cfg.sig_group ** 2)
            sol = np.linalg.solve(Pinv + A.T @ A, A.T @ b)
            c = cfg.huber_c
            if c > 0:
                for _ in range(2):
                    r = b - A @ sol
                    wgt = np.minimum(1.0, c / np.maximum(np.abs(r), 1e-12))
                    sw = np.sqrt(wgt)
                    Aw = A * sw[:, None]
                    sol = np.linalg.solve(
                        Pinv + Aw.T @ Aw, Aw.T @ (b * sw))
                Af = Aw
        except np.linalg.LinAlgError:
            return None
        if not np.all(np.isfinite(sol)):
            return None
        # WP18: data-only position information (see docstring)
        min_info = float('nan')
        try:
            M = Af.T @ Af
            I33 = M[0:3, 0:3]
            if G:
                MGG = M[3:, 3:] + np.eye(G) / (cfg.sig_group ** 2)
                I33 = I33 - M[0:3, 3:] @ np.linalg.solve(MGG, M[3:, 0:3])
            min_info = float(np.min(np.linalg.eigvalsh(0.5 * (I33 + I33.T))))
        except np.linalg.LinAlgError:
            pass
        try:
            Pfull = np.linalg.inv(Pinv + Af.T @ Af)
            P3 = 0.5 * (Pfull[0:3, 0:3] + Pfull[0:3, 0:3].T)
            P3 += np.eye(3) * cfg.cluster_sig_floor ** 2
        except np.linalg.LinAlgError:
            P3 = np.asarray(Pprior, dtype=float).copy()
        sol[0:3] += centre
        return sol, min_info, P3


WL_F = 9   # band index of the wide-lane groups (sys, WL_F)


class RbpfFgoWL(RbpfFgo):
    """WP49 wide-lane RB-PF (RBPF_WLPF=1): the same particle machinery on
    a wide-lane epoch problem built from the main PF's problem -- WL
    pairs ((sys, WL_F), ref, j) on band 0's reference, WL floats T y with
    covariance T Q T', and WL carrier rows lam_wl (b0 - b1 - N_WL) = J dx
    next to the DD-PR rows. Diagnostic only: no feedback to the graph,
    no witness; a slip on either band drops the satellite's WL integer."""

    def __init__(self, cfg, main):
        cfg.fb_enable = 0
        cfg.dump_rows = 0
        cfg.afv = 0
        cfg.seed = cfg.seed + 7919
        super().__init__(cfg)
        self.main = main

    def _attach_witness(self, prob, witness_prob):
        return

    def _extract(self, rtk, ctx):
        prob = EpochProblem()
        src = getattr(self.main, '_last_prob', None)
        if src is None or not src.ok:
            prob.reason = 'no_nl_problem'
            return prob
        idx, refs, mem = {}, {}, {}
        for pi, (g, ref, j) in enumerate(src.pairs):
            idx[(g[0], g[1], ref, j)] = pi
            refs[g] = ref
            mem.setdefault(g, set()).add(j)
        cp = {int(src.rows_pair[r]): r for r in range(len(src.rows_v0))
              if src.rows_iscp[r]}
        nb = len(src.pairs)
        T, pairs, rows = [], [], []
        for sys_id in sorted({g[0] for g in refs}):
            g0, g1 = (sys_id, 0), (sys_id, 1)
            if g0 not in refs or g1 not in refs:
                continue
            ra, rb = refs[g0], refs[g1]
            in1 = mem[g1] | {rb}
            if ra not in in1:
                continue
            for j in sorted(mem[g0]):
                if j not in in1:
                    continue
                p0 = idx[(sys_id, 0, ra, j)]
                p1j = idx.get((sys_id, 1, rb, j)) if j != rb else None
                p1a = idx.get((sys_id, 1, rb, ra)) if ra != rb else None
                if (j != rb and p1j is None) or (ra != rb and p1a is None):
                    continue
                t = np.zeros(nb)
                t[p0] += 1.0
                if p1j is not None:
                    t[p1j] -= 1.0
                if p1a is not None:
                    t[p1a] += 1.0
                k = len(pairs)
                T.append(t)
                pairs.append(((int(sys_id), WL_F), int(ra), int(j)))
                r0 = cp.get(p0)
                r1j = cp.get(p1j) if p1j is not None else None
                r1a = cp.get(p1a) if p1a is not None else None
                if (r0 is None or (p1j is not None and r1j is None)
                        or (p1a is not None and r1a is None)):
                    continue
                b1, lam1, s1 = 0.0, None, 0.0
                if r1j is not None:
                    b1 += src.rows_bmeas[r1j]
                    lam1, s1 = src.rows_lam[r1j], src.rows_sig[r1j]
                if r1a is not None:
                    b1 -= src.rows_bmeas[r1a]
                    lam1, s1 = src.rows_lam[r1a], max(s1, src.rows_sig[r1a])
                lam0 = src.rows_lam[r0]
                if lam1 is None or not (lam0 > 0 and lam1 > 0) or lam0 == lam1:
                    continue
                lw = 1.0 / (1.0 / lam0 - 1.0 / lam1)
                rows.append((src.rows_J[r0], abs(lw) * math.hypot(
                    src.rows_sig[r0] / lam0, s1 / lam1), k, lw,
                    src.rows_bmeas[r0] - b1))
        if len(pairs) < 2 or not rows:
            prob.reason = 'no_wl_pairs'
            return prob
        T = np.asarray(T)
        Qb = T @ src.Qb @ T.T
        prob.set_pairs(pairs)
        prob.y = T @ src.y
        prob.Qb = 0.5 * (Qb + Qb.T) + np.eye(len(pairs)) * self.cfg.qb_jitter
        prob.Qab3 = src.Qab3 @ T.T
        prob.pos = src.pos.copy()
        prob.Ppos = src.Ppos.copy()
        J, v0, sig, iscp, rp, lam, bm = [], [], [], [], [], [], []
        for r in range(len(src.rows_v0)):
            if not src.rows_iscp[r]:
                J.append(src.rows_J[r]); v0.append(src.rows_v0[r])
                sig.append(src.rows_sig[r]); iscp.append(False)
                rp.append(-1); lam.append(0.0); bm.append(0.0)
        for (Jr, sg, k, lw, db) in rows:
            J.append(Jr); v0.append(0.0); sig.append(sg); iscp.append(True)
            rp.append(k); lam.append(lw); bm.append(db)
        prob.rows_J = np.asarray(J, dtype=float)
        prob.rows_v0 = np.asarray(v0, dtype=float)
        prob.rows_sig = np.asarray(sig, dtype=float)
        prob.rows_iscp = np.asarray(iscp, dtype=bool)
        prob.rows_pair = np.asarray(rp, dtype=int)
        prob.rows_lam = np.asarray(lam, dtype=float)
        prob.rows_bmeas = np.asarray(bm, dtype=float)
        prob.cands, prob.cand_sqnorm = self._lambda_topk(
            prob.y, prob.Qb, self.cfg.topk)
        prob.ok = True
        return prob


# --------------------------------------------------------------------------
# shared-filter subclass with the PF bolted on
# --------------------------------------------------------------------------

class RbpfGtsamRtkTc(GtsamRtkTc):
    """GtsamRtkTc + RB-FGO-PF layer. The base pipeline runs untouched
    (pure-float config comes from the runner's env); this subclass only
    stashes per-epoch context and steps the PF after `_do_ar`."""

    def __init__(self, nav, imu_data, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, imu_data, pos0, logfile)
        self.rbpf = RbpfFgo(RbpfConfig())
        self.rbpf_wl = (RbpfFgoWL(RbpfConfig(), self.rbpf)
                        if _env_i('RBPF_WLPF', 0) else None)
        self._rbpf_wl_last = None
        self.rbpf_run_base_ar = _env_i('RBPF_BASE_AR', 1)
        self._rbpf_ctx = None
        self._rbpf_released = set()
        self._rbpf_release_reasons = {}
        self._rbpf_graph_reseeded = False
        self._rbpf_last = None
        self._rbpf_witness_prob = None
        self._rbpf_witness_pos = None
        self._rbpf_witness_P = None

    # stash the epoch context the PF needs (geometry + SD observations)
    def _build_dd_factors_arm(self, graph, new_values, obs, obsb, obs_sd,
                              rs, rsb, sat, el, iu, ir, pos_pred_ecef, ep):
        self._rbpf_ctx = {
            'obs': obs, 'obsb': obsb, 'obs_sd': obs_sd, 'rs': rs,
            'rsb': rsb, 'sat': sat, 'el': el, 'iu': iu, 'ir': ir, 'ep': ep}
        if self.rbpf.cfg.track:
            # WP43: IMU-predicted antenna motion since the last committed
            # epoch (the prediction this epoch's graph update starts from)
            dm = None
            try:
                if (self.current_estimate is not None
                        and self.last_valid_epoch is not None):
                    a0 = self._antenna_ecef(self.current_estimate.atPose3(
                        self.PX(self.last_valid_epoch)))
                    a1 = self._antenna_ecef(new_values.atPose3(self.PX(ep)))
                    dm = a1 - a0
            except RuntimeError:
                dm = None
            self._rbpf_ctx['dmotion'] = dm
        return super()._build_dd_factors_arm(
            graph, new_values, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir,
            pos_pred_ecef, ep)

    def _antenna_ecef(self, pose):
        pe = self.ecef_T_nav.compose(pose)
        return (np.array(pe.translation())
                + pe.rotation().matrix() @ self.lever_arm)

    def predict_antenna_at(self, t):
        """WP43 output gap fill: IMU dead-reckoned antenna ECEF at obs
        time `t` from the last committed phase-2 state, without touching
        the filter. Returns (predicted, antenna at the last committed
        epoch) or None. `imu_idx` always points just past the last
        committed epoch (abandoned epochs roll it back)."""
        if (getattr(self, 'phase', 1) != 2 or self.current_estimate is None
                or self.last_valid_epoch is None):
            return None
        try:
            ep = self.last_valid_epoch
            pose = self.current_estimate.atPose3(self.PX(ep))
            vel = np.array(self.current_estimate.atVector(self.PV(ep)))
            bias = self.current_estimate.atConstantBias(self.PB(ep))
            _, tow = time2gpst(t)
            pim, n_int, _, _ = build_pim(self.imu_params, bias,
                                         self.imu_data, self.imu_idx,
                                         target_tow=tow)
            if n_int < 1:
                return None
            import gtsam
            pred = pim.predict(gtsam.NavState(pose, vel), bias)
            return self._antenna_ecef(pred.pose()), self._antenna_ecef(pose)
        except (RuntimeError, IndexError, ValueError):
            return None

    # every ambiguity release (slip/MW/CMC/outage/FDE/ladder) invalidates
    # that (sat,f) in every particle's basin
    def _release_ambiguity(self, sat, freq, reason=''):
        self._rbpf_released.add((int(sat), int(freq)))
        self._rbpf_release_reasons.setdefault(
            (int(sat), int(freq)), []).append(str(reason))
        return super()._release_ambiguity(sat, freq, reason=reason)

    # a full graph re-seed (warm reset / transition) clears all basins
    def _seed_phase2_graph(self, pose0, vel_enu, bias0, obs_time, imu_idx0):
        self._rbpf_graph_reseeded = True
        return super()._seed_phase2_graph(
            pose0, vel_enu, bias0, obs_time, imu_idx0)

    def _do_ar(self, obs, rs, vs, dts, sat, el, iu):
        if self.rbpf_run_base_ar:
            super()._do_ar(obs, rs, vs, dts, sat, el, iu)
        elif self.per_sat_gate_enable and getattr(self, 'phase', 1) == 2:
            # keep tc/'s per-sat residual gate on the PF support even when
            # the base AR call is skipped for throughput
            per_sat = ((self._main_ddpr_per_sat_pr
                        if self.per_sat_gate_pr_only
                        else self._main_ddpr_per_sat) or {})
            for _s, _rmax in per_sat.items():
                _s = int(_s)
                if _rmax > self.per_sat_res_thresh and 1 <= _s <= self.nav.vsat.shape[0]:
                    self.nav.vsat[_s - 1, :] = 0
        if getattr(self, 'phase', 1) != 2 or self._rbpf_ctx is None:
            return
        if self.rbpf.cfg.track:
            # WP43 odometer: accumulate committed epochs' IMU increments;
            # a PIM break (pose untied from the IMU chain) or a missing
            # increment invalidates every particle's position belief.
            ctx = self._rbpf_ctx
            dm = ctx.get('dmotion')
            nbreak = getattr(self, 'pim_break_count', 0)
            reset = (dm is None or not np.all(np.isfinite(dm))
                     or nbreak != getattr(self, '_rbpf_pim_breaks', 0))
            self._rbpf_pim_breaks = nbreak
            if not hasattr(self, '_rbpf_odo'):
                self._rbpf_odo = np.zeros(3)
            if not reset:
                self._rbpf_odo = self._rbpf_odo + dm
            ctx['odo'] = self._rbpf_odo.copy()
            ctx['track_reset'] = reset
        if self.rbpf_wl is not None:
            rel = set(getattr(self, '_rbpf_released', None) or set())
            why = dict(getattr(self, '_rbpf_release_reasons', None) or {})
            reseed = bool(getattr(self, '_rbpf_graph_reseeded', False))
        self._rbpf_last = self.rbpf.step(self, self._rbpf_ctx)
        if self.rbpf_wl is not None:
            # the main step consumed the release / re-seed flags: hand the
            # WL instance the same events on its band (a slip on either
            # band breaks the wide lane)
            self._rbpf_released = {(s_, WL_F) for (s_, _f) in rel}
            wl_why = {}
            for (s_, _f), r_ in why.items():
                wl_why.setdefault((s_, WL_F), []).extend(r_)
            self._rbpf_release_reasons = wl_why
            self._rbpf_graph_reseeded = reseed
            self._rbpf_wl_last = self.rbpf_wl.step(self, self._rbpf_ctx)
            self._rbpf_released = set()
            self._rbpf_release_reasons = {}
            self._rbpf_graph_reseeded = False


class WitnessGtsamRtkTc(GtsamRtkTc):
    """Immutable evidence FGO: same inputs, no AR/hold mutation."""

    def __init__(self, nav, imu_data, pos0=np.zeros(3), logfile=None):
        super().__init__(nav, imu_data, pos0, logfile)
        self._rbpf_ctx = None

    def _build_dd_factors_arm(self, graph, new_values, obs, obsb, obs_sd,
                              rs, rsb, sat, el, iu, ir, pos_pred_ecef, ep):
        self._rbpf_ctx = {
            'obs': obs, 'obsb': obsb, 'obs_sd': obs_sd, 'rs': rs,
            'rsb': rsb, 'sat': sat, 'el': el, 'iu': iu, 'ir': ir, 'ep': ep}
        old_skip = self._recov_cp_skip_now
        if _env_i('RBPF_WITNESS_PR_ONLY', 0):
            self._recov_cp_skip_now = True
        try:
            return super()._build_dd_factors_arm(
                graph, new_values, obs, obsb, obs_sd, rs, rsb, sat, el, iu, ir,
                pos_pred_ecef, ep)
        finally:
            self._recov_cp_skip_now = old_skip

    def _do_ar(self, obs, rs, vs, dts, sat, el, iu):
        return None
