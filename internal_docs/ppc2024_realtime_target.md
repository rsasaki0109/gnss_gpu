# PPC2024 Realtime Target

Goal: optimize PPC2024 before returning to Kaggle/GSDC.  The benchmark target is
to beat TURING's PPC2024 winning private score of 85.6%.

Constraints:

- Processing must be realtime/causal.  Do not use future observations.
- Backward smoothers and post-run smoothing are out of scope for this target.
- Fixed-lag methods must declare their latency before being compared with the
  realtime target.
- Primary score is the PPC2024 traveled-distance ratio with 3D error at or below
  0.5 m, averaged across the six official runs.

Implementation notes:

- Use `gnss_gpu.ppc_score.score_ppc2024` for local scoring.
- `experiments/exp_ppc_wls_sweep.py` ranks configurations by `ppc_score_pct`,
  not RMS/P95.
- Current Tokyo/run1 realtime fusion smoke segment:
  `start=1300,max_epochs=200` reaches 56.01% with TDCP + DD-PR/WL anchors,
  causal height hold, and stale-velocity height release.
- Use `experiments/exp_particle_visualization.py --renderer enu` for quick
  particle-cloud trajectory videos without depending on map tiles.
- The initial scorer can derive distance weights from adjacent reference ECEF
  positions.  If `reference.csv` exposes official speed-derived distance
  weights, pass those weights into `score_ppc2024`.
- For the next PF phase, consider Reservoir Stein Particle Filter as a
  bounded-memory, diversity-preserving realtime update pattern before reviving
  the heavier PF experiments.
- `gnss_gpu.reservoir_stein` is the first local RSPF-style building block:
  keep a bounded weighted reservoir, pin elite particles, then run a small
  SVGD-style attraction/repulsion transport step before any CUDA integration.
- ICRA 2025 lead from `DoongLi/ICRA2025-Paper-List`: "Range-Based 6-DoF
  Monte Carlo SLAM with Gradient-Guided Particle Filter on GPU" uses
  likelihood-gradient particle updates, compact keyframe state, and dead
  particle pruning.  For GNSS/PPC, map this to DD/WL likelihood gradients,
  bounded anchor/keyframe history, and pruning only fully collapsed particles.
- `experiments/exp_ppc_rsp_diagnostic.py` applies DD likelihood gradients to a
  local reservoir Stein correction.  On Tokyo/run1 `start=1300,max_epochs=200`,
  correcting only epochs 153-174 with fused-height/radius projection improved
  the diagnostic score from 56.01% to 60.46%; use this as the next candidate
  for integration into the realtime fusion path.
- The realtime fusion path now enables gated DD-gradient reservoir Stein
  horizontal corrections by default.  On the same Tokyo/run1 smoke segment it
  reaches 66.21% with 135/200 PPC 3D pass epochs after lowering stale-velocity
  height-release gating to 0.4 m DD-PR disagreement.
- `experiments/exp_ppc_realtime_fusion_sweep.py` runs the realtime fusion over
  fixed validation segments.  On the positive6 200-epoch smoke, the current
  realtime path is still weak: aggregate PPC is 2.75%, mean per-segment PPC is
  2.39%, and epoch pass is 90/1200.  The 0.4 m height-release gate improves
  epoch pass versus 1.0 m (90 vs. 65) but not distance-weighted PPC.  A tested
  wide-lane-aware height-release guard worsened aggregate PPC and should not be
  used as a default.

Example quick video:

```bash
PYTHONPATH=python:experiments python3 experiments/exp_particle_visualization.py \
  --data-root /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/nagoya \
  --run run2 --systems G,E,J --renderer enu --max-epochs 300 \
  --n-particles 5000 --dump-every 5 --max-dump-particles 1200 \
  --output experiments/results/paper_assets/particle_viz_ppc_nagoya_run2_300ep.mp4
```
