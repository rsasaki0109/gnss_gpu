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
- Use `experiments/exp_particle_visualization.py --renderer enu` for quick
  particle-cloud trajectory videos without depending on map tiles.
- The initial scorer can derive distance weights from adjacent reference ECEF
  positions.  If `reference.csv` exposes official speed-derived distance
  weights, pass those weights into `score_ppc2024`.

Example quick video:

```bash
PYTHONPATH=python:experiments python3 experiments/exp_particle_visualization.py \
  --data-root /media/sasaki/aiueo/ai_coding_ws/datasets/PPC-Dataset-data/nagoya \
  --run run2 --systems G,E,J --renderer enu --max-epochs 300 \
  --n-particles 5000 --dump-every 5 --max-dump-particles 1200 \
  --output experiments/results/paper_assets/particle_viz_ppc_nagoya_run2_300ep.mp4
```
