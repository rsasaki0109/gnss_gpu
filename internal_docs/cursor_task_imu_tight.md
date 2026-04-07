# Task: PF IMU Tight Coupling

## ゴール
IMU を predict だけでなく weight/position_update にも使い、GNSS 劣化区間を慣性航法で乗り切る。

## 背景
- 現在の IMU 統合は loose coupling (predict に velocity を使うだけ)
- Shinjuku P50: 3.38m → 3.18m (IMU predict のみ)
- SPP の 32.7% が 10m 超エラー — この区間で IMU dead reckoning が効くはず
- FGO tight coupling は 1m 切ってるが、PF の枠内でも同様の効果が期待できる

## 実装方針

### 1. IMU Dead Reckoning Position の算出

エポックごとに:
```python
# prev_pf_estimate = pf.estimate()[:3] (前エポックの PF 推定位置)
# imu_velocity = IMU from ComplementaryHeadingFilter (ENU→ECEF変換済み)
# dt = epoch interval
imu_predicted_pos = prev_pf_estimate + imu_velocity * dt
```

### 2. Adaptive IMU Position Update

GNSS の品質に応じて IMU position_update の sigma を変える:

```python
# GNSS quality indicators:
n_sats = len(measurements)
spp_residual_rms = compute_spp_residual_rms(measurements, spp_pos)

# Adaptive sigma: GNSS が良い時は IMU を緩く、悪い時は締める
if n_sats < 6 or spp_residual_rms > 20.0:
    imu_pu_sigma = 3.0   # GNSS 劣化 → IMU を強く信頼
elif n_sats < 8 or spp_residual_rms > 10.0:
    imu_pu_sigma = 8.0   # GNSS やや劣化
else:
    imu_pu_sigma = 30.0  # GNSS 良好 → IMU は弱く

pf.position_update(imu_predicted_pos, sigma_pos=imu_pu_sigma)
```

### 3. SPP Position Update との共存

1 エポックで 2 回の position_update を行う:
1. SPP position_update (既存、sigma=1.5)
2. IMU position_update (新規、adaptive sigma)

両方の constraint が独立に作用する。GNSS 劣化時は IMU が主導、GNSS 良好時は SPP が主導。

### 4. 実装場所

`experiments/exp_pf_smoother_eval.py` の `run_pf_with_optional_smoother()` を修正:

- 新パラメータ: `imu_tight_coupling: bool = False`
- CLI: `--imu-tight-coupling` フラグ
- `--predict-guide imu` と組み合わせて使う

```python
# After SPP position_update:
if imu_tight_coupling and imu_velocity is not None and prev_pf_estimate is not None:
    imu_predicted_pos = prev_pf_estimate + imu_velocity * dt
    
    # Compute GNSS quality
    ranges = np.linalg.norm(sat_ecef - spp_pos[:3], axis=1)
    cb_est = np.median(pr - ranges)
    residuals = np.abs(pr - ranges - cb_est)
    spp_res_rms = float(np.sqrt(np.mean(residuals**2)))
    n_sats = len(measurements)
    
    # Adaptive sigma
    if n_sats < 6 or spp_res_rms > 20.0:
        imu_sigma = 3.0
    elif n_sats < 8 or spp_res_rms > 10.0:
        imu_sigma = 8.0
    else:
        imu_sigma = 30.0
    
    if np.all(np.isfinite(imu_predicted_pos)):
        pf.position_update(imu_predicted_pos, sigma_pos=imu_sigma)
```

### 5. 実験コマンド

```bash
cd /workspace/ai_coding_ws/gnss_gpu/experiments
export PYTHONPATH="../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python:../python:."

# IMU tight coupling, Odaiba
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --position-update-sigma 1.5 --predict-guide imu --imu-tight-coupling

# IMU tight coupling, Shinjuku
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Shinjuku \
  --n-particles 100000 --position-update-sigma 1.5 --predict-guide imu --imu-tight-coupling

# IMU tight + residual downweight + PR accel (全部盛り)
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --position-update-sigma 1.5 --predict-guide imu \
  --imu-tight-coupling --residual-downweight --pr-accel-downweight

python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Shinjuku \
  --n-particles 100000 --position-update-sigma 1.5 --predict-guide imu \
  --imu-tight-coupling --residual-downweight --pr-accel-downweight
```

### 6. 判断基準
- Odaiba P50 < 1.5m なら有望
- Shinjuku P50 < 3.0m なら有望
- 特に SPP > 10m のエポックでの改善度を見る

### 注意
- `prev_pf_estimate` は毎エポック `pf.estimate()[:3]` を保存しておく
- IMU velocity の ENU→ECEF 変換は既に `exp_pf_smoother_eval.py` の IMU predict コードにある (エージェントが実装済み)
- `imu_predicted_pos` が NaN にならないよう guard を入れる
