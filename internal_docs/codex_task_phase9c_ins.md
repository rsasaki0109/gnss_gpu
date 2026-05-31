# Codex Task: PPC Phase 9c — 本格 INS-GNSS Tight Coupling

ステータス: **未着手 (Phase 9b 完了 = commit `f510f23` 時点)**
依頼者意図: 「本格 INS (online bias 推定 + 正しい gravity alignment) wo yarou」
作業場所: `feature/ppc-realtime-turing-target` ブランチ (現在の HEAD)
評価基準: 6-run honest aggregate で **50.72% 天井を超える** (現在 Phase 9b は 50.52%)

---

## 0. 必読 — Phase 9b までの結論と Phase 9c の意味

PPC2024 の post-process / in-loop 拡張は 4 lever 全敗 (`memory/project_ppc_postprocess_ceiling.md` 参照):

| Phase | 内容 | Δ vs hybrid 50.72% |
|---|---|---|
| 4 (5 反復) | FGO + LAMBDA partial fix (post-process) | -3.65pp → -0.004pp に収束 |
| 8 | TDCP-anchored Kalman smoother (post-process) | -0.58pp |
| 9a | ZUPT (post-process IMU stop detection) | -0.004pp |
| **9b** | **tight-coupled IMU pre-integration in PF (in-loop)** | **-0.20pp** |

Phase 9b が負けた **真の理由** (これが Phase 9c で解決すべき問題):

1. **姿勢が yaw のみ・pitch/roll=0 固定** → vehicle が傾いていると重力が水平に漏れて加速度誤差。
2. **Yaw が GNSS 速度 (course-over-ground) 由来** → hybrid 速度自体が m-class wrong な NLOS 区間で yaw が狂う。停止中 (speed<0.7m/s) は yaw が更新されず固定。
3. **IMU bias が常に 0 と仮定** → accel bias 0.1 m/s² が 5s で 1.25m drift、PPC 0.5m PASS threshold を超えてしまう。
4. **平均 dead-reckoning が 7-28 秒と長い** → 上記 1-3 の誤差が増幅して PF estimate が hybrid から >20m wander、95%+ epoch で drift-skip → hybrid に戻る。
5. **Anchor が hybrid Status=4 のみ** → アンカーが切れた後、INS state (姿勢/bias) が流れ続けて補正が効かない。

Phase 9c は **本格 15-state INS-GNSS EKF** で上記 1-5 を全て根本解決する。

---

## 1. ゴール / 完了条件

### 必須 (Phase 9c 成功)
- 6-run honest aggregate で **>= 50.72% を達成** (Phase 9b 50.52% を +0.2pp 以上回復)。
- Phase 9b の `enable_imu_tc` フラグを残しつつ、**新しい `enable_ins_tc` フラグ** で `--methods rbpf+dd+gate+hybrid+ins_tc` を追加 (既存 method の挙動は完全に保持)。
- `experiments/results/ppc_ctrbpf_fgo_phase9c_full_runs.csv` を作成し、commit する。
- `python/gnss_gpu/ins_ekf.py` を新規モジュールとして追加 (PF とは独立にユニットテスト可能)。

### 目標 (Phase 9c 上振れ)
- 6-run honest aggregate で **51%+ (+0.3pp 以上 vs hybrid)**。これが達成できれば PPC で初めて hybrid baseline を超える成功事例となる。

### 失敗とみなす条件
- 6-run aggregate が 50.72% を超えない (= Phase 9b 同様、INS でも hybrid を超えられない構造的限界が確定)。この場合は memory `project_ppc_postprocess_ceiling.md` を更新して「INS でも no-op」を 5 つ目の lever 失敗例として記録、libgnss++ tuning に進む。

---

## 2. リポジトリ前提と参照ポイント

### 2.1 既に書いた / 動くもの

- `python/gnss_gpu/io/ppc.py:250-271` — `PPCDatasetLoader.load_imu()`。`imu.csv` の列: `time` (GPS TOW s), `acc_x/y/z` (m/s²), `gyro_x/y/z` (deg/s)。サンプリング 100Hz。bias 補正なし。
- `python/gnss_gpu/particle_filter_device.py:579-601` — `pf.position_update(ref_ecef, sigma_pos)`。GPU 上で per-particle Gaussian log-likelihood を加算する soft constraint。**Phase 9c でも INS prediction を渡す経路はこれを使う**。新規 CUDA kernel は不要。
- `experiments/exp_ppc_ctrbpf_fgo.py:880-1141` — PF メインループ。Phase 9b の挿入点 (1117 行付近 hybrid PU の後) と同じ場所に Phase 9c の INS update を入れる。
- `experiments/exp_ppc_ctrbpf_fgo.py:576-749` — Phase 9b の `_IMUAnchor` / `_integrate_imu_between` ヘルパ群。Phase 9c では完全に置き換えるが、PPC IMU 軸規約 (body x=forward, y=left, z=up; 静止時 acc_z ≈ +9.81) はそのまま継承。

### 2.2 PPC データの実験的事実 (確認済み)

```
$ head -3 datasets/PPC-Dataset-data/tokyo/run1/imu.csv
GPS TOW (s), GPS Week, Acc X (m/s^2), Acc Y (m/s^2), Acc Z (m/s^2),
   Ang Rate X (deg/s), Ang Rate Y (deg/s), Ang Rate Z (deg/s)
187470.00, 2324, 0.38587500, -0.32891250, 9.80980000, ...
```

- **Body frame**: x=forward, y=left, z=up (right-handed, +z aligned with vehicle vertical).
- **Specific force convention**: 静止時に `f_meas_body = (0, 0, +9.81)` ≈ 重力反作用上向き。よって `f_meas = R_n2b · (a_inertial - g_n)` で `g_n_enu = (0, 0, -9.81)` (重力ベクトルは下向き)。
- IMU レート: 100Hz (epoch 間 0.01s), GNSS rover epoch 5Hz (0.2s)。
- 全 6 run の hybrid Status 内訳 (`libgnss_rtk_pos_v5/*_full.pos`):
  - Status=4 (cm-class) は全体の **75-85%** (PPC データの大半)
  - Status=3 (m-class) が 10-25%
  - Status=1 (NLOS-heavy) が 1-5%
- ZUPT/Phase 9a の static 検出閾値 (`zupt_acc_norm_low=9.6`, `zupt_acc_norm_high=9.95`, `zupt_gyro_norm_max=1.5deg/s`) で大半の停止が拾える (典型 run で anchors の 60-75% が static 判定)。

### 2.3 hybrid passthrough = 何を意味するか

`experiments/results/libgnss_rtk_pos_v5/{city}_{run}_full.pos` は libgnss++ (third_party/gnssplusplus) が出力した 2026-04 commit `76ca2d8` 時点の RTK 解。各 epoch の Status:
- **4**: cm-class fix (RTK ambiguity 整数解、median 0.04m)。INS の position 観測として使う (sigma=0.05m 等の tight 値)。
- **3**: float / m-class (median ~5m)。INS の観測から外す (覆ってしまうと drift 補正が効かない) **か** 緩い sigma で入れる。
- **1**: NLOS-heavy (median 17-23m)。観測には使わない。

### 2.4 PF runner の既存変更パターン (Phase 9b で確立)

```python
# CTRBPFConfig dataclass にフィールド追加
enable_imu_tc: bool = False
imu_tc_pos_sigma_base_m: float = 0.5
# ...

# _config_variants() で新規 method label を作る
if "rbpf+dd+gate+hybrid+imu_tc" in args.methods:
    variants.append(CTRBPFConfig(..., enable_imu_tc=True,
                                 method_label="RBPF-velKF+DD+gate+hybrid+imu_tc"))

# main parser に CLI を生やす
parser.add_argument("--imu-tc-pos-sigma-base-m", type=float, default=0.5, ...)

# _run_ctrbpf_on_segment() の return に stats を追加
return positions, ms_per_epoch, dd_stats, ..., zupt_stats, imu_tc_stats

# loop 内: hybrid PU の直後に INS update を呼ぶ
if use_imu_tc:
    ...
    pf.position_update(imu_pred_pos, sigma_pos=sigma_imu)
```

Phase 9c も完全にこのパターンを踏襲する。**既存の Phase 9b コードは触らない** (新ロジックは別フラグで coexistence)。

---

## 3. Phase 9c 設計

### 3.1 EKF 状態 (15D, error-state)

ノミナル状態 `x_nom` をクラス内に持ち、誤差状態 `δx ∈ R^15` を EKF で扱う:

```
x_nom = (p_enu, v_enu, q_b2n, b_a, b_g)  # 13D nominal (quaternion 4D)
δx    = (δp,   δv,   δθ,    δb_a, δb_g) # 15D error
        |3    |3    |3     |3     |3
```

- `p_enu`: ENU 座標 (m)。原点は **run 開始時の最初の ground-truth または hybrid pos の ECEF**。EKF の P 行列もこの局所 ENU で扱う。Earth rotation effect は PPC の 数分間スケールでは無視可能 (10⁻⁵ rad/s × 数分 → 0.01° 未満)。
- `v_enu`: ENU 速度 (m/s)。
- `q_b2n`: クォータニオン (scalar-last `[qx, qy, qz, qw]`)。body → ENU rotation。
- `b_a`: accelerometer bias (body frame, m/s²)。
- `b_g`: gyroscope bias (body frame, rad/s)。

ノミナルから観測値への投影:
```
f_meas_body = R_n2b · (a_inertial_enu - g_enu) + b_a + n_a
ω_meas_body = ω_true_body + b_g + n_g
g_enu = (0, 0, -9.81)
```

### 3.2 INS Mechanization (ノミナル更新)

各 IMU サンプル間 dt について:

```
# 1. 角速度補正と姿勢更新
ω_corr = ω_meas - b_g
δq = quat_from_axis_angle(ω_corr * dt)   # exp(0.5 * Ω * dt) 近似
q_new = quat_normalize(quat_multiply(q, δq))

# 2. 比力補正と慣性加速度
f_corr = f_meas - b_a
a_inertial_enu = R(q_new) @ f_corr + g_enu

# 3. 速度・位置更新 (mid-point)
v_new = v + a_inertial_enu * dt
p_new = p + v * dt + 0.5 * a_inertial_enu * dt**2
```

**注**: q の更新と f の rotate に同じ q を使うと一段遅れる。実装上は `q_avg = slerp(q, q_new, 0.5)` を使うか、簡易的に rotate に `q_new` を使う。

### 3.3 EKF Predict (誤差状態)

連続時間 F (15×15):
```
F = [[0, I, 0,             0,    0      ],
     [0, 0, -[R·f_corr]_×, -R,   0      ],
     [0, 0, -[ω_corr]_×,   0,    -I     ],
     [0, 0, 0,             0,    0      ],
     [0, 0, 0,             0,    0      ]]
```
- `[v]_×` は 3×3 skew-symmetric 行列。
- `R = R(q)` は body→ENU rotation matrix。

離散化 (1 次):
```
Phi = I_15 + F * dt
P = Phi @ P @ Phi.T + Q * dt
```

プロセスノイズ Q の対角成分 (推奨初期値、必要に応じてチューニング):
```
sigma_a   = 0.05 m/s²/√Hz   # accel ホワイトノイズ
sigma_g   = 0.005 rad/s/√Hz # gyro ホワイトノイズ
sigma_ba  = 1e-4 m/s³/√Hz   # accel bias random walk
sigma_bg  = 1e-5 rad/s²/√Hz # gyro bias random walk
Q_diag = [0,0,0, sigma_a², sigma_a², sigma_a², sigma_g², sigma_g², sigma_g², sigma_ba², sigma_ba², sigma_ba², sigma_bg², sigma_bg², sigma_bg²]
```

### 3.4 EKF Update (GNSS 位置観測)

hybrid Status=4 epoch で hybrid pos を観測値とする:
```
H = [I_3, 0_{3×12}]   # 1-3 列が p_enu に対応
z = p_meas_enu - p_nom_enu
S = H @ P @ H.T + R
K = P @ H.T @ inv(S)
δx = K @ z
P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T  # Joseph form

# ノミナルへの注入 (injection)
p_nom += δx[0:3]
v_nom += δx[3:6]
q_nom = quat_normalize(quat_multiply(q_nom, quat_from_axis_angle(δx[6:9])))
b_a   += δx[9:12]
b_g   += δx[12:15]
```

R は対角:
- Status=4: `R = diag([0.05², 0.05², 0.10²])` (cm-class horizontal、垂直は少し緩く)
- Status=3 で観測に使う場合: `R = diag([5², 5², 10²])` (loose)
- Status=1: 観測に使わない

### 3.5 静止アライメント初期化 (重要)

最初の N ≥ 50 IMU サンプル (= 0.5s) が **連続して静止判定** された時点で初期化:

```
# 静止判定: |a| in [9.6, 9.95], |ω| <= 1.5 deg/s (Phase 9a と同じ閾値)
f_avg = mean(accel_body[static_window])  # 平均比力 = ほぼ重力反作用
ω_avg = mean(gyro_body[static_window])   # 平均角速度 = gyro bias

# Roll, pitch from gravity alignment (yaw は未確定)
# body x=forward, y=left, z=up; static で f = (g sin θ_pitch, -g sin φ_roll, g cos θ_pitch cos φ_roll)
roll  = atan2(-f_avg[1], f_avg[2])
pitch = atan2( f_avg[0], hypot(f_avg[1], f_avg[2]))
yaw   = 0.0  # 未確定。最初の動き出し (planar speed > 1.0 m/s) で確定。

# Bias 初期推定
b_g_init = ω_avg                     # gyro bias = static 平均
b_a_init = (0, 0, 0)                 # accel bias は roll/pitch とカップルするので静止からは分離不能。0 で開始し EKF が学習。

# 初期共分散 P
P_init = diag([1², 1², 1²,             # p (anchored to first hybrid pos)
               0.5², 0.5², 0.5²,       # v (静止なら 0)
               (1°)², (1°)², (10°)²,    # roll/pitch tight, yaw loose (まだ動いてない)
               0.1², 0.1², 0.1²,        # b_a (やや loose)
               (0.5°/s)², ..., ...])    # b_g (gyro avg から少し loose)
```

**Yaw 確定**: 動き出し (planar speed >= 1.0 m/s) を検知した最初の epoch で:
- `yaw_meas = atan2(v_north_enu, v_east_enu)` を hybrid velocity から算出。
- これを INS の状態に注入: `δθ_yaw = yaw_meas - current_yaw`、δθ = (0, 0, δθ_yaw) を quat_multiply で適用。
- P の yaw 対角を (5°)² 程度に絞る。

### 3.6 ENU ↔ ECEF 変換

EKF は **ENU 局所座標** で動かす (数値安定性とフレーム規約の単純化のため)。runner 側で:
- 入力 (hybrid_pos, IMU): 元のフレーム → 局所 ENU (差分のみ ECEF→ENU 回転で変換)。
- 出力 (INS p_enu): 局所 ENU → ECEF 復元 → `pf.position_update` に渡す。

局所 ENU 原点は run 開始時に固定 (`origin_ecef`, `origin_lat`, `origin_lon`)。run 中の ENU 距離は最大数 km なので flat-earth 近似で十分。

### 3.7 PF への INS 注入

各 rover epoch (5Hz) で:
1. INS を IMU 100Hz サンプルで `t_{i-1}` → `t_i` まで propagate。
2. 現在の `p_nom_enu` を ECEF に戻す → `p_ins_ecef`。
3. 現在の `P[0:3,0:3]` の trace を sigma に変換: `sigma_ins = sqrt(trace(P[0:3,0:3]) / 3)`、ただし 0.05m floor + 5m ceiling。
4. `pf.position_update(p_ins_ecef, sigma_pos=sigma_ins)` を呼ぶ。
5. hybrid Status=4 epoch なら、INS にも hybrid を観測値として update step を発行。Status=3 はオプショナル (CLI で有効化、デフォルト OFF)。Status=1 は INS 更新しない。
6. 出力 emission: Status=1/3 epoch では `pf.estimate()` を出力 (Phase 9b と同じ)。Status=4 は hybrid passthrough。

### 3.8 Safety Gates

Phase 9b と同様、以下を CLI で:
- `--ins-tc-emit-max-diff-m` (default 20m): PF estimate が hybrid から離れすぎたら hybrid に戻る。
- `--ins-tc-max-dr-seconds` (default 10s): INS が長時間補正なく drift したら更新しない。
- `--ins-tc-max-disagreement-m` (default 30m): INS 予測が hybrid から離れすぎたら sanity skip。

---

## 4. ファイル変更計画 (順序付き)

### 4.1 新規ファイル: `python/gnss_gpu/ins_ekf.py`

```python
"""15-state INS-GNSS EKF (error-state, ENU frame, online IMU bias estimation).

Conventions:
  - Body frame: x=forward, y=left, z=up (PPC IMU convention)
  - Navigation frame: local ENU centered at first epoch
  - Gravity: g_enu = (0, 0, -9.81)
  - Quaternion: scalar-last [qx, qy, qz, qw], body-to-navigation
  - Specific force: f_meas = R_n2b · (a_inertial - g_enu) + b_a + n_a
"""

from dataclasses import dataclass
import numpy as np

@dataclass
class INSConfig:
    sigma_acc_noise: float = 0.05      # m/s²/√Hz
    sigma_gyro_noise: float = 0.005    # rad/s/√Hz
    sigma_acc_bias_rw: float = 1e-4    # m/s³/√Hz
    sigma_gyro_bias_rw: float = 1e-5   # rad/s²/√Hz
    static_acc_low: float = 9.6
    static_acc_high: float = 9.95
    static_gyro_max_dps: float = 1.5
    align_min_static_samples: int = 50  # 0.5s @ 100Hz
    yaw_init_min_speed_mps: float = 1.0
    init_pos_sigma_m: float = 1.0
    init_vel_sigma_mps: float = 0.5
    init_attitude_sigma_rp_rad: float = np.deg2rad(1.0)
    init_attitude_sigma_yaw_rad: float = np.deg2rad(10.0)
    init_acc_bias_sigma: float = 0.1
    init_gyro_bias_sigma_rps: float = np.deg2rad(0.5)


class INSEKF:
    def __init__(self, config: INSConfig):
        self.config = config
        self.aligned: bool = False
        self.yaw_initialized: bool = False
        self._static_buffer: list[tuple[float, np.ndarray, np.ndarray]] = []  # (t, accel, gyro) of latest static samples
        # Nominal state
        self.p = np.zeros(3)         # ENU (m)
        self.v = np.zeros(3)         # ENU (m/s)
        self.q = np.array([0., 0., 0., 1.])  # scalar-last (qx,qy,qz,qw)
        self.b_a = np.zeros(3)       # body (m/s²)
        self.b_g = np.zeros(3)       # body (rad/s)
        # Error covariance
        self.P = np.zeros((15, 15))
        self.last_t: float | None = None

    # --- Initialization -------------------------------------------------
    def feed_imu_for_alignment(self, t: float, accel_body: np.ndarray, gyro_body_dps: np.ndarray) -> None:
        """Buffer static IMU samples; align (set q, b_g, P) when enough collected."""

    def initialize_position(self, p_enu_init: np.ndarray) -> None:
        """Anchor INS position to a known ENU coordinate (e.g. first hybrid pos converted to ENU). Resets P[0:3,0:3]."""

    def initialize_yaw_from_velocity(self, v_enu: np.ndarray) -> bool:
        """Set yaw from ENU velocity course-over-ground; require |v_planar| >= yaw_init_min_speed_mps."""

    # --- Mechanization & EKF predict ------------------------------------
    def propagate(self, imu_samples: np.ndarray) -> None:
        """Integrate INS through a sequence of IMU samples [(t, ax, ay, az, gx, gy, gz)],
        gyro in deg/s. Updates nominal state and EKF covariance P."""

    # --- EKF update -----------------------------------------------------
    def update_position_enu(self, p_meas_enu: np.ndarray, sigma_pos_m: tuple[float, float, float]) -> None:
        """Position-only EKF update. Injects δx into nominal state."""

    # --- Output ---------------------------------------------------------
    def position_sigma_m(self) -> float:
        """Scalar 3D position sigma derived from trace(P[0:3,0:3])."""

    def position_enu(self) -> np.ndarray:
        return self.p.copy()

    def position_ecef(self, origin_ecef: np.ndarray, origin_lat: float, origin_lon: float) -> np.ndarray:
        """Convert ENU position to ECEF using local origin frame."""
```

実装メモ:
- `quat_from_axis_angle`, `quat_multiply`, `quat_normalize`, `quat_to_rotmat` は内部ヘルパで OK。`scipy.spatial.transform.Rotation` を使うと依存が増えるので NumPy 自前で書く (40 行程度)。
- `_skew_symmetric(v)` ヘルパを 1 つ。
- `propagate` は IMU を 1 サンプルずつ for-loop で回して OK (100Hz × 0.2s = 20 サンプル/epoch、性能的に問題なし)。

ユニットテスト推奨 (`tests/test_ins_ekf.py`):
- 静止 IMU を 5s 流して位置・速度・姿勢が drift しないこと (bias 0 前提)。
- 一定加速 (bx=1 m/s², 5s) で `v ≈ (5, 0, 0)`, `p ≈ (12.5, 0, 0)` ENU になること (yaw=0, roll=pitch=0)。
- 静止 → 5s gyro_z=10 deg/s → 静止で yaw が 50 度回ること。
- Position update が誤差を吸収して P が縮むこと。

### 4.2 `experiments/exp_ppc_ctrbpf_fgo.py` の変更

#### A. CTRBPFConfig (line 67〜) に新規フィールド

```python
# Phase 9c: full INS-GNSS EKF (15-state). Replaces the Phase 9b yaw-only
# pre-integration with proper attitude tracking + online IMU bias estimation.
enable_ins_tc: bool = False
ins_tc_emit_pf_hybrid_statuses: tuple[int, ...] = (1, 3)
ins_tc_obs_status_4_sigma_m: float = 0.05
ins_tc_obs_status_3_sigma_m: float = 0.0     # 0 = skip Status=3 obs; >0 = use as loose obs
ins_tc_max_dr_seconds: float = 10.0
ins_tc_max_disagreement_m: float = 30.0
ins_tc_emit_max_diff_m: float = 20.0
ins_tc_pf_pu_floor_sigma_m: float = 0.1   # floor on sigma fed to pf.position_update
ins_tc_pf_pu_ceiling_sigma_m: float = 5.0 # ceiling
# 静止アライメント閾値 (Phase 9a と共通でも可)
ins_tc_align_acc_low: float = 9.6
ins_tc_align_acc_high: float = 9.95
ins_tc_align_gyro_max_dps: float = 1.5
ins_tc_align_min_samples: int = 50
ins_tc_yaw_init_min_speed_mps: float = 1.0
```

#### B. 新規 stats dataclass (Phase 9b の `_IMUTCStats` の隣)

```python
@dataclass
class _INSTCStats:
    aligned_at_epoch: int = -1           # -1 = never aligned
    yaw_initialized_at_epoch: int = -1
    epochs_evaluated: int = 0
    pu_applied: int = 0
    pu_skipped_not_aligned: int = 0
    pu_skipped_no_yaw: int = 0
    pu_skipped_dr_too_long: int = 0
    pu_skipped_disagreement: int = 0
    obs_status_4_used: int = 0
    obs_status_3_used: int = 0
    emit_pf_estimate: int = 0
    emit_skipped_pf_drift: int = 0
    final_acc_bias_norm: float = 0.0
    final_gyro_bias_norm_dps: float = 0.0
    final_pos_sigma_m: float = 0.0
```

#### C. `_run_ctrbpf_on_segment` の signature & 初期化

```python
def _run_ctrbpf_on_segment(
    ...
) -> tuple[..., _IMUTCStats, _INSTCStats]:
    ...
    ins_tc_stats = _INSTCStats()
    use_ins_tc = config.enable_ins_tc and imu is not None
    ins_ekf: INSEKF | None = None
    ins_origin_ecef: np.ndarray | None = None
    ins_origin_lat: float | None = None
    ins_origin_lon: float | None = None
    if use_ins_tc:
        from gnss_gpu.ins_ekf import INSEKF, INSConfig
        ins_ekf = INSEKF(INSConfig(
            sigma_acc_noise=...,
            static_acc_low=config.ins_tc_align_acc_low,
            ...
        ))
```

#### D. PF loop 内の挿入点

Phase 9b の `if use_imu_tc:` ブロック (現在 `experiments/exp_ppc_ctrbpf_fgo.py:1399` 付近) の **すぐ下** に並列で `if use_ins_tc:` ブロックを追加。Phase 9b と Phase 9c は同時 enable しない (config 検証で排他)。

擬似コード:
```python
if use_ins_tc:
    t_now = float(times[i])
    # 1. ENU origin を最初の有効 hybrid pos で確定
    if ins_origin_ecef is None and hp is not None and np.all(np.isfinite(hp)):
        ins_origin_ecef = np.asarray(hp, dtype=np.float64).copy()
        lat, lon, _ = _ecef_to_llh(*ins_origin_ecef)
        ins_origin_lat = lat
        ins_origin_lon = lon

    # 2. IMU を t_{i-1} から t_i まで取り出して INS に流す
    if i > 0 and ins_origin_ecef is not None:
        imu_window = _slice_imu(imu, imu_t, times[i-1], t_now)  # [(t, ax, ay, az, gx, gy, gz)]
        # 静止検出してアライメント (まだなら)
        if not ins_ekf.aligned:
            for sample in imu_window:
                ins_ekf.feed_imu_for_alignment(sample[0], sample[1:4], sample[4:7])
            if ins_ekf.aligned:
                ins_tc_stats.aligned_at_epoch = i
                p_enu_init = _ecef_to_enu(hp, ins_origin_ecef, ins_origin_lat, ins_origin_lon)
                ins_ekf.initialize_position(p_enu_init)

        # アライメント済みなら propagate
        if ins_ekf.aligned:
            ins_ekf.propagate(imu_window)
            # 動き出してて yaw 未確定なら yaw 注入
            if not ins_ekf.yaw_initialized:
                v_guide = hybrid_velocity.get(round(t_now, 1))
                if v_guide is not None:
                    v_enu = _ecef_velocity_to_enu(hp, v_guide)
                    if ins_ekf.initialize_yaw_from_velocity(v_enu):
                        ins_tc_stats.yaw_initialized_at_epoch = i

    # 3. hybrid Status=4 なら INS の position update
    st_now = hybrid_status.get(round(t_now, 1)) if hybrid_status else None
    if ins_ekf is not None and ins_ekf.aligned and hp is not None:
        if st_now == 4 and config.ins_tc_obs_status_4_sigma_m > 0:
            p_meas_enu = _ecef_to_enu(hp, ins_origin_ecef, ins_origin_lat, ins_origin_lon)
            sigma = config.ins_tc_obs_status_4_sigma_m
            ins_ekf.update_position_enu(p_meas_enu, (sigma, sigma, sigma * 2))
            ins_tc_stats.obs_status_4_used += 1
        elif st_now == 3 and config.ins_tc_obs_status_3_sigma_m > 0:
            p_meas_enu = _ecef_to_enu(hp, ...)
            ins_ekf.update_position_enu(p_meas_enu, (loose,) * 3)
            ins_tc_stats.obs_status_3_used += 1

    # 4. INS prediction を PF に注入
    if ins_ekf is not None and ins_ekf.aligned and ins_ekf.yaw_initialized:
        p_ins_ecef = ins_ekf.position_ecef(ins_origin_ecef, ins_origin_lat, ins_origin_lon)
        sigma_ins = max(config.ins_tc_pf_pu_floor_sigma_m,
                        min(config.ins_tc_pf_pu_ceiling_sigma_m, ins_ekf.position_sigma_m()))
        # Sanity & dr-too-long check
        run_pf_pu = True
        ...
        if run_pf_pu:
            pf.position_update(p_ins_ecef, sigma_pos=sigma_ins)
            ins_tc_stats.pu_applied += 1

    # 5. Emission switch (Phase 9b と同じパターン)
    if st_now is not None and int(st_now) in {int(s) for s in config.ins_tc_emit_pf_hybrid_statuses}:
        ins_tc_emit_pf_here = True
```

`est = pf.estimate()` の後の emission 分岐に `ins_tc_emit_pf_here` を追加。

#### E. CLI / variant / CSV 列

- 新規 CLI: `--ins-tc-obs-status-4-sigma-m`, `--ins-tc-obs-status-3-sigma-m`, `--ins-tc-max-dr-seconds`, `--ins-tc-max-disagreement-m`, `--ins-tc-emit-max-diff-m`, `--ins-tc-pf-pu-floor-sigma-m`, `--ins-tc-pf-pu-ceiling-sigma-m`, `--ins-tc-emit-pf-hybrid-statuses` (default `"1,3"`).
- 新規 method label:
  - `rbpf+dd+gate+hybrid+ins_tc` — 単独 INS。
  - `rbpf+dd+gate+hybrid+zupt+ins_tc` — ZUPT (Phase 9a) と併用。
- CSV 行に `_INSTCStats` の全フィールドを追加 (列名 prefix `ins_tc_`)。
- `_config_variants()` の base 辞書に新規 config フィールドを通す。

#### F. main() 側の IMU loader 起動条件

```python
if any(v.enable_zupt or v.enable_imu_tc or v.enable_ins_tc for v in variants):
    imu_run = loader.load_imu()
```

```python
imu_for_variant = imu_run if (variant.enable_zupt or variant.enable_imu_tc or variant.enable_ins_tc) else None
hybrid_v_for_variant = hybrid_velocity_run if (variant.enable_hybrid_velocity_guide or variant.enable_imu_tc or variant.enable_ins_tc) else None
hybrid_status_for_variant = hybrid_status_run if (... or variant.enable_ins_tc) else None
```

### 4.3 ユニットテスト

`tests/test_ins_ekf.py` (新規):
- `test_static_zero_drift`: 5s 静止で `np.allclose(ins.p, 0, atol=0.01)`.
- `test_constant_accel`: 5s ax=1m/s² で `v[0] ≈ 5.0`, `p[0] ≈ 12.5`.
- `test_yaw_rotation`: 5s gyro_z=10°/s で yaw が 50° 回る (RPY 表現で確認).
- `test_static_alignment`: 静止 0.5s 流して `aligned=True`, `q ≈ identity` (level 想定).
- `test_position_update_reduces_p`: 適当な誤差を入れた後 update して P[0:3,0:3] の対角が縮む.

---

## 5. 実行 / 評価

### 5.1 Smoke (まずこれを通す)

```bash
python3 experiments/exp_ppc_ctrbpf_fgo.py \
  --runs tokyo/run1 \
  --max-epochs 1500 \
  --methods rbpf+dd+gate+hybrid,rbpf+dd+gate+hybrid+ins_tc \
  --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 \
  --hybrid-pos-suffix _full.pos \
  --n-particles 20000 \
  --results-prefix ppc_ctrbpf_fgo_phase9c_smoke
```

期待される挙動:
- `aligned_at_epoch >= 1` (最初の数 epoch でアライメント完了)
- `yaw_initialized_at_epoch < 100` (動き出しで yaw 確定)
- `obs_status_4_used > 0` (Status=4 で INS が更新される)
- `pu_applied >= 50%` of epochs
- `final_acc_bias_norm < 0.5 m/s²`, `final_gyro_bias_norm_dps < 1.0 deg/s` (バイアスが現実的範囲)
- score が baseline と同等以上 (-0.1pp 以上)

### 5.2 Honest 6-run aggregate

```bash
python3 experiments/exp_ppc_ctrbpf_fgo.py \
  --runs all \
  --methods rbpf+dd+gate+hybrid,rbpf+dd+gate+hybrid+ins_tc,rbpf+dd+gate+hybrid+zupt+ins_tc \
  --hybrid-pos-dir experiments/results/libgnss_rtk_pos_v5 \
  --hybrid-pos-suffix _full.pos \
  --n-particles 20000 \
  --results-prefix ppc_ctrbpf_fgo_phase9c_full
```

`experiments/results/ppc_ctrbpf_fgo_phase9c_full_runs.csv` を commit する。所要時間目安: 15-25 分 (Phase 9b で 6-run × 2 method = 14 分だったので、3 method なら ~22 分)。

### 5.3 評価ロジック

```python
import csv, collections
rows = list(csv.DictReader(open("experiments/results/ppc_ctrbpf_fgo_phase9c_full_runs.csv")))
agg = collections.defaultdict(lambda: [0.0, 0.0])
for r in rows:
    agg[r["method"]][0] += float(r["honest_pass_m"])
    agg[r["method"]][1] += float(r["honest_total_m"])
for m, (p, t) in agg.items():
    print(f"{m}: {100*p/t:.2f}% (pass {p:.0f}m / total {t:.0f}m)")
```

判定:
- INS aggregate >= 50.92% → **大成功**。Phase 4/8/9a/9b 全敗の天井を初突破。memory `project_ppc_postprocess_ceiling.md` 更新 + commit + 完了報告。
- INS aggregate 50.72-50.91% → **break-even 成功**。hybrid と同等まで戻せたので、INS の正しさは示された。次は libgnss++ tuning に進む。
- INS aggregate < 50.72% → **失敗**。memory 更新して INS でもダメだった事を記録、libgnss++ tuning に進む。

---

## 6. リスクと対処

| リスク | 兆候 | 対処 |
|---|---|---|
| アライメントが取れない (vehicle が動き始めても static 検出に入らない) | smoke で `aligned_at_epoch=-1` | `--ins-tc-align-min-samples` を緩める (50→20)、acc 範囲を広げる (9.5-10.0) |
| Yaw 注入後も姿勢が drift して INS pos が hybrid と合わない | `disagreement` skip が多発 | `init_attitude_sigma_yaw_rad` を 5° → 15° に loose、最初の数 epoch で yaw を再注入 |
| Bias が発散 (acc bias > 1 m/s²) | `final_acc_bias_norm` が大きい / observability 不足 | Q の bias RW を小さく (1e-4 → 1e-5)、Status=4 観測の頻度を上げる (sigma 緩めて Status=3 も観測に使う) |
| score が Phase 9b より下がる (PF が INS で誤誘導される) | `pf_drift_skip` 大、`emit_pf` で逆ギレ | `--ins-tc-emit-max-diff-m` を 20m → 10m に厳しく、または emit_statuses から 1 を外して 3 のみに |
| 性能が遅い (PF loop が ms/epoch 倍増) | `ms/epoch > 5` | Numba で `propagate` を JIT、または IMU 100Hz を 20Hz にデシメート |
| EKF が NaN を吐く (Cholesky 失敗等) | smoke でクラッシュ | Joseph form で update、P の対称化 `P = 0.5*(P + P.T)`、最低固有値クランプ |

---

## 7. やってはいけない事 (sanity guardrails)

- **既存 Phase 9b コード (`enable_imu_tc` 関連) を削除しない**。並列して残し、両者を比較できる状態を保つ。
- **PF GPU kernel に新規 update を追加しない**。CUDA 変更は範囲外。`pf.position_update` の既存 API のみで対応する。
- **Earth rotation / Coriolis 補正を入れない**。PPC 数分間スケールでは < 0.01m の影響、複雑度を増すだけ。後段の最適化候補。
- **Multi-frequency / L2 carrier 等を併用しない**。INS 単独の効果を測りたい。Phase 9c 結果が出てから他を被せる。
- **commit 時のメッセージ規約**: 既存 commit (`PPC Phase 9b: ...`) に倣い、honest 6-run aggregate 数値を本文に必ず含める。`Co-Authored-By` は付けない (~/.claude/CLAUDE.md 規約)。

---

## 8. 完了時のアウトプット

1. **コード**:
   - `python/gnss_gpu/ins_ekf.py` (新規)
   - `experiments/exp_ppc_ctrbpf_fgo.py` (Phase 9c 統合)
   - `tests/test_ins_ekf.py` (新規)
2. **データ**:
   - `experiments/results/ppc_ctrbpf_fgo_phase9c_full_runs.csv`
3. **commit** 1 つ:
   - 件名: `PPC Phase 9c: 15-state INS-GNSS EKF tight coupling (XX.XXpp vs hybrid 50.72%)`
   - 本文に per-method aggregate, per-run delta, INS stats (alignment ms, final bias)
4. **memory 更新** (`/home/sasaki/.claude/projects/-media-sasaki-aiueo-ai-coding-ws-gnss-gpu/memory/project_ppc_postprocess_ceiling.md`):
   - Phase 9c 結果を追記
   - 結論 (天井突破した / しなかった) を更新
   - 次の推奨方針を更新

---

## 9. 用語集 / 参考

- **PPC2024 honest aggregate**: `score_ppc2024(estimated, ref)` を全 epoch (rover が emit していない epoch も `[0,0,0]` で埋めて denominator 含める) で計算。`ppc_score.py` 参照。
- **hybrid passthrough**: libgnss++ の `.pos` 出力を Status≥1 epoch でそのまま emission に使う mode (`hybrid_emit_pf_estimate=False`)。Phase 9b/c では Status=4 のみ passthrough、Status=1/3 は PF estimate に切替。
- **DD AFV**: double-differenced carrier phase Ambiguity Function Value. `pf.update_dd_carrier_afv` で per-particle Gaussian に整数解探索なしで likelihood を加算。
- **RBPF velocity-KF**: 各 particle に Kalman filter で velocity 状態を持たせる。Doppler 観測で update。
- **Status 4/3/1**: libgnss++ の `.pos` Q 列。4=fix (cm), 3=float (m-class), 1=NLOS-heavy (m-class、品質低)。

---

## 10. 引き継ぎチェックリスト (codex 側)

- [ ] `git log --oneline -5` で `f510f23` (Phase 9b) が HEAD だと確認
- [ ] `python/gnss_gpu/ins_ekf.py` を新規作成 (§4.1 のテンプレに沿って)
- [ ] `tests/test_ins_ekf.py` で 5 テスト全パス
- [ ] `experiments/exp_ppc_ctrbpf_fgo.py` に Phase 9c 統合 (§4.2)
- [ ] Smoke (§5.1) で `aligned`, `yaw_initialized`, `pu_applied > 0` を確認
- [ ] 6-run aggregate (§5.2) を実行し CSV を commit
- [ ] Memory 更新 (§8.4)
- [ ] コミット 1 つ作成 (§8.3)
