# Task D: Wide-lane carrier-phase PR を実験パイプラインに統合して Odaiba 評価

## ゴール
`wide_lane.py` (実装済み) を PF 実験パイプラインに統合し、Odaiba 全区間で wide-lane carrier-phase pseudorange が P50 を改善するか評価する。

## 背景
- 現在の Odaiba ベスト: P50=1.65m, RMS=6.45m (PF 100K, full observation stack)
- TDCP, adaptive TDCP, smoother の組み合わせでは改善限界に到達
- Wide-lane は carrier phase の integer ambiguity (N_wl) を固定して pseudorange 精度を向上させる手法
- N_wl 固定後の wide-lane PR は ~0.86m 精度 (vs 通常 PR の ~3-5m)

## 実装済みモジュール
- `python/gnss_gpu/wide_lane.py`: `WidelaneResolver` クラス
  - `update(prn, L1_cycles, L2_cycles, P1_m, P2_m)` — N_wl 蓄積
  - `get_fixed_ambiguity(prn) -> int | None` — 固定判定 (std < 0.4, >= 5 epochs)
  - `get_widelane_pseudorange(prn, L1_cycles, L2_cycles) -> float | None` — 固定済み衛星の wide-lane PR

## やること

### Step 1: RINEX から L2 データを取得する仕組み追加

**`python/gnss_gpu/io/urbannav.py`** の `load_experiment_data()` を拡張:

- 現在は L1 のみ取得 (`obs_code="C1C"`)
- L2 pseudorange (`C2W`) と L2 carrier phase (`L2W`) も衛星ごとに取得して返す
- L1 carrier phase (`L1C`) も取得する（wide-lane に必要）
- 返り値の dict に `l1_carrier_per_epoch`, `l2_carrier_per_epoch`, `l2_pr_per_epoch` を追加
  - 各要素は `list[dict[str, float]]` (satellite_id → value)

**RINEX obs types** (Odaiba rover_trimble.obs から確認済み):
- GPS: `C1C, L1C, D1C, S1C, C2W, L2W, D2W, S2W, ...`
- L2 fallback codes: GPS `C2W`/`L2W`, QZSS `C2X`/`L2X`

**追加する fallback dict** (L1 carrier, L2 PR, L2 carrier):
```python
_SYSTEM_L1_CARRIER_FALLBACKS = {
    "G": ("L1C", "L1W", "L1X"),
    "E": ("L1X", "L1C"),
    "J": ("L1C", "L1X", "L1Z"),
}
_SYSTEM_L2_PR_FALLBACKS = {
    "G": ("C2W", "C2X", "C2L"),
    "J": ("C2X", "C2L"),
}
_SYSTEM_L2_CARRIER_FALLBACKS = {
    "G": ("L2W", "L2X", "L2L"),
    "J": ("L2X", "L2L"),
}
```

### Step 2: 実験スクリプトに wide-lane 統合

**新規作成: `experiments/exp_widelane_eval.py`**

`exp_pf_smoother_eval.py` をベースに、以下を追加:
1. `WidelaneResolver` を import
2. エポックループ内で:
   - L1/L2 carrier phase と L2 pseudorange を取得
   - `resolver.update(prn, L1, L2, P1, P2)` で N_wl を蓄積
   - 固定済み衛星は `get_widelane_pseudorange()` で carrier-phase PR を取得
   - **PF の weight ステップ**: 固定済み衛星は wide-lane PR を使い、未固定は通常 PR を使う
     - gnssplusplus の `CorrectedMeasurement` の `corrected_pseudorange` を wide-lane PR で**置換**
     - 置換時は satellite_ecef 等は変えない（同じ衛星なので）

**重要な対応関係**:
- gnssplusplus の CorrectedMeasurement には `prn` と `system_id` がある
- RINEX の satellite_id は "G01", "G15" 等の文字列
- マッチング: system_id=0 (GPS) + prn=1 → "G01"

### Step 3: 評価

```bash
cd /path/to/gnss_gpu/experiments
export PYTHONPATH=".:../python:../third_party/gnssplusplus/build/python:../third_party/gnssplusplus/python"

# Wide-lane 評価 (Odaiba のみ、L1+L2 が必要)
python3 exp_widelane_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --position-update-sigma 1.5

# ベースライン比較
python3 exp_pf_smoother_eval.py --data-root /tmp/UrbanNav-Tokyo --runs Odaiba \
  --n-particles 100000 --position-update-sigma 1.5
```

**出力**: `experiments/results/widelane_eval.csv` に P50, P95, RMS, n_wl_fixed_sats 等を記録。

### Step 4: ログ出力

エポックごとに以下をログ (verbose or summary):
- 固定済み衛星数 / 全衛星数
- N_wl 固定に使ったエポック数
- wide-lane PR で置換した衛星の割合

## 注意事項

1. **Shinjuku/HK は L1 only** → Odaiba のみ対象。rover_source は `"trimble"` を使う（ublox は L1 only）
2. **N1 (L1 ambiguity) は未解決** — `wide_lane.py` は N_wl のみ。carrier-phase PR は wide-lane レベル (~0.86m) の精度
3. **cycle slip**: carrier phase が途切れたら `resolver.reset(prn)` でその衛星をリセットすべき
4. `load_experiment_data()` の変更は **既存の呼び出し元を壊さない** よう、新パラメータはデフォルト None で optional に
5. exp_pf_smoother_eval.py の `load_pf_smoother_dataset()` とは別に、RINEX 直接読みが必要（gnssplusplus 経由では L2 が取れないため）

## 判断基準
- Odaiba P50 < 1.5m なら有望
- N_wl 固定率が低い（< 30%）場合は cycle slip 検出の改善が必要
