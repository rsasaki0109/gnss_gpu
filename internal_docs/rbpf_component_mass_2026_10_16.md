# RBPFのDD展開と成分一致質量をまとめて計算する (D235, 2026-10-16)

**PF全ステップのCPU比較で3/6条件が21.58〜23.16%短縮し、事前gateを通過した。** 64粒子/6または32初期仮説/64行/4groupの2条件は元のcluster_relin無効構成で、各5/5round改善した。もう1条件は64粒子/6初期仮説/128行/8group。前回D234のキー・親照合高速化を含むCPUを対照に使い、さらに今回の準備/guardを含めて測定した。GPU対最適化後CPUの主screenは0/6。条件付きの実験版として保存し、製品・既定動作は維持する。

## 変更と数値の保持

元stepのFIX成分判定では、全粒子の`_pair_dd`でNumPy配列を作り、MAP側のDDが有限なペアごとに全粒子を走査する。`isfinite(di) and int(di)==int(dd_map[pi])`を満たす粒子の`float(w[i])`を粒子順に加算する。新しいCPython C++拡張はこのDD展開と質量集計を1callへまとめ、粒子ごとのNumPy配列/スカラー操作を省く。重みの分母は元の全粒子のままで、空/未定義DDが一致質量へ加算されない意味も維持する。

最初に全H/group/sat/value/pairsのexact builtin container/intをpreflightする。各DDは元と同じ任意精度Python整数の`zr-zj`を先に計算し、その差だけを元と同じFP64へ1回変換する。group/sat/整数を固定幅型へ詰めない。2**16000のgaugeが相殺される場合や2**53を超える変換も元と同じbitsになる。**元のDD自体がFP64であることは維持し、無限の整数精度をDD配列へ追加したとは主張しない。**

DD展開は粒子→全pairの元順序で行い、有限targetのないpairも省略しない。そのため非target pairで起きる整数→doubleのOverflowErrorも飛ばさない。元と同じ各成分の比較では、有限binary64を0方向へ切り捨てた値がbinary64で正確に表せる整数である性質を使う。`trunc(di)==trunc(target)`は任意長のPython `int(di)==int(target)`と同じで、int64へ変換しない。±0、fractional target、非常に大きい値も元と一致する。

各成分の質量を+0.0から始め、元と同じ粒子i昇順でFP64加算する。木構造の並列和、reassociation、FP32、FMA、許容差、近似閾値は使わない。`/fp:strict`でbuildする。finite target列の昇順にbuiltin intキー/float値のdictを返す。元の`am > cfg.gamma_fix`、gammaの最小値、部分FIX、conditional、feedback、resampling/RNGは元のPython codeのまま。

exactなNumPy float64 vector/C-contiguous buffer、元のParticle/EpochProblem、有限[0,1]重みとliteral `_pair_dd`をnative domainに限定する。custom型/helper/binding、layout/dtype/重みの範囲外は抽出した元blockを1回実行する。NumPyのunaligned exportは`=d`となるため元へfallbackする。aligned readonly/writableはnativeで扱える。strict NumPy policyはこれまでどおり元step全体へ戻して部分state/RNGも保つ。

コンパイラはD227 move adaptationと同じASTから、この3statementだけをcallbackへ置換する。置換を戻して元ASTと一致することもassertした。実runtime/moduleを上書きせず、D233のlocal view/D234親照合を継承する。全粒子/候補/prior、FP64/2IRLS、FIX/hold/graphを変更しない。D226数値DLL/D233キー拡張/D234親拡張はbyte単位で同じで、D231 warp/D232 DD packingとは併用しない。

## 完全PF-stepの測定

6条件×5round×3group×5mode=450timed calls、30warm snapshots。D234 CPU/新CPU/新GPU/D234 GPU/D234 CPU A/Aを直列回転する。guard、Hlist作成、preflight、DD展開、全質量計算、既存move/pack/transfers/conditionals/FIX/resamplingを含む。restore/setup/warm/検証は時間外。extract/time/satellite/integer-searchはfixtureで、graph solveは対象外。
自processの1許可CPU/1BLAS thread、通常priority、host/GPU非占有。paired group短縮率のround内中央値→5round中央値で判定し、独立p50同士の比とは異なる。CPUは同backend旧CPU比10%以上、4/5round以上改善、A/A中央値abs5%未満を要求した。GPU主screenは旧GPUと新CPUの両方へ同じ改善基準を要求する。判定は丸め前。失敗・負値・対照を保持し、速度の再測定はしない。

| particles/initial_unique/rows/groups | CPU p50 ms D234→新 | CPU同士paired短縮 | GPU同士paired短縮 | GPU対新CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|---:|---:|---:|---:|---:|:---|:---|
| 64/6/64/4 | 14.116 → 11.044 | +21.578% | +19.877% | -2.950% | -0.643% | PASS | FAIL |
| 64/32/64/4 | 28.024 → 22.270 | +21.824% | +20.951% | -2.846% | +3.691% | PASS | FAIL |
| 64/64/64/4 | 41.300 → 37.020 | +5.230% | +8.493% | -14.091% | -3.157% | FAIL | FAIL |
| 128/32/64/4 | 35.196 → 28.068 | +19.076% | +12.975% | -13.509% | -6.716% | FAIL | FAIL |
| 64/6/128/8 | 30.405 → 22.665 | +23.163% | +22.397% | -13.254% | +0.239% | PASS | FAIL |
| 128/128/128/8 | 401.603 → 412.178 | +1.544% | +4.888% | -8.009% | +1.674% | FAIL | FAIL |

CPU主screen3/6、旧GPU同士の副screen3/6、GPU主screen0/6。A/A失敗1/6を残した。128粒子/32初期仮説のCPU19.08%は3/5round改善でA/Aも−6.72%のため採用範囲に含めない。高多様性128/128/128/8はCPU1.54%で基準未達。全条件へ一般化せず、重い親照合やgraph全体の利得へ読み替えない。個別roundが揺れる原因を外部負荷等に断定しない。

## 正確性と実走

86tests PASS。3seed×5サイズ×20case=300matrix比較、巨大gauge/FP64変換、fractional/NaN/Inf target、非target overflow、順序和がtree sumと異なる反例、FIX境界の前後1ULP、custom型/binding/error、10,000call参照数・入力更新、3backend×3seed×3構成×4epochで全保存state/位置/P/decision/RNGを比較した。全450timed state hashと30warm snapshotも同backendのD234と一致し、新しい数値許容差は追加しない。既知warning3はconfig1と元strict-policy underflow2。CUDA memcheckは17targeted tests PASS、warning1、error/leak0。新CUDAコードを追加した検証ではなく、既存数値DLLとの統合検証である。

初回testは84PASS/1FAILで、巨大targetをdtype未指定でNumPy配列へ渡したfixtureがobject配列になり、元Python oracleがTypeErrorを投げた。初回source/plan/logを保存し、fixtureへFP64指定を加え、object配列の元例外fallbackテストも追加して86PASSとした。実装/C++/buildは変えない。補助buffer validatorもunalignedをnativeと期待した初回assert失敗を保存し、native alignedと元fallback unalignedを分けた別validatorで各readonly/writableのbit非変異を確認した。速度の再測定はしていない。

実GTSAM+Tokyo run3/seed20260710/400epochを、元のcluster_relin0/1×CPU/GPUの4prefixで実行した。各200、計800paired PF stepの保存state JSONL全行と軌跡NPZ全配列のdtype/shape/bytesが、凍結D234の同backend/同構成と一致した。今回は既定構成でも新処理の呼出しを確認した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含み、有限scalarと全数値配列bitsは一致する。Python scalar NaN payloadのcanonical化は既存codecの制約。source/input/external tc/3拡張/数値DLLのhashを前後照合した。

cfg1は保存cfg全項目が対照と一致。cfg0は元の明示default0を指定し、過去cfg1の保存cfgからcluster_relinだけを0へ戻した全項目とも照合した。既存top2/min3/fraction0.8等は同じ。過去/今回の単発実走時間は速度基準に使わず、実走は精度確認に限る。1route/seedの2構成prefixであり、全route/multi-seed測位精度gateは未完了。

実走counter（時間内訳ではない）:

```json
{
  "r3_seed0_mass_cpu_c0": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12504,
    "extend_calls": 2110,
    "shrink_calls": 2110,
    "requested_moves": 206522,
    "unique_moves": 4363,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "native_canonical_native_calls": 50268,
    "native_canonical_fallback_calls": 0,
    "parent_score_native_calls": 0,
    "parent_score_fallback_calls": 0,
    "component_mass_native_calls": 196,
    "component_mass_fallback_calls": 0,
    "component_mass_native_particles": 12544,
    "component_mass_native_pairs": 7906
  },
  "r3_seed0_mass_gpu_c0": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12504,
    "extend_calls": 2110,
    "shrink_calls": 2110,
    "requested_moves": 206522,
    "unique_moves": 4363,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "native_canonical_native_calls": 50268,
    "native_canonical_fallback_calls": 0,
    "parent_score_native_calls": 0,
    "parent_score_fallback_calls": 0,
    "component_mass_native_calls": 196,
    "component_mass_fallback_calls": 0,
    "component_mass_native_particles": 12544,
    "component_mass_native_pairs": 7906
  },
  "r3_seed0_mass_cpu_c1": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12508,
    "extend_calls": 2007,
    "shrink_calls": 2007,
    "requested_moves": 206341,
    "unique_moves": 4260,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "native_canonical_native_calls": 54915,
    "native_canonical_fallback_calls": 0,
    "parent_score_native_calls": 8382,
    "parent_score_fallback_calls": 0,
    "component_mass_native_calls": 196,
    "component_mass_fallback_calls": 0,
    "component_mass_native_particles": 12544,
    "component_mass_native_pairs": 7816
  },
  "r3_seed0_mass_gpu_c1": {
    "skipped_steps": 3,
    "valid_steps": 197,
    "particles": 12608,
    "held_particles": 12508,
    "extend_calls": 2007,
    "shrink_calls": 2007,
    "requested_moves": 206341,
    "unique_moves": 4260,
    "existing_cache_hits": 197,
    "scalar_fallbacks": 0,
    "native_canonical_native_calls": 54915,
    "native_canonical_fallback_calls": 0,
    "parent_score_native_calls": 8382,
    "parent_score_fallback_calls": 0,
    "component_mass_native_calls": 196,
    "component_mass_fallback_calls": 0,
    "component_mass_native_particles": 12544,
    "component_mass_native_pairs": 7816
  }
}
```

## メモリ・適用範囲

C++はN×nb×8bytesの一時DD vectorを持ち、そのcallで解放する。合成条件の最大は128×64×8=65,536bytes。元DD配列の数値payloadと同じ要素数で、旧N個のNumPy配列objectを省くが、Python/NumPy heap peakを実測したとは扱わない。入力Hlist、buffer export、PyLong減算結果、返却dict/数値objectは別の一時領域。保持H/DD/key/mass cacheやnative永続work buffer、CUDA allocationを追加しない。GILを保持し、RAIIでbuffer/参照を解放する。CPU ASanや任意Python版への全面的な保証はしない。

既存score workspaceはcapacity512、device/host各1,143,352bytes/context。benchmarkの2contextは各合計2,286,704bytes。driver/context/JIT/stack等は別。36製品sources、5installed extensions、PF build PE、D226 DLL、D233/D234拡張、過去30reports/14archives（1,630files）を維持する。条件付きで有効なCPU側最適化として残し、製品・凍結モデル・既定dispatchは維持する。GPU/全graph/全routeの採用判断は別gateとする。

再現: [build/API](../benchmarks/probes/rbpf_component_mass/README.md)、[完全step](../benchmarks/bench_rbpf_component_mass.py)、[実走](../benchmarks/bench_rbpf_component_mass_field.py)、[全結果](../benchmarks/rbpf_component_mass_20261016.json)、[完了照合](../benchmarks/rbpf_component_mass_completion_check_20261016.json)。
