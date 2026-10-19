# RBPFのhold解除用質量を既存native kernelで集計する (D236, 2026-10-16)

**D235の最適化CPUに対して、feedback有効な完全PF-stepの2/6条件が14.70〜19.28%短縮し、事前gateを通過した。** 64粒子/6初期仮説/64行/4groupで19.279568740851804%、128粒子/32初期仮説/64行/4groupで14.696727254544085%、双方5/5round改善。既存holdを事前設定した同じworkloadの新旧を比較した。別にfeedback無効controlを1条件残した。GPU対最適化後CPUの主screenは0/7。製品・モデル・既定dispatchは維持する。

## 変更

FIX成分質量に続き、`_feedback`のhold解除passにも残っていた全粒子DD展開と一致質量を、**前回D235のC++拡張をそのまま再利用**して集計する。C++/CUDA sourceと3個のCPU拡張、D226数値DLLはbyte単位で同じで、新しいnative buildはない。

置換するのは`dd_all`生成から`held_pairs/group_held/group_bad`を構成する5statementだけ。元のsupport-collapse/DDPR-release preludeは1回だけ実行した後、その時点の粒子とcommittedを読む。前段のFIX質量とのDD/mass cache共有はしない。preludeのgraph callbackが粒子やholdを変更しても古いDDを使わない。下流の厳密な`mass < fb_rel_mass`、参照衛星を疑うmajority規則、bad streak、unseen処理、解除順、commit条件とgraph注入順は元codeのまま。置換を戻して元ASTと一致することをassertする。

native適格域では、元の`int(np.rint(vr-vj))`を同じpair順で1回ずつ評価し、target vectorへ置く。元と同じbuiltin floatの有限holdに限り、絶対値1e100以下をprefetch安全域とする。これはモデルのhold制限ではなく、それ以外を元blockへ戻すdispatch条件である。この域では差・rint・整数→double round-tripが有限かつ同じ整数を表す。丸め規則を変えず、int64へ詰めない。

D235 kernelは全粒子・全pairを元順で展開する。任意精度整数を引いてから元のFP64へ1回変換し、非target pairの変換例外も飛ばさない。成分ごとの和を+0.0から粒子順にFP64加算し、tree reduction/reassociation/FP32を使わない。有限doubleの切り捨て比較は元のPython int比較と同じ。元の全粒子重み分母を保ち、空/未定義DDが一致質量に加算されない意味を維持する。

元のplain Particle/EpochProblem/RbpfConfig、builtin dict/int/float、通常のNumPy binding、同じweights layoutを適格域にする。custom mapping/rounding binding、NumPy scalar hold、非有限/大きなhold、unsupported buffers/weightsは元blockを1回実行する。preludeを繰り返さない。対象の粒子・holdはblock中に外部から変更されない前提で、同一filterの並行実行は既存caller同様に扱わない。strict NumPy policyは従来どおり元step全体へ戻し、部分state/RNGを保つ。

全粒子・候補・prior・閾値・FP64/2IRLS・重み・RNGを維持する。実runtime/moduleを上書きせず、per-call viewのfeedback cloneだけを使う。D231 warp/D232 DD packは併用しない。

## 完全PF-stepの測定

6条件はfeedback有効・元fixtureの最初の非空Hから一貫したSD holdを事前設定する。両variantに同じcfg/ordered committed/navを渡し、候補/粒子/閾値は減らさない。追加の1条件はfeedback無効control。D235 CPU/新CPU/新GPU/D235 GPU/D235 CPU A/Aを直列回転し、7条件×5round×3group×5mode=525timed calls、35warm snapshotを保存した。

各試行でRTKをdeepcopyした後、元fixtureと同じbodyのinject/release callbackをコピー先へ結び直す。Python関数のclosureはdeepcopyで再束縛されないため、試行間の状態漏れを避ける処置である。callback bodyのASTが元fixtureと一致することを計測前にassertし、全warm/timed callでRTK templateのdigest不変も確認した。restore/binding/hash/warm/検証は時間外。guard、hold target作成、DD展開、質量・group集計、既存move/pack/transfers/conditionals/FIX/feedback/resamplingはtimerへ含める。extract/time/satellite/integer-searchとRTK graph callbacksは外部入力fixtureで、GTSAM graph solve/実注入costの速度を測ったとは扱わない。

自processの1許可CPU/1BLAS thread、通常priority、host/GPU非占有。paired group短縮率のround内中央値→5round中央値で評価し、独立p50同士の比とは異なる。CPUは同backend旧CPU比10%以上、4/5round以上改善、A/A中央値abs5%未満を要求した。GPU主screenは旧GPUと新CPUの両方へ同じ改善基準を要求する。丸め前で判定し、全負値・失敗を保持して再測定しない。feedback無効controlの時間変化を新処理の利得とは解釈しない。

| particles/initial_unique/rows/groups | feedback | held targets/step | CPU p50 ms D235→新 | CPU同士paired短縮 | GPU同士paired短縮 | GPU対新CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|:---|---:|---:|---:|---:|---:|---:|:---|:---|
| 64/6/64/4 | on | 32 | 14.005 → 11.808 | +19.280% | +19.707% | +0.722% | -1.032% | PASS | FAIL |
| 64/32/64/4 | on | 32 | 25.606 → 21.992 | +15.153% | +15.280% | +1.758% | +11.164% | FAIL | FAIL |
| 64/64/64/4 | on | 32 | 38.878 → 37.092 | +7.828% | +4.174% | -17.045% | +2.471% | FAIL | FAIL |
| 128/32/64/4 | on | 32 | 34.716 → 31.773 | +14.697% | +14.665% | -16.793% | +4.165% | PASS | FAIL |
| 64/6/128/8 | on | 64 | 26.138 → 21.066 | +16.683% | +17.725% | -18.475% | -9.278% | FAIL | FAIL |
| 128/128/128/8 | on | 64 | 404.910 → 380.737 | +4.504% | +3.420% | -2.776% | +4.581% | FAIL | FAIL |
| 64/6/64/4 | off control | 0 | 11.086 → 11.334 | -2.280% | -2.906% | -28.519% | -4.376% | FAIL | FAIL |

CPU主screen2/7（有効条件2/6）、旧GPU同士の副screen2/7、GPU主screen0/7。A/A失敗2/7を残した。64粒子/32初期仮説の15.15%、64粒子/6初期仮説/128行の16.68%もA/A基準外なので採用範囲に含めない。高多様性4.50%と64粒子/64初期仮説7.83%は10%未達。無効controlのCPU−2.28%も保持する。個別roundの揺れを外部負荷等へ断定しない。

今回のfeedback有効/事前hold workloadはD235の初回screenとは異なる。D235の短縮率を今回の値へ足したり、累積の実FGO速度比を作らない。新旧D235/D236を同じ今回workloadで測った追加効果のみを扱う。

## 正確性・実走

79tests PASS。3seed×4サイズ×25case=300統計比較、missing hold、round-to-evenと大整数、元の逐次和がtree sumと異なる反例、解除閾値前後1ULP、custom型/mapping/rounding、unaligned/invalid weights、非target overflowの例外順、10,000call参照数・更新反映を確認した。support/DDPR preludeのcallbackが全粒子をclearする場合も、preludeの1回実行と新しいDDの使用を比較した。3backend×3seed×4構成×4epochで全保存state/位置/P/RNGとgraphイベントが一致した。warning3は既知config1と元strict-policy underflow2。全525timed state hashと35warm snapshotもD235の同backendに一致し、新しい数値許容差は追加しない。

CUDA memcheckは18targeted tests PASS、config warning1、error/leak0。CPP/CPU拡張/数値DLL自体はD235と同じで、新しいGPU kernelやCPU ASan/general-version検証ではない。

実GTSAM+Tokyo run3/seed20260710/400epochをcluster_relin0/1×CPU/GPUの4prefixで実行した。各200、計800paired PF stepで、凍結D235の同backend/同構成と保存state JSONL全行・軌跡NPZ全配列のdtype/shape/bytesが一致した。全cfg項目・source/input/external tc/拡張/数値DLLのhashを照合する。既定構成でも新feedback native呼出しと実held targetが正で、fallback0を確認した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む有限scalarと全数値配列bitsは一致し、Python scalar NaN payloadのcanonical化は既存codecの制約として残る。

実走counter（速度比や時間内訳ではない）:

```json
{
  "r3_seed0_feedback_cpu_c0": {
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
    "component_mass_native_pairs": 7906,
    "feedback_mass_native_calls": 197,
    "feedback_mass_fallback_calls": 0,
    "feedback_mass_native_particles": 12608,
    "feedback_mass_native_pairs": 7933,
    "feedback_mass_held_targets": 4814
  },
  "r3_seed0_feedback_gpu_c0": {
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
    "component_mass_native_pairs": 7906,
    "feedback_mass_native_calls": 197,
    "feedback_mass_fallback_calls": 0,
    "feedback_mass_native_particles": 12608,
    "feedback_mass_native_pairs": 7933,
    "feedback_mass_held_targets": 4814
  },
  "r3_seed0_feedback_cpu_c1": {
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
    "component_mass_native_pairs": 7816,
    "feedback_mass_native_calls": 197,
    "feedback_mass_fallback_calls": 0,
    "feedback_mass_native_particles": 12608,
    "feedback_mass_native_pairs": 7843,
    "feedback_mass_held_targets": 4796
  },
  "r3_seed0_feedback_gpu_c1": {
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
    "component_mass_native_pairs": 7816,
    "feedback_mass_native_calls": 197,
    "feedback_mass_fallback_calls": 0,
    "feedback_mass_native_particles": 12608,
    "feedback_mass_native_pairs": 7843,
    "feedback_mass_held_targets": 4796
  }
}
```

これらは1route/seedの2構成prefixである。凍結対照や今回の単発実走時間を速度の基準に使わず、全route/multi-seed測位精度gateと全graph速度gateは未完了。

## メモリ・再現

既存D235の一時DD payloadはN×nb×8bytes（合成最大65,536bytes）。前段のcomponent-mass vectorは呼出し内で解放済みで、今回のfeedback vectorと重ねて保持しない。追加はcall中のtarget vector/record list/group objects等で、H/DD/key/mass cache・native永続work buffer・CUDA allocationは増やさない。heap peakは実測せず、Python/NumPy/allocator/driver/JIT/stackは別。既存score workspaceはcapacity512、device/host各1,143,352bytes/context、benchmarkの2contextは各合計2,286,704bytes。

36製品sources、5installed extensions、PF build PE、D226 DLL、D233/D234/D235拡張、過去32reports/15archives（1,773files）を維持する。製品・凍結モデル・既定dispatchは維持し、条件付きの明示実験版として保存する。

```python
from rbpf_likelihood_batch import BatchScorer
from rbpf_feedback_mass import NativeFeedbackMassStep

with BatchScorer(explicit_unchanged_D226_math_dll) as scorer:
    runtime.step = NativeFeedbackMassStep(runtime,scorer,'native_cpu',
        extension=explicit_D233_key_pyd,
        parent_extension=explicit_D234_parent_pyd,
        mass_extension=explicit_unchanged_D235_mass_pyd)
    # scorer/workspaceの寿命をcaller使用中は保つ。
```

`benchmarks`をimport pathへ追加する。新buildは不要で、[D235 build/API](../benchmarks/probes/rbpf_component_mass/README.md)のbinaryを使う。testsは`GNSS_GPU_RBPF_PROBE_DLL`、`GNSS_GPU_RBPF_CANONICAL_EXT`、`GNSS_GPU_RBPF_PARENT_EXT`、`GNSS_GPU_RBPF_MASS_EXT`を明示して`python -m pytest tests/test_rbpf_feedback_mass.py -q`。
[完全step](../benchmarks/bench_rbpf_feedback_mass.py)は`--dll --extension --build-record --canonical-extension --parent-extension --output`を受け、extension/build-recordはD235の不変massを指定する。新しい出力先・直列GPU jobを使い、失敗を保持して採用目的の再測定をしない。
[実走helper](../benchmarks/bench_rbpf_feedback_mass_field.py)、[全結果](../benchmarks/rbpf_feedback_mass_20261016.json)、[完了照合](../benchmarks/rbpf_feedback_mass_completion_check_20261016.json)。
