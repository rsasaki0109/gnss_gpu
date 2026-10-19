# RBPFの任意精度整数キーをC++で生成する (D233, 2026-10-16)

**PF全体のCPU比較で2/6条件が10%gateを満たした。条件付きの実験版として保存し、既定動作は維持する。** 64/128粒子・32初期仮説・64行・4groupで、D229の最適化済みCPUに対して10.27%/13.21%短縮し、両方5/5round改善した。全6条件への一般化はできず、GPU対最適化後CPUの主screenは0/6だった。graph solveを含む実FGO全体の高速化とは扱わない。

## 変更

凍結した`_canon_basin_id`と同じキーを、明示指定するCPUのCPython拡張で作る。全groupをsortし、各groupの最小satelliteの整数をbaseとして、元と同じ`z-base`のpairをsatellite順のtupleへまとめる。Pythonの任意精度整数の減算とbuiltin tuple/int比較を使い、固定幅整数やfloatへ変換しない。2**16000の共通gauge、2**4096の差でも元と同じkey/hashを保つ。group/satelliteの挿入順に依存せず、空groupも元と同様に省く。

全入力が正確なbuiltin dict/tuple/intであることを算術より前に確認する。異なるmapping、数値subclass、NumPy整数、bool/float等は、元のPython関数を1回実行する。`sorted/min/tuple`の有効bindingを毎call照合し、custom helperや変更されたfunction bodyも元へ戻す。元のliteral codeは同じSHAのmodule compile形とisolated AST形から導き、別のモデルを黙って適用しない。
拡張関数はGILを保持し、返却key以外に入力/keyを保持しない。追加のcanonical/DD cacheはない。`NativeCanonicalStep`は元methodのcode objectと更新したglobalsをper-call viewで実行し、canonical関数だけを差し替える。D229のmove reuse/staging memo、全候補/prior、FP64/2IRLS、重み/choice/commit、RNG、conditional mean、FIX閾値とgraphを変更しない。strict NumPy policyは元step全体へ戻して部分error stateも維持する。実runtimeのmethod/moduleは一時的にも上書きしない。

D226の数値/転送DLLはbyte単位で同じものを使う。D231 warpとD232 DD入力展開は併用せず、独立した変更の効果を測る。新しいCUDA数値kernelやGPU機械語の改善ではなく、GPU入力前を含むPython側のキー構築を削減する。

## 完全PF-stepの比較

6条件×5round×3group×5mode=450timed calls。D229 CPU/native-key CPU/native-key GPU/D229 GPU/D229 CPU A/Aを直列回転した。キー生成、validation、move構築、全pack/transfer/readback、conditional mean、mass/FIX、resampling/RNGを含む。復元/warm/setup/hash/検証は時間外。graph extraction/time/satellite/integer-search入力はfixtureであり、graph solveは含まない。
自processの1許可CPU・1BLAS thread、通常priority、host/GPU非占有。paired group内比のround内中央値→5round中央値で判定する。独立p50同士の比とは異なる。CPUは同backend旧CPU比10%以上、4/5round以上改善、A/A中央値abs5%未満を要求した。GPU主screenは旧GPUと新CPUの両方に同じ短縮基準を要求する。全raw/負値/失敗を保持して再測定しない。gateは丸め前の値を使う。

| particles/initial_unique/rows/groups | CPU p50 ms 旧→新 | CPU同士paired短縮 | GPU同士paired短縮 | GPU対新CPU短縮 | CPU A/A短縮 | CPU主screen | GPU主screen |
|:---|---:|---:|---:|---:|---:|:---|:---|
| 64/6/64/4 | 27.666 → 25.282 | -12.369% | -1.092% | +3.411% | +9.825% | FAIL | FAIL |
| 64/32/64/4 | 52.897 → 50.528 | +10.267% | -1.461% | -0.794% | +2.973% | PASS | FAIL |
| 64/64/64/4 | 67.112 → 58.940 | +2.408% | +16.621% | -0.053% | +4.658% | FAIL | FAIL |
| 128/32/64/4 | 73.755 → 68.038 | +13.209% | +4.685% | -3.193% | -0.469% | PASS | FAIL |
| 64/6/128/8 | 63.808 → 55.702 | +12.831% | +9.996% | -6.938% | -8.887% | FAIL | FAIL |
| 128/128/128/8 | 1851.405 → 2065.412 | -3.080% | -6.258% | -1.100% | -2.478% | FAIL | FAIL |

CPU主screenは2/6、旧GPU同士の副screenは1/6、GPU主screenは0/6。2条件のA/A失敗を残した。旧GPU比16.62%の64粒子/64初期仮説も、新CPU比は−0.05%なのでGPU主screenを満たさない。128粒子/128仮説の高多様性条件はCPUもslowdownで、採用範囲へ含めない。大きな個別roundやp50だけで判断を置き換えず、view/guard/allocator等へ原因を断定しない。

## 正確性・寿命・実走

65tests PASS。1,200ランダム整数mapと各gauge/order変換、key値/type/hash、巨大整数の区別、空group、custom型/helper/例外、10,000callの参照数とfresh key、2種compile形を比較した。PFは3backend×3seed×3構成×4epochで全保存state・位置/P・RNGが一致し、全450timed state hashと30warm snapshotも同じbackendのD229と一致した。新しい数値許容差を加えない。設定warning1と元のunderflow warning2はstrict-policy比較の元処理で発生した。
CUDA memcheckは14targeted tests PASS、設定warning1、error/leak0。元のGPU数値DLLは不変で、新たなCUDA race/sync構造はない。C++は所有参照をRAIIで解放し、入力参照はcall中だけ借用する。CPU ASanによる全面的な証明は行っていない。

使用したCPython3.12.10ではallocation時のGCをeval-breakerへ予約する。[公式実装](https://github.com/python/cpython/blob/v3.12.10/Modules/gcmodule.c#L2133-L2178)も確認した。builtin整数/tupleだけのnative経路はPython callbackを実行せず、GILも解放しない。別child processでGC callbackが入力をclearする100回のstressを行い、100callbackと全native返却の完全性を確認した。これは当該runtimeでの補助検証で、別Python版の全面的な保証ではない。

実GTSAM+Tokyo run3/seed20260710/400epochをCPU/GPU各1回実行した。各200、計400PF stepで、凍結D229の同backendに対し全保存stateと軌跡NPZ全配列が一致した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む。配列dtype/shape/bytesと有限scalar bitsを照合し、従来codecによるPython scalar NaN payloadのcanonical化は除く。過去の時間を現在の速度対照へ使わず、実走は精度のみの検証とする。1route/seedのprefixであり、全route/multi-seed測位精度gateは未完了。

実走counter（時間の内訳や速度比ではない）:

```json
{
  "r3_seed0_key_cpu": {
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
    "canonical_native_calls": 50268,
    "canonical_fallback_calls": 0
  },
  "r3_seed0_key_gpu": {
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
    "canonical_native_calls": 50268,
    "canonical_fallback_calls": 0
  }
}
```

## メモリ・適用範囲

新CPU拡張はPythonのkey/一時list/tuple/intをそのcallだけ作り、永続native work bufferやCUDA allocationを追加しない。engineは明示module/元関数/code参照と整数counterを所有するが、H/keyはcacheしない。元と同じtuple keyのpayloadサイズは整数のbit長に依存するため、固定byte数やheap peakを実測したとは扱わない。
既存score workspaceはcapacity512でdevice/host各1,143,352bytes/contextのまま。benchmarkの2contextは各合計2,286,704bytesで、driver/context/JIT/stack/NumPy/Python heapは別。
36製品sources、5installed extensions、PF build PE、D226 DLL、過去26reports/12archives（1,368files）を維持した。凍結RBPF/FGO sourceや既定dispatchに適用していない。CPU側の一部条件に有効な選択肢として残し、GPU/全route/全graphへの採用判断は別gateとする。

再現: [build/API](../benchmarks/probes/rbpf_canonical/README.md)、[完全step](../benchmarks/bench_rbpf_native_canonical.py)、[実走wrapper](../benchmarks/bench_rbpf_native_canonical_field.py)、[全結果](../benchmarks/rbpf_native_canonical_20261016.json)、[完了照合](../benchmarks/rbpf_native_canonical_completion_check_20261016.json)。
