# RBPFのDD整数入力をC++で展開する (D232, 2026-10-16)

**大きい尤度バッチの改善を実験版として保存し、PF既定適用は見送る。** Pythonの仮説辞書からDD配列を作るループをC++/CPython C APIへ移した。64行・64仮説の完全尤度バッチで旧GPU比25.14%、最適化後C++比24.22%短縮した。一方、PF全ステップのCPU/GPU主screenはともに0/6合格だった。準備・転送込みのバッチ利得を、全PFやFGOの速度利得として扱わない。

## 精度を維持する入力展開

独立したCPU DLLは、正確なbuiltin list/dict/tuple/intだけを検証してから出力へ書く。各DDはPythonの任意精度整数で`zr-zj`を求め、最後に一度だけFP64へ丸める。共通gaugeが2**180でも、int64へ狭めたり各整数を先にfloat化したりしない。欠損値は元と同じNaNのbits、pair/hypothesis順も同じ。全shape/type検査を先に行い、未対応mapping/数値型/subclassは元のPython処理へ戻す。custom DD hookも元の呼出し回数・値・例外を維持する。

`ctypes.PyDLL`でGILを保持する。C APIの一時intは各pairで解放し、参照やDDを保持しない。毎回freshな配列を返し、入力やRNGを変更しない。容量超過は全件を元へ戻し、候補を切り捨てない。尤度のFP64式、2IRLS、LU/Huber、事前共分散、閾値、全候補とprior、重み、choice/commit、測位モデルを変更しない。D226の数値/転送DLLは同じファイルで、D231 warp実験を併用しない。

`PackedDdScorer`は凍結した既存scoreのDD生成式1か所だけを置換する。`PackedDdStep`は元のmethod codeとglobalsをper-call viewで実行し、他のDD消費箇所にも同じ変換を使う。追加のDD/canonical cacheはない。D229のmove reuse/staging key memoをそのまま使い、strict NumPy policyでは元のstep全体へ戻して例外時の部分state/RNGも維持する。実runtimeのmethodを一時的に上書きしない。

## 測定結果

尤度9条件×5round×3group×5mode=675timed calls。従来C++/従来GPU/native DD C++/native DD GPU/従来GPU A/Aを直列回転した。dedup・入力検証/作成・cluster lookup・pageable upload・計算・同期readback/scatterを含み、workspace/setup/warm/hash/検証は時間外。

PF全体は6条件×5round×3group×5mode=450timed calls。D229 CPU/native DD CPU/native DD GPU/D229 GPU/D229 CPU A/Aを直列回転した。実際の元stepの維持処理、move生成/score、conditional mean、mass/FIX、resampling/RNGを含むが、graph/extraction/time/satellite/integer-search入力はfixture。これは実FGO graph solveの測定ではない。

自processの1許可CPU・1BLAS thread、通常priority、host/GPU非占有。同一group内の短縮率をround内中央値→5round中央値で集計し、10%以上・4/5round以上改善・対照A/A中央値abs5%未満を要求した。GPU主screenは、同じbackendの従来GPUと最適化後C++の両方に同基準を要求する。再測定せず、全条件・失敗・rawを保持した。

| 尤度 rows/groups/unique | C++同士短縮 | GPU同士短縮 | GPU対最適化C++短縮 | A/A短縮 | C++ screen | GPU主screen |
|:---|---:|---:|---:|---:|:---|:---|
| 64/4/4 | +4.15% | +1.66% | -42.21% | +1.24% | FAIL | FAIL |
| 64/4/21 | +9.57% | +6.52% | +4.51% | -5.46% | FAIL | FAIL |
| 64/4/32 | +17.23% | +26.55% | +21.47% | +0.78% | PASS | PASS |
| 64/4/64 | +21.37% | +25.14% | +24.22% | -1.20% | PASS | PASS |
| 64/4/128 | +8.01% | +23.60% | +24.65% | +2.60% | FAIL | PASS |
| 64/4/512 | +20.57% | +30.25% | +32.15% | -5.86% | FAIL | FAIL |
| 128/8/21 | +8.68% | +8.78% | -1.73% | -2.00% | FAIL | FAIL |
| 128/8/64 | +6.35% | +11.30% | +13.57% | -2.63% | FAIL | FAIL |
| 128/8/128 | +22.72% | +2.06% | +10.18% | +8.17% | FAIL | FAIL |

尤度のC++同士は2/9、GPU同士は4/9、GPU主screenは3/9合格。32/64/128仮説のGPUバッチで主screenを満たした。64仮説は旧GPU比25.14%・最適化後C++比24.22%、ともに5/5round改善。小バッチの利得は限定的で、21仮説の64行条件はA/Aも不合格だった。512仮説の大きな値もA/A不合格を覆さない。各条件のp50と全roundはJSONへ残した。

| PF particles/initial_unique/rows/groups | CPU同士短縮 | GPU同士短縮 | GPU対最適化CPU短縮 | A/A短縮 | CPU/GPU主screen |
|:---|---:|---:|---:|---:|:---|
| 64/6/64/4 | +1.24% | -7.13% | +3.03% | +11.92% | FAIL |
| 64/32/64/4 | +3.72% | +6.80% | -7.37% | +8.41% | FAIL |
| 64/64/64/4 | +12.43% | -1.51% | -6.02% | +5.78% | FAIL |
| 128/32/64/4 | -5.49% | +5.81% | +5.03% | +0.77% | FAIL |
| 64/6/128/8 | -3.57% | +6.01% | +4.67% | +4.63% | FAIL |
| 128/128/128/8 | +2.17% | +4.75% | +6.36% | +1.95% | FAIL |

PF全体はCPU/GPUとも0/6合格、3条件はA/A不合格。64粒子・64初期仮説のCPU短縮12.43%/4of5も、A/A5.78%のため不採用。ほかの小さな値やslowdownを保持し、全体で効果が確立したとは扱わない。viewや検証処理に原因を断定しない。

## 実走への照合と修正の扱い

最初の実走は初期化時の厳密guardで停止し、PF stepを実行していない。同じ凍結ソースでも、CPythonがmodule全体をcompileした場合とisolated AST classをcompileした場合で、`LOAD_GLOBAL/LOAD_ATTR`のcall命令列が異なった。変更された数値モデルを受け入れるための緩和ではない。
測定済みv1の全source/helper/resultを凍結したまま、実走対応variant `rbpf_dd_pack_runtime.py`を分離した。同じSHAの元ソース全体を実行せずcompileし、その正確なDD codeとisolated AST codeだけを照合する。変更bodyは拒否する。guard導出/照合とdocstring以外のASTはv1と同一で、pack/scorer/view/step/数値は不変。variantの時間を再測定せず、v1の速度判定もvariantへ移さない。最初の拒否log/plan/sourceと全失敗を残す。

実走対応版129tests PASS。元の127casesを再使用し、2種類のcompile形の全score/diagnostic比較と変更body拒否、imported compile形の完全PF state/RNG比較を追加した。129+127をunique256testsとは数えない。任意精度/rounding/欠損/NaN payload/order、unsupported/custom hooks、overflow/refcount、capacity/canary/late-invalid入力、54尤度条件のCPU/GPU全bits、3backend×3seed×3構成×4epochの全PF state/RNGを確認。設定warning1と元のunderflow warning2は既存処理の比較で発生した。最初のtest fixture誤り2件も元log/sourceを保持し、許容差を広げていない。
memcheckは測定版16、実走対応版18targeted cases PASS、各設定warning1、CUDA error/leak0。これらはunit testsのsubset。GPU kernel/数値DLLは変更しておらず、新しいGPU機械語やrace/sync性能を主張しない。CPU canary/容量/refcount検査はCPU ASanによる全面的な証明ではない。

Tokyo run3/seed20260710/400epochをCPU/GPUで各1回実行した。各200、合計400PF stepで、凍結D229の同じbackendと全保存state・軌跡NPZの全配列が一致した。位置/P/FIX/hold/粒子/重み/RNG/feedbackを含む。配列dtype/shape/bytesと有限scalar bitsを照合し、Python scalar NaN payloadの従来codec canonical化は除く。凍結した過去の時間を現時点の速度対照にしない。1route/seedのprefixであり、全route/multi-seed測位精度gateは未完了。

実走のDD利用counter（call数であって速度利得ではない）:

```json
{
  "r3_seed0_packed_cpu": {
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
    "DD_pack_native_calls": 30794,
    "DD_pack_native_hypotheses": 34763,
    "DD_pack_fallback_calls": 0
  },
  "r3_seed0_packed_gpu": {
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
    "DD_pack_native_calls": 30794,
    "DD_pack_native_hypotheses": 34763,
    "DD_pack_fallback_calls": 0
  }
}
```

## メモリと再現

新C++ DLLはCUDA allocationも永続work bufferも持たない。新たな数値配列payloadはbulk最大512×128×8=524,288bytes、single最大128×8=1,024bytes。CPythonの一時整数、型検証、static guard code objectsなどのPython heapやpeak実測は含めない。元の一仮説ずつの配列list→contiguous copyを、1枚の出力配列へ置き換える。
旧scoring workspaceはcapacity512でdevice/host各1,143,352bytes/contextのまま。benchmarkの2scorerはそれぞれ合計2,286,704bytesで、driver/context/JIT/stack/NumPy入力は別。workspace setupを測定から除外した。

製品36sources、installed5extensions、PF build PE、D226 DLL、過去24reports/11archives（1,229files）は不変。製品や凍結RBPF/FGO source、既定dispatchに適用していない。

再現: [build/API](../benchmarks/probes/rbpf_dd_pack/README.md)、[完全バッチ](../benchmarks/bench_rbpf_dd_pack_likelihood.py)、[全step](../benchmarks/bench_rbpf_dd_pack_step.py)、[実走wrapper](../benchmarks/bench_rbpf_dd_pack_runtime_field.py)。[全結果](../benchmarks/rbpf_dd_pack_20261016.json)、[完了照合](../benchmarks/rbpf_dd_pack_completion_check_20261016.json)。
