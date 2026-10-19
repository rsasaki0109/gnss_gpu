# D298: 反射グループを必要時だけ準備する (2026-10-19)

**大バッチ2条件は両controlとA/Aを通過。全4条件は小入力のA/A変動とround別非退行条件で不合格のため、製品へは未採用。** warm元比は25.117% / 19.554%短縮。cold元比は13.655%短縮 / 2.821%増加で、後者も事前の5%非退行範囲内だった。最大条件で初回時間も短縮できたとは主張しない。

## 変更と精度

D297はconstructorで全ordered三角形のv0をCPUへ24*N bytesコピーして保持していた。D298はこのコピーをなくし、LOSのみ、小入力、無効化状態ではgroup準備を実行しない。最初の対象reflectionで所有済みGPU meshからv0をpackし、元normalとともに読み戻す。pack kernelは64bit load/storeと整数命令だけで、FP演算なし。元13 GPU関数（grouped候補を含む）はSASS/encodingが全て一致した。

GPU unit normalとv0の全6 uint64 key、hash衝突時の完全比較、最大8枚への分割、元ordinal順の全候補、交差・遅延・同点選択を維持する。浮動小数点精度、幾何tolerance、候補数の変更はない。callerが元mesh/nodesを書き換えた後も所有済みsnapshotを使う。default offから後で有効化でき、rejected geometryも準備は一度だけ。符号付きゼロをpack後も区別する。

GPU一時scratchとhost v0 readbackは12k/120k枚で各288000 / 2880000bytes。GPU scratchはCPU grouping/CSR allocationより前に解放する。API後の`allocated_bytes`はlive所有量で、この解放済みscratchを含まない。記録したpeakはpost-call live + scratchの保守的上限で、実測peakとは呼ばない。CSRは72004 / 720004bytes、ray依存FP cacheは0。追加pack起動・D2H・解放も全てcold時間に含めた。

## 検証と事前条件

49 unique tests PASS（projection20、native26、sparse GPU3）。memcheck/racecheck/synccheckは同じsparse3件でerror/leak/hazard0、全観測processのexit0。late enable、LOS/smallからbulk、caller mutation、signed zero、rejected grouping一度だけ、legacy90形状、scale/ECEF、chunk、fallback、closeを確認した。元CPU builderと全reflection関数もliteral。Ruff F checksはPASS。

D297保存済みの同一実装対照の変動を受け、**D298初回実行前**にcoldを8反復へ増やし、各反復で全5 modesを交互に回し、順序を回転・反転させた。warmは2反復、両phaseとも5round×3group。元実装 / 同PE off / on / 元A/A / off A/A、固定4条件。bulk warmは両control比10%以上かつ4/5roundで正、small/coldは両control比-5%以上かつ4/5roundで-5%以上。各A/Aは絶対差中央値5%未満かつ4/5roundで絶対差5%未満。閾値はD297と同じで、全4条件合格後にpublic/urbanへ進む条件も維持した。

12setup・60warm・600timed・2400cold、**全3072呼出し**のLOS・point・delay dtype/shape/bitsが元実装と一致。60warm全状態、4input NPZ、全raw ns/順序/負roundを保持。独立auditorを速度実行前に固定し、全grid/input/hash/統計を再計算した。終了済みD296/D297を再測定していない。cold反復増加と本実装変更は別の要因なので、D297との差分を因果的な速度改善とは解釈しない。

| 条件 | warm元比 | warm同PE off比 | cold元比 | cold同PE off比 | 全control合格 |
|---|---:|---:|---:|---:|---:|
| 12000三角形 / 128epoch×12sat | +25.117% | +25.475% | +13.655% | +14.516% | True |
| 120000三角形 / 32epoch×32sat | +19.554% | +19.787% | -2.821% | -0.716% | True |
| 12000三角形 / 1epoch×12sat | +3.441% | +1.835% | -3.406% | -1.065% | False |
| 4096三角形 / 16epoch×8sat | +0.870% | +0.963% | +2.244% | +1.093% | False |

12000枚smallではwarm A/Aが両方3/5round、cold A/Aが3/5・2/5roundしか5%以内に入らない。cold元比も5%以内が3/5roundで非退行不合格。4096枚smallではwarm合格、coldのA/Aは2/5・3/5roundで不合格。中央値だけで合格へ読み替えず、8反復でも対照が安定しないという結果を保存した。smallでgroup準備は0であり、この変動からgroup実装の確定した退行も断定しない。全4条件gateはFalseなのでpublic/urban stage、product/defaultへの反映は行っていない。

計時はnative同期APIのvalidation/H2D/LOS/reflection/sync/D2H/返却配列を含む。coldはconstructor/selector/初回trace/closeも含む。CPU mesh/BVH構築とhash/保存は計時外。CPU1/BLAS1/通常GC/通常priority・非専有、直列。計時中profiler/sanitizer/debugger/周期stack dumpなし。測位route/RMS、PF/FGO全体速度、GPU化率・使用率は新規に測っていない。

158旧報告・78旧archive・21094旧files、D295製品36source・既存5PE/PFbuild/installed BVHを保持する。D296/D297の不合格結果は変更していない。

[結果](../benchmarks/bvh_lazy_groups_20261019.json)、[完了照合](../benchmarks/bvh_lazy_groups_completion_check_20261019.json)、[生成手順](../benchmarks/probes/bvh_lazy_groups/README.md)、[全呼出し計測](../benchmarks/bench_bvh_lazy_groups.py)、[独立照合](../benchmarks/audit_bvh_lazy_groups.py)、[native tests](../tests/test_bvh_lazy_groups.py)。
