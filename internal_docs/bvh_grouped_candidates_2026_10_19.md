# D297: 同一GPU threadで反射方向を再利用 (2026-10-19)

**大バッチ2条件でwarm/coldとも両controlとA/Aを通過した。4条件全体はsmall coldの不安定さ等により不合格で、製品へは未採用。** warmの元比短縮中央値は25.261% / 20.114%、coldは16.868% / 2.136%。native APIの指定条件の観測であり、public/urban、PF/FGO、測位処理全体の速度や精度の新保証ではない。

## 実装

直前D296は56bytes/group/rayの中間cacheを使い、最大128MiBの書出しと準備を要した。D297は別kernelであり、このcacheと準備launchをなくす。元v0のraw bitsと元GPU unit normalのraw bits、全6 uint64成分が一致する三角形だけをまとめ、1threadで元の鏡像点と正規化を一度計算してから、groupの全三角形を評価する。groupは最大8枚で、それを超える同じkeyは複数groupへ分ける。host側でCSR索引を作り、memberは元ordinal順。各反射点・遅延の計算式、全valid候補、(delay, ordinal)の同点選択を維持する。無効normalは元と同じく候補にならない。

6000/60000groups、12000/120000membersに対する追加GPU metadataは72004/720004bytes。ray依存のFP64中間cacheは0、元のnormal bufferと上限付きwinner scratch/ray chunksを使う。fresh setupの全owned cudaMallocは2059780/19152196bytes。coldにはv0snapshot、元GPU normalのD2H、CPUのfull-bit grouping/CSR、H2D、初回traceとcloseを含める。hash衝突は全key比較で解決し、CPU normal近似、幾何tolerance、候補間引きは使わない。

## 検証と計測

39unique tests PASS（projection18、native18、追加sparse GPU3）。元12GPU関数は全SASS/encodingが一致し、追加はgrouped candidate1kernel。元CPU builderもliteral。memcheck/racecheck/synccheckは同じsparse3件ずつでerror/leak/hazard0、全観測launcher/worker/processのexit0まで確認した。通常stack、raw signed zero/scale/ECEF/snapshot/fallback/groupmax8/chunk/close/元legacy corpus等を確認した。

元/同PE off/on/両A/A、固定4条件。warm/coldの両方で5round×3group×2repにし、12setup・60warm・600timed・600cold、全1272呼出しの返却dtype/shape/bitsが一致した。60warm全状態、全input NPZ、全raw ns/順序/負roundを保持する。coldの反復は本candidate初回計測前に固定し、同一実装対照も全く同じ反復を使う。D296の終了済み測定は取り直していない。独立auditorを速度実行前に固定し、全grid/input/状態hash/統計を再計算した。

各groupで2repの全呼出しnsを合計し、3group中央値→5round中央値で判定。bulk warmは両control比で中央値10%以上かつ4/5roundで正、small/coldは5%以内の非退行と4/5roundで5%以内。両A/Aは絶対中央値5%未満かつ4/5roundで絶対差5%未満を要求する。元validation/H2D/compute/sync/D2H/返却配列を含む。CPU mesh/BVH構築・hash/保存は計時外。CPU1/BLAS1/通常GC/priority・非専有環境で直列、計時中profiler/sanitizer/debugger/周期stack dumpはない。

| 条件 | warm元比 | warm同PE off比 | cold元比 | warm/cold全control合格 |
|---|---:|---:|---:|---:|
| 12000三角形 / 128epoch×12sat | +25.261% | +25.200% | +16.868% | True |
| 120000三角形 / 32epoch×32sat | +20.114% | +20.377% | +2.136% | True |
| 12000三角形 / 1epoch×12sat | +1.740% | +0.472% | -4.098% | False |
| 4096三角形 / 16epoch×8sat | -0.185% | -0.358% | +0.808% | False |

両bulkのwarm/cold controlは安定し、利得条件も満たす。一方small coldでは、12000三角形のA/A絶対中央値4.139/9.035%、4096三角形では7.186/7.462%となった。round単位の非退行条件も含めて全4条件gateはFalse。small coldの値を確定した速度退行と断定せず、不安定な対照下で合格にも読み替えない。事前条件どおりpublic/urban stageには進まず、default/source/installed PEを維持した。

## 初回失敗と再現

最初のprojectorは、古いepoch-cache resetを空文字へ置換した後、可逆proofがcount1を要求して失敗した。compile/GPU/数値実行前で、元source/logを残した。checked projectorはそのhost resetを固有commentへ置き換えて可逆性を確認する。GPU数学は同じで、元D295→D296全source undoとD297→D296全source undoを合成して証明する。正しい入口は[generate_checked.py](../benchmarks/probes/bvh_grouped_candidates/generate_checked.py)。初回generate.pyは失敗記録として維持する。[再現手順](../benchmarks/probes/bvh_grouped_candidates/USAGE.md)を参照。

156旧報告・77archives・20748filesと、D295で採用した36製品source・5PE/PFbuildおよびinstalled BVH PEを保持する。D296の不採用は維持する。GPU化率・GPU使用率、新規実RTK走行/RMS/全route速度は測っていない。

[結果](../benchmarks/bvh_grouped_candidates_20261019.json)、[完了照合](../benchmarks/bvh_grouped_candidates_completion_check_20261019.json)、[全呼出し測定](../benchmarks/bench_bvh_grouped_candidates.py)、[独立照合](../benchmarks/audit_bvh_grouped_candidates.py)、[native tests](../tests/test_bvh_grouped_candidates.py)。
