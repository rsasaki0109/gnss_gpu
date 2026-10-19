# D296: 同じ反射方向の再利用と中間cache (2026-10-19)

**全体条件は不合格で不採用。** 12,000三角形・128epoch×12satのwarm全呼出しは元比23.735%短縮、同PE off比23.749%短縮で両A/Aも合格。しかしcoldは元比5.531%悪化し、120,000三角形・32epoch×32satはwarm17.409%・cold31.675%悪化した。良い条件だけで採用せず、製品/defaultを維持した。

元GPU unit normalと頂点v0の全6 uint64成分が一致する場合だけ共有する。CPUの近似法線や幾何toleranceを使わず、hash衝突もfull key比較で解決。各rayの鏡像点・平方根・3除算を1groupで一度評価し、全三角形の元交差式・delay・ordinal・tieを維持した。6000/60000groupsで元12000/120000entriesの半数へ減ったが、56bytes/group/rayを中間GPU cacheへ書き出す。上限128MiBのchunk処理で、測定fresh cacheは133056000/107520000bytes。coldにはv0snapshot、元GPU法線D2H、CPU grouping、map H2D、準備とcloseを含めた。メモリ帯域を原因とする断定はprofilerで行っていない。

36unique tests PASS、memcheck/racecheck/synccheck各3件でerror/leak/hazard0、全観測processのexit0。元12GPU関数の全SASS/encoding一致。4固定条件、元/同PE off/on/両A/A、5round×3group×2repの600timed、12setup・60warm・100coldの全772callで返却dtype/shape/bits一致。60warm全状態も保存。事前固定の独立auditorが全grid/順序/ns/入力/状態hash/統計を再計算した。両大規模warm10%以上・4/5正、small/cold5%以内非退行、両A/A絶対中央値5%未満かつ4/5で5%未満を全条件に要求し、cold/smallの不安定さと負roundも残した。計時にprofiler/sanitizer/周期stack dumpは使わず、CPU1/BLAS1/通常GC/priority・非専有環境で直列実行した。

最初のCUDA workerは親が作ったmetadata directoryを再作成して、ready/pytest/GPU import前に終了した。失敗source/plan/log/exitを保持し、既存のatomic ready/bounded handshakeを使う別worker/runnerだけで修正。数値source/test/helper/PEは不変で、速度測定の取り直しはない。154旧報告・76archive・20422filesとD295採用source/PEを保全する。測位精度の新保証、PF/FGO/実RTK速度、GPU使用率やpublic/urban pipelineの速度へは拡張しない。次は中間cacheを持たず、同じthreadでgroup内の全三角形を評価する別実装を検討する。

[結果](../benchmarks/bvh_grouped_directions_20261019.json)、[実装](../benchmarks/probes/bvh_grouped_directions/README.md)、[全呼出し測定](../benchmarks/bench_bvh_grouped_directions.py)、[独立照合](../benchmarks/audit_bvh_grouped_directions.py)。
