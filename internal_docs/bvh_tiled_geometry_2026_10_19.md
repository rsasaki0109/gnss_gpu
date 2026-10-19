# D294: 反射入力の32要素SoA配置 (2026-10-19)

**大規模2条件とも全呼出しの短縮が1%未満で、事前の10%条件を満たさない。採用しない。** 全772呼出しの返却dtype・shape・bitsは一致した。製品source、installed PE、default、候補数、精度条件はこの評価の境界で変更していない。

## 変更と検証

並列反射の32隣接候補が読むraw頂点と元GPU法線をSoAの連続要素に複製する。3200bytes/32要素。edge/planeの新しい演算、近似、候補削減は行わず、既存の交点式・走査順・同点時の選択をそのまま使う。既存12GPU関数に触らず4関数を追加し、明示設定かつ4096候補・512ray以上の並列経路だけで有効になる。128MiBを上限とし、小入力は元経路。新APIは実験PEだけでdefault off。

fresh bulkでは元normal32bytes/entryがtile100bytes/entryになり、元geometry72bytes/entryも残る。差は+68bytes/entry。off/on切替では両cacheが残る場合もある。全owned cudaMallocを集計し、準備・転送を含むcold時間、close後0、snapshot不変、tile境界、chunk、cap、NaN、scale、同点等を検証した。

69unique tests PASS（projection19、checked native38、CPU重複9、sparse GPU3）。追加のmemcheck/racecheck/synccheckは同じsparse3件ずつ、全launcher/worker/観測processのexit0とerror/leak/hazard0まで確認した。周期stack dumpやdebuggerを使っていない。CUDA検査は速度時間に含めていない。

## 速度

2resident contextを使い、original/original_AAは同じ元scene、off/on/off_AAは同じ候補sceneで比較した。CPU1/BLAS1/通常GC/priority、非専有環境、直列実行。12初期setup・60warm・600timed・100coldを全保存し、60warm全状態も保存した。whole synchronous trace_batchのvalidation/H2D/LOS/reflection/sync/D2H/返却配列を計時し、coldはconstructor/selector/firsttrace/closeを含む。入力構築・CPU BVH構築・hash/保存は計時外。warmのselectorは計時前に実行する。

全4条件でwarm/coldとも両controlおよびA/Aに合格し、大規模warmは両control比で中央値10%以上かつ4/5roundで正、small/coldは5%以内の非退行を要件として事前固定した。両A/Aは絶対差中央値5%未満かつ4/5roundで5%未満。独立auditorも実行前に固定し、全grid/順序/ns/input/60状態/hash/統計を再計算した。以下は観測差であり採用利得ではない。

| 条件 | warm元比 | warm同PE off比 | cold元比 | 全条件合格 |
|---|---:|---:|---:|---:|
| 12000三角形 / 128epoch×12sat | +0.018% | +0.074% | -3.327% | False |
| 120000三角形 / 32epoch×32sat | +0.652% | +0.582% | +1.409% | False |
| 12000三角形 / 1epoch×12sat | +9.355% | +3.108% | +1.837% | False |
| 4096三角形 / 16epoch×8sat | -1.324% | -0.954% | -1.976% | False |

大規模warmの両A/Aは安定しているが、利得は小さい。small1件のwarmと両smallのcoldではA/Aや非退行条件も不合格。遅いcallや負roundを除外せず、取り直しもない。native段階で不合格なのでpublic/urban pipelineの速度評価には進まない。PF/FGO/実RTK全体の速度や新しい測位精度保証、GPU使用率には換算しない。

## 見つかった既存CPU問題と保全

初回native suiteは50件後、重複形状のCPU BVH構築で0xC00000FDになった。元installed PEだけをロードした独立processでも8192個の零三角形で再現し、GPU候補が原因ではないと確認した。元SAHは同点の最初のsplitを選び続け、深い再帰になる。分割式やtie/sort順を維持し、右を先にpending vectorへ積む反復schedulerを別実験PEで検証した。通常stackの候補と64MiB CPU threadの元参照で全node/permutationが一致し、修正前後の全16GPU関数はSASS/encodingも一致した。D294ではこの修正も製品へ入れず、別のbugfixとして扱う。

最初のbuild preflightにはPE一覧の誤想定、historical D215記録と実installed PEの不一致、CMakeのWindows backslash解釈があった。失敗source/plan/logを全保存し、実際の元PEを別snapshotに固定、pathのみforward slashにした版でbuildした。初回のCPU fatal suiteもそのまま残し、checked suiteはGPU test関数を同一bytesのままimportし、元CPU参照のthread stackだけを増やした。失敗suiteを成功と数えない。

150旧報告・74archive・19547files、36製品source・既存5PE/PFbuildおよび実installed BVHを保全する。既完了FAILの再判定・再測定は行っていない。

[結果](../benchmarks/bvh_tiled_geometry_20261019.json)、[完了照合](../benchmarks/bvh_tiled_geometry_completion_check_20261019.json)、[projection/build](../benchmarks/probes/bvh_tiled_geometry/README.md)、[全呼出し測定](../benchmarks/bench_bvh_tiled_geometry.py)、[独立照合](../benchmarks/audit_bvh_tiled_geometry.py)、[checked tests](../tests/test_bvh_tiled_geometry_checked.py)、[CPU回帰](../tests/test_bvh_cpu_duplicates.py)。
