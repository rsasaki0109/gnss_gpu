# GPU先行評価を1回のbatchで完結できるか (D281, 2026-10-18)

最初の粒子を登録した時点で共通仮説をGPUへ先行投入し、残りの粒子登録と重ねる案を調べた。保存済み197有効callのうち、少なくとも140callではこの時点に全評価対象が揃わない。全対象が揃い得るcallの楽観的上限は57/197（28.934%）で、事前90%work triageを満たさない。この単純な1batch先行案は実装へ進めない。GPU並行化全般を否定した結果ではない。

## 再実走を使わない調査

D279で精度確認済みのwork記録と元D236 metadataを用い、200PF callを全て照合した。197有効・3skipを保持し、有利なepochを選ばない。今回のestimator/GPU実行・性能計測は0回。D230の提案確率再利用やD280の実RTK速度FAILを取り直していない。

元の凍結move sourceのSHAとASTを検査すると、最初の粒子の固有候補はsurvive/extend/shrink/releaseの最大4件、空Hではstay1件。続いてconsensus/adopt/spawnの共通候補を登録する。独立したAST検査でも、固有分岐に追加loopがなく、scoreは全粒子登録後にあることを確認した。

各callの最終pendingは`unique_moves - existing_cache_hits`で、保存済み実GPU batchのunique数と一致する。最初の粒子で準備できるpendingの**楽観的上界**を`min(final_pending, common_records + 4)`とする。重複や既存cache hitを最初の登録から引かず、全登録が新しい要求だった場合まで許す。したがって上界に収まるcallも実際に揃っているとは限らないが、上界を超えるcallはこの時点の1batchで全要求を処理できない。

| 保存済みの作業量 | 結果 |
|---|---:|
| 最終評価対象の総数 | 4166 |
| 最初に準備できる数の楽観的上界合計 | 3280 |
| 後で追加される数の下界合計 | 886 |
| 全対象が揃い得るcallの上限 | 57/197 |
| 確実に未完了のcall | 140/197 |

仮説数で重み付けした準備率上界は78.733%。これは時間割合・GPU稼働率・速度倍率・GPU処理量の削減率ではない。個別割合の平均も使わず、件数を合計してから比を求める。

この案で全callを最初から先行投入するなら、少なくとも140callで残りの評価処理が必要になる。後続batchには追加起動・packing・転送が伴う。単にAsync APIへ変えて高速化を前提にする段階ではない。今回はその費用を測定しておらず、split batchが必ず遅いとも断定しない。事前90%は、この単純な1batch構成を試作するためのcoverage条件であり、全てのGPU方式の必要条件ではない。

## 検証と残る設計

14unique tests PASS。凍結source/変更拒否、4件の上界、cacheの二重控除防止、未評価0件、無効count、90%境界、600個の重複/cache集合例、件数の加算順を確認した。初回は13PASS/1件のtmp_path setup PermissionError。元source/log/resultを保存し、別runnerで新しいworkspace内basetempだけを指定して14PASS。helperとtestのbytes・分析末尾AST・基準は同じ。初回13件を14件へ加算しない。

別auditorが凍結source・全200行・197件それぞれのpending/上界・全合計・triageを独立に再計算した。入力SHAはD279証拠へ結び付け、122過去reports・60archives/12819files・36製品source・5installed PE・PF buildを保全する。新しい数値/CUDA実装、製品/既定変更、精度許容差の変更はない。過去の200状態完全一致は保存済み証拠であり、新しい実走精度試験として数えない。

次の設計候補は、全仮説の発見を先に完了し、GPUへ一括投入してから残りのCPU record構築を進める構造。別案は明示的split batchの追加費用を含めた評価になる。どちらも未実装・未採用であり、元の候補順/全候補/数値式/RNG、buffer寿命、例外時の待機/解放、同backendのstate/軌跡一致と新しい速度対照が必要になる。今回の上界から利得を予告しない。

[結果](../benchmarks/rbpf_gpu_frontier_20261018.json)、[完了照合](../benchmarks/rbpf_gpu_frontier_completion_check_20261018.json)、[上界helper](../benchmarks/rbpf_gpu_frontier.py)、[tests](../tests/test_rbpf_gpu_frontier.py)。
