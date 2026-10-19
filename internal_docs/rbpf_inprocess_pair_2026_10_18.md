# D284: 独立した2つのRTKを1プロセスで交互に計測する (2026-10-18)

同じD236実装同士の新しいA/A計測を実装した。400エポック×2を1組とし、ウォームアップ1組と本測定5組、全12prefix/4800RTK/2400PFを残す。校正結果は**FAIL**。これは最適化の速度向上や連続した単独driverのthroughputを示す測定ではない。D282の速度FAILと既定実装を維持する。

## 処理と精度の境界

元のWP16 main全体をASTで保持し、parserの同じ明示tokens、import後のローカルclass factory、唯一の`rtk.process(obs,obsb=obsb)`のticket転送だけを追加する。3変更を戻したmain全体のASTは元と一致する。独立した観測decoder、nav、FGO graph、粒子、RNGを2つ持ち、両入力を準備してから元のbound processを同一threadで1回ずつ呼ぶ。各epoch・組で実行順を交互に反転し、phaseごとの先行回数を均等にする。共有するのは同じ同期D236 scorerのcontextと通常のmodule/JIT cache。数値関数やmodel classのglobal bindingは変更しない。

ウォームアップでは両RTK時計が止まった後に200状態ずつを採取し、A/Aは全状態一致。旧PF-return基準とは、採取境界が異なるroot `rtk`だけを事前宣言どおり除き、粒子/出力/RNG等を完全一致で比較する。全12の最終状態はroot `rtk`を含む宣言済みboundaryごと旧post-driver基準と完全一致、全12軌跡の全NPZ配列もdtype/shape/bytes一致。metadata/config/input/source/PEも照合した。許容誤差や粒子数・候補数・FP64・prior・feedbackの基準は変えていない。full route/多seed精度を新しく確立したわけではない。

## A/A校正

| RTK本体の境界 | 5組の絶対差中央値 | 絶対差5%未満 | 判定 |
|---|---:|---:|---|
| 全400エポック | 1.507% | 3/5 | FAIL |
| 前半200・PFなし | 0.732% | 5/5 | PASS |
| 後半200・PFあり | 1.871% | 3/5 | FAIL |

全3区間で絶対差中央値<5%かつ4/5組以上が<5%を同時に要求する。符号を相殺した中央値、CPU時計への置換、遅いepochの除外、完了setの取り直しはしない。wall/process CPU/thread CPUの全生時計、3200events×6組、warmの全時計、負の対照差も保存する。cache有効性はTrue、warm後の56filesについて本測定前後のpath/byte manifestを照合した。通常GC、許可CPU1つ、BLAS1、非専有環境。原因をOS/GC/周波数/GPU待ちのどれかに決めつけない。

単独driverを途切れずに処理する環境とは、2つのgraphの常駐、交互実行、shared module/cache/context、入力準備やtailの時間関係が異なる。この校正が通っても連続RTKの既存速度gateを置き換えず、D282候補を再測定して合格にしてはいない。新しい最適化の利得やGPU使用率も主張しない。

## 初期失敗と修正

最初の45testsは43PASS/2FAIL。未開始generatorのcloseはbodyのfinallyを実行しないというPythonの挙動をテストが誤って仮定していた。別ファイルの修正版では両generatorのGEN_CLOSEDと開始済みstreamのfinallyを検証し、45PASS。初版はそのまま保全し、通常pytest収集は修正版に限定した。

最初の実データwarmは初回有効PF更新のepoch203で比較assertにより停止した。初版はassert前に途中の生時計を書き出しておらず、source/plan/log/run recordを保全した。別の204epoch/arm診断では408生時計と8状態を保存し、2つの状態は完全一致、保存後のlegacy比較も数値/metadata差0だった。原因はfresh encodeの配列shapeタプルとJSON保存済みリストの比較。別版でJSON container形式をそろえ、全floatタグ/array byte/shape/順序を保ったまま比較する。6個の回帰testsでsigned zero/NaN/Inf/整数/tuple keys/順序/shape/1bit差の拒否を確認し、合計51unique tests PASS。初版失敗・途中診断・修正版の最初の完了setを保存し、途中終了を12完了prefix/4800校正時計や性能roundへ水増ししない。

## 保全

別auditorが全4800時計、19200protocol events、400warm状態、12最終状態、12軌跡、設定とSHAを独立照合する。128過去reports、63archives/14128files、36製品sources、5installed PE、PF buildを保全する。製品数値source/既定の変更はない。

[結果](../benchmarks/rbpf_inprocess_pair_20261018.json)、[完了照合](../benchmarks/rbpf_inprocess_pair_completion_check_20261018.json)、[protocol](../benchmarks/rbpf_inprocess_pair.py)、[修正版field caller](../benchmarks/bench_rbpf_inprocess_pair_checked_field.py)、[tests](../tests/test_rbpf_inprocess_pair_checked.py)。
