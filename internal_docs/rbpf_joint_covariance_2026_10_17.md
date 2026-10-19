# FGOの共分散ブロック取得をまとめる (D272, 2026-10-17)

64tests PASS。CPUcluster0/1の計400保存PF状態・全軌跡array/設定/入力/従来metadataが元D236にbits一致し、実走の各共分散copyも元loopに一致した。全RTK交互screenはPASS。通常連続実行の18prefix/7200epochも完了し、事前gateはFAIL。全400短縮中央値 +1.695%、phase1 +6.229%、phase2 +0.808%。全400の対照絶対差中央値 3.437%。 製品/既定は変更しない。

## 変更する処理

元の`_write_back`/`_write_back_tc`は、全曖昧度の組ごとに`JointMarginal.at(k1,k2)`を呼び、同じ値をnav.Pの対称位置へ書く。その二重loopだけを置換し、AST undoが元の完全methodへ一致することを確認する。位置共分散・各位置との交差項の取得/座標回転、対角項、graph solve、最後のP copy、held-state branchは元の式と順序で残す。全観測/粒子/候補/prior/閾値/IRLS/RNG/FIX/feedbackに変更はない。

既存`fullMatrix()`から1回取り出した同じmatrixを使い、activeの元順の上三角要素を取り、両側へ同じbitsを代入する。下三角と平均せず、signed zero/NaN payloadも算術せずコピーする。GTSAM PEを固定し、異なるkey順/3D・6D位置の2656blockが`.at()`とbits一致することを実GTSAMで確認した。plain Nav/array/IB/source/bindings、index範囲・一意性・matrix shapeを検査し、未対応のcopy blockは元loopを実行する。任意GTSAM版の対応は主張しない。

## 大きさと実データの型

最初の無条件bulk componentは16/48曖昧度で速いが、4個では遅かった。全6条件を残し、4/6通過・両48条件通過という事前screenを満たした。その後の**別構成**は16未満を元loopへ戻す。16は今回の測定点からの保守的dispatchで、最適な境界を一般保証しない。

| 位置dim | 曖昧度 | component短縮 | 元A/A | bulk A/A | gate |
|---|---:|---:|---:|---:|---|
| 3 | 4 | -232.314% | +0.070% | +5.887% | False |
| 3 | 16 | +67.158% | +3.124% | +1.195% | True |
| 3 | 48 | +92.493% | -2.953% | +3.149% | True |
| 6 | 4 | -208.178% | +0.688% | -2.746% | False |
| 6 | 16 | +67.958% | +1.096% | -1.418% | True |
| 6 | 48 | +92.305% | +0.735% | -3.267% | True |

5round×3group×16call、各mode16warm、全5760timed/384warmを保持し、順序/個数/元と候補両方の対照/round統計を独立再計算した。これは実GTSAM objectに対するcopy blockのscreenで、graph solveやRTK全体の短縮率ではない。元loop/guard/NumPy一時配列/代入をtimerへ含む。P全bytesは各group後に照合する。

初回component起動は検索パス不足で数値開始前に失敗し、元source/logを残した。別launcherで同じ数値sourceを最初に完了した。最初の実走は400epoch/200PF状態を完走後、高速経路usageのassertで止まった。短い別診断で実satが`np.int32`であることを確認し、その型だけを受け入れる別adapterを追加した。検査用metadataだけをPython intへ変換し、実際のIBには元のnp.int32を渡す。元のindex算術を変えず、unsafe na/int64/custom callbackは元処理へ戻す。

無条件版36、size版14、実schema版14の計64testsが各suite最初の実行でPASS（既存pytest設定warningが各1）。完全writer、Pose3回転/held/subset、key順、実int32と高sat番号、非対称・signed-zero・NaN payload、custom IB/binding、buffer/strides/read-only/重複index、元code変更のfallbackを確認した。

別のpytest collection hookは、この3test moduleだけを対象にoptional依存の有無と認証済みGTSAM/cssrlib hashをtest moduleのimport前に確認する。認証環境では64件を収集し、依存がない通常環境では理由付きで3moduleをskipするcollection-only検査を行った。計測済みのtest/sourceを変更せず、数値testの再実行もしない。他testのcollectorは変更しない。

最初のcollection checkerはpytestのskip表示書式を誤って期待してassertした。実際の3module skipは正常であり、元checker/logを残して別checkerで確認した。数値sourceやhookの変更、数値testや完了した速度測定の再実行はない。

続くcheckerは対応環境のcollectionで長時間停止したため、起動時刻と親子関係を確認して所有childだけを終了した。別のBLAS1・上限時間付きdiagnosticでは64件を2.29秒で収集したが、設定も異なるため停止原因は断定しない。停止したcheckerと空logも保存し、default環境の3module skipは上限時間付きで確認した。いずれもcollection-onlyである。

## 実走の精度と利用経路

Tokyo run3/seed20260710、CPUcluster0/1各400epoch/200保存PF状態。各copyに対し別Pへ元loopを実行して全P bytesを照合し、その後の全保存PF state・RNG・navigation x/P・全trajectory配列dtype/shape/bytes・設定/従来metadataが元と一致した。large経路は正、未対応fallbackは0、小さいactive setは元loop。失敗した初回prefixの200保存状態/軌跡も別auditで一致を確認したが、高速経路の精度件数400には加えない。scalar NaN payloadを正規化する既存snapshot codecの制限は継承し、新たな許容差は加えていない。

## 全RTKの評価

cluster0の完全RTKを候補/元のpairと、元/元の対照pairで比較する。candidate labelをroundごとに交換し、凍結D271 coordinatorの両ready/両done barrier、先行順交代、D270 complete process/3clocks bodyを維持する。guard/metadata正規化/size判断/全FGO・PF・ARをtimer内、snapshot/JSON/gzip/constructorは外。warm2pair後に5candidate+5control pair、24prefix/9600RTKepoch/4800PFcall。全trajectory/cfg/input/source/PEが一致する。

事前gateは全400中央値3%以上/4of5正、phase1は5%以上/4of5正、phase2は中央値-2%以上/4of5が-2%以上。全3集計で対照絶対差中央値5%未満かつ4of5組が5%未満を要求する。起動直後/負値/失敗も全て保持し、完了済みの取り直しは行わない。

| round | 全400短縮 | phase1短縮 | phase2短縮 | 全400元A/A |
|---|---:|---:|---:|---:|
| 1 | +2.570% | +9.415% | -1.505% | +1.125% |
| 2 | +0.925% | +7.693% | -3.042% | +1.711% |
| 3 | +3.101% | +9.802% | -0.453% | -1.087% |
| 4 | +3.451% | +10.463% | -1.218% | -0.344% |
| 5 | +3.307% | +8.263% | +0.414% | -0.586% |

交互実行gateはPASS。強制待機のcache/working-set影響があるため、通過した場合だけ別の通常連続実行（3warm+5round×3mode、同gate）を必要とする。通常連続実行の18prefix/7200epochも完了し、事前gateはFAIL。全400短縮中央値 +1.695%、phase1 +6.229%、phase2 +0.808%。全400の対照絶対差中央値 3.437%。 componentの大きな短縮率を全体へ読み替えず、過去候補のFAILも変更しない。CPU1本/BLAS1/通常優先度/非占有、全データを直列実行した。


| 通常連続実行 | 短縮中央値 | 元A/A絶対差中央値 | 候補条件 | 対照条件 |
|---|---:|---:|---|---|
| 全400 | +1.695% | 3.437% | False | False |
| phase1 | +6.229% | 4.468% | False | False |
| phase2 | +0.808% | 8.536% | True | False |

## 保存と制限

36製品source・5installed PE・PF build PE・104既報・51archive/9876filesを保全する。新しいGPU kernel/buffer/transferやGPU使用率の測定はない。new allocationはfull dense matrix、indices、上三角値等のcall-local NumPy/Python objectで、heap peakは未測定。frozen dependency/ordinary unmodified APIを対象とし、任意namespace/property改変、同じstateへの並行書込み、OOM/他version/全route多seedを一般保証しない。

計測済みcomponent sourceはbyte単位で固定したため、setupのSHA lambdaについてE731のstyle例外をSHA付きで記録し、そのfileだけper-file除外して残るRuff/compile/EOL/diff検査を通す。失敗した起動・初回実走・guard診断・無条件smallの負値も保存する。

[結果](../benchmarks/rbpf_joint_covariance_20261017.json)、[完了照合](../benchmarks/rbpf_joint_covariance_completion_check_20261017.json)、[copy core](../benchmarks/rbpf_joint_covariance.py)、[size dispatch](../benchmarks/rbpf_joint_covariance_sized.py)、[実schema](../benchmarks/rbpf_joint_covariance_field_schema.py)、[全RTK](../benchmarks/bench_rbpf_joint_covariance_route.py)。
