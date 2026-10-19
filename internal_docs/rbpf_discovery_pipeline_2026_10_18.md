# 全仮説の発見後にGPU評価を先行する (D282, 2026-10-18)

全仮説の発見を先に完了して1回のGPU評価を開始し、その間に粒子のrecord構築と元のprior正規化/logを進める実装を追加した。新しいGPU計算式や近似は導入していない。PF全体の新規測定では2条件とも旧D236 GPU比と同じ処理を同期実行する対照の基準を通過した。連続RTKの事前速度基準は通過しなかった。PF fixtureでの利得は確認できたが、実RTK全体の高速化は未成立。

## 実装と精度

元の修正版register-LU translation unitをbytesそのままで残し、hostのbegin/finish/drain APIだけを追加した。既存device workspaceを専有し、所有するpinned入力へコピーしてから同じkernelと出力転送を1streamへ投入する。finishは完了待ち後に元のfallback/status/scatter処理を実行する。入力・出力参照をjobが保持し、同時再利用を拒否、例外時はdrainしてからscopeとbufferを解放する。blocked対照も同じDLL・stream・pinnedコピー・packing・CPU処理を使い、begin内で待つ。

全仮説/key/pendingの初出順、粒子ごとのsurvive/stay H所有、releaseの独立辞書、各prior式、FP64、重みと乱数順を保持する。各粒子で元のarray/sum/divide/logを1回ずつ実行する。提案確率やnormalizerの再利用はない。元のmove/score ASTへ戻せることを検査し、3つ全てのdevice関数でaddress/encodingを除くSASSの命令・operandが一致した。通常Generator/slot/関数bindingと正のprior範囲を検査し、独自RNGやhook等は元のweight loopへ戻す。

76unique tests PASS（新規57、既存共通RTK protocol19）。内訳32+17+6+21で、sanitizer反復を加算していない。score/status/途中行列、複数seed・cluster・backendの4連続PF epoch、全状態/乱数/feedback、順序・所有、input snapshot、finish/cancel/close、例外時drain、元のstrict/fallbackを確認。memcheck/racecheck/synccheckは各16testでerror0、memcheck leak0。plugin自動読込を止めたsanitizer実行のpytest設定warningは保持した。

新しい実データ実行は400RTK/200PF。旧D236保存分と200状態行を文字列ごと完全照合し、軌跡のdtype/shape/bytes、config/入力/変更していないmetadataも一致した。197有効callは全てcomplete batch1回・finish1回、drain/error0。4166仮説を評価し、206522 recordsと12608粒子分の元prior処理をbegin後へ置いた。3skipも保持。配置を示すcounterであり、GPU実稼働率や実際の重複時間の証拠には使わない。3つの元work mapの実行回数減少はD279 validatorで明示照合した。

## PF全体の新しい測定

6mode、2条件、16seed、4連続epoch、5round×3group。768warm/11520timed callsを全て保存。各epochで全modeの入力を用意し、順をrotate/reverseして全PF timerを完了してから状態を保存した。BLAS1・通常GC・normal priority・非専有。発見/guard/packing/コピー/全prior/待機/weight/RNG/commit/feedbackまでtimer内。復元・hash・snapshot/serializeとcontext作成は外。元の失敗済みD279/D280測定を取り直していない。

正が処理時間短縮。group内で整数nsを加算してから比を求め、3group median→5round median。旧GPU比10%以上、matched blocked比3%以上かつ各4/5改善round、GPU A/A絶対差median5%未満かつ4/5安定roundを要求。CPU比較jointは旧CPU比10%以上・4/5改善とCPU A/Aも要求する。

| 条件 | 旧GPU比 | matched blocked比 | 旧CPU比 | same GPU / CPU joint通過 |
|---|---:|---:|---:|---|
| 64 rows / cluster 0 | +16.325% | +4.684% | +2.883% | True / False |
| 128 rows / cluster 1 | +22.401% | +4.775% | +16.275% | True / True |

両A/Aは2/2で通過、同GPU基準2/2、CPU比較joint1/2。全12288 warm+timed状態・metadata hashを照合し、別auditorが768 full snapshotとraw順/件数/整数時刻/全gateを再計算した。この値は状態保存を挟むPF fixtureのcall-time合計であり、連続RTK throughputへ読み替えない。

## 連続RTKの新しい測定

新しいD282候補に対し、元D280共通clock/gate関数のbytesを保った別runnerを用意した。4warm+20timed、24prefix/9600RTK/4800PF。各epochで状態保存・metadata encode・barrier・GC observerを置かず、最後のPF戻り参照だけを保持し、全driver後にsnapshotした。全24軌跡と最終PF/RNG/output/config/入力が一致し、root rtkを含む新しいpost-driver状態も全条件で一致した。旧PF-return goldとの最終比較だけはcapture時点の違いによりroot rtkを除くことを開始前に明記した。別の200行精度試験には除外がない。

新規共有cacheを4warmで用意し、20timedのbefore/after byte manifestを照合した。warm-cache有効=True。起動/JITのcold値と全phase1も保存した。全400 process時間3%以上・後半5%以上・driver3%以上を、元D236とmatched blockedの両方に要求。各4/5改善と同じ3metric全ての絶対A/A median5%未満・4/5安定も必須。初回1setのみで、gateや失敗値を変更していない。

| 対象 | 旧GPU比 | matched blocked比 | 絶対A/A median | A/A有効 | 速度gate |
|---|---:|---:|---:|---|---|
| 全400 RTK epoch | +3.554% | -0.159% | 19.689% | False | False |
| 後半200 epoch | +3.716% | -0.344% | 15.663% | False | False |
| driver全体 | +3.681% | -1.672% | 14.403% | False | False |

連続RTK joint speed gate=False。今回の結果からGPU稼働率、full route、複数seedの実測位精度は主張しない。製品source/既定は変更せず、実装と全証拠をprivate prototypeとして保存した。実際のCUDA/割当失敗注入、外部からの同時変更、raw ABI誤用は検証範囲外。

## 保全

prior testの初回Ruff F811は数値実行前のfixture binding問題で、初回testとpreflight記録を保存した。数値freeze前にbindingだけ修正した。RTK protocol preparerの初回は文字列の出現回数assert（masterでは2回）で停止し、数値/測定/新protocol file生成0。元preparerとfailureを保存し、別checked preparerで件数だけを修正した。124過去reports・61archives/13032files、36製品source・5installed PE・PF buildを保全する。

[結果](../benchmarks/rbpf_discovery_pipeline_20261018.json)、[完了照合](../benchmarks/rbpf_discovery_pipeline_completion_check_20261018.json)、[非同期API](../benchmarks/rbpf_async_score.py)、[発見とrecord構築](../benchmarks/rbpf_discovery_pipeline.py)、[priorの配置](../benchmarks/rbpf_prior_pipeline.py)。
