# D260: 整数DDによる親状態検索の一括化 (2026-10-17)

D259でcluster有効時の状態・centre/prior準備が全scoreの約23%だったため、仮説群の状態選択を1回のCPU C APIへまとめた。正確性の照合は通過したが、全scoreはc1でCPU 79.721%・GPU 35.742%遅くなり、不採用。既定・製品・GPU数値実装は変更していない。

## 実装と判定の保持

全仮説・全親状態を確認し、親Hの型確認をbatch単位にまとめる。整数差分は任意精度PyLongのまま、agree/comparedは元と同じPython整数のtrue division。閾値・exact hit前のsupport条件・辞書の走査順・同点時None・返すstate objectの同一性を保つ。順位は(agree,compared)の最大値と最大同点を逐次記録する。canonicalとoverlapの数値bodyは元の実装へ独立auditで戻せる。

scoreのparameter blockだけを置換し、centre/priorのNumPy式・並び・代入はliteral保持。元のlive globalsを使い、補助関数はclosureで渡す。guardは元のmethod/code/globals/builtins/native export/普通のobjectとarray/policyを確認し、独自callback・型・unsafeなcentre計算では元blockを1度実行する。cluster無効時も元blockへ戻す。キー・DD・状態・priorのcross-call/epoch cacheはない。既存canonical/parentのnative_callsはlogical request数として保ち、新しいbatch_callsを別に記録する。logical数を実C API呼出し数へ読み替えない。

## 全score計測

全394保存実入力、各cluster197件、original/off/onのCPU/GPUとCPU/GPU A/Aの8モード、5round×3groupで47280timed/3152warm。各roundは197件integer-ns合計のpaired短縮率を3group中央値にし、採否値は5round中央値。>=10%かつ>=4改善round、対応A/Aの絶対中央値<5%が事前基準。CPUはoriginalと同一extension無効版、GPUは両GPU対照、jointはさらにonCPUとの比較と両A/Aを要求した。完了した測定のやり直しなし。

正は短縮、負は遅化。全round値・対照の揺れも結果JSONに残す。

| 対比 | c0 | c1 |
|---|---:|---:|
| cpu_prior | -1.704% | -79.721% |
| cpu_control | -2.489% | -78.568% |
| gpu_prior | +0.150% | -35.742% |
| gpu_control | -0.364% | -36.504% |
| gpu_vs_cpu | -161.125% | -67.267% |
| cpu_AA | -0.307% | -3.698% |
| gpu_AA | -0.156% | +0.170% |

CPU component0/2、GPU same-backend0/2、GPU joint0/2。A/A中央値は両条件・両backendで基準内だが、c1 CPU A/Aには-6.891/-5.361%の個別roundもある。c1の新実装は全5roundで遅化。c0のGPU+0.150%も改善として採用しない。ここでのGPU対CPU値はこのcomponentと環境の比較であり、全PF/FGO速度やGPU化率ではない。

時計には全dedup/guard/C API/整数処理/カウンタ更新/NumPy parameter準備/数値処理/転送/完了待ち/返却を含む。復元/setup/warm/hash/検証/圧縮保存は時計外。BLAS1・許可CPU1個・通常優先度・非占有。異なる実験の絶対時間を引いて原因や改善量を算出しない。

c1の1corpusは197C API batch、4063仮説、4063canonical logical requests、7299parent logical requests、335exact hits、2674parent selections、1030ties。元のcallback logical countsと全callで一致した。c0は197回の元blockで新batch0。これらは作業数であり速度ではない。

## guardの追跡診断

別の1回の入力走査でeligible guardの_literal呼出し3個のみを、隔離globalsで時刻計測した。guard wall span中央値1.719400ms、うちliteral確認のcall内割合の中央値93.956%。このhelperは各回に関数filenameと固定sourceのPath.resolveを実行し、batchあたり計6回。guard内のsource/code/path検査が大きいことを示す診断値で、純粋なfilesystem時間やCPU cyclesではない。較正済み性能試験ではなく、この値を全scoreから引いて修正版の速度を予測しない。

次の候補は、source SHAを初期化時に確認したうえで、既知のfilename表記とcode/binding検査を使い、hot pathのfilesystem解決を取り除く別wrapper。元と同じ数値処理・live binding guardを保ち、未知の表記は元blockへ戻す。これは本記録時点では未実装・未計測であり、初版の不採用を撤回しない。

## 検証と保存

基本120testsと追加checked19tests PASS、各suiteに既存設定warning1件。270random batchesと順序・16000bit gauge変更、任意精度整数・support・exact hit・順位同点・NaN/Inf/隣接float閾値・state identity・10000回refcount・入力変更を検査。54fixtureのCPU/GPU全score/135診断/status/input bits、callback/型/strict policy/例外/警告fallback、3seed×4設定×4epochの全PF state/RNG/graph/prior metadataが一致。

最初の追加guard試験は18PASS/1FAIL。dtype=floatのfloat名をcallableに変えるfixtureで、元実装もTypeErrorになるのに配列の成功を仮定していた。元と新実装の同じ例外を比較する別checked testへ直し、初回testはbyte同一で非test名へ移して旧pathとplan/logを保存。数値許容差・実装・C++・PEを最初の数値試験以降変更していない。

全3152warmのscore/status/135診断/input bitsと全47280timedの返却hashは凍結した各backend参照と一致、scalar fallback0。全rawの実際の列順・重複欠落・logical counts・新path・独立paired集計を再照合した。CPU extensionのbuildとsource undoも通過。CUDA数値source/binaryは凍結D253のままで、新たなCUDA sanitizer/CPU ASanは実行していない。

400状態field用source/runner/auditは準備したが、初版の性能不採用後は実行していない。新しいfullroute/multiseed測位精度や全graph速度、allocation/async障害注入も未検証。既存NumPy/native差とscalar NaN codecの限界を引き継ぐ。

3timing contextのnumeric stagingはhost/device各3430056B、各context1143352Bのまま。新しいCUDA bufferはない。新CPU一時listは最大8*unique bytesのpointer payloadとPython header/reference、canonical key一時object、heap peak未計測。選択stateはparameter準備中だけ強参照し、保持cacheはない。36製品source・5installed PE・PF build PE・80過去reports・39archives6464filesを保存照合した。

[結果](../benchmarks/rbpf_cluster_batch_20261017.json)、[完了照合](../benchmarks/rbpf_cluster_batch_completion_check_20261017.json)、[wrapper](../benchmarks/rbpf_cluster_batch.py)、[C++](../benchmarks/probes/rbpf_cluster_batch/select.cpp)、[driver](../benchmarks/bench_rbpf_cluster_batch.py)。
