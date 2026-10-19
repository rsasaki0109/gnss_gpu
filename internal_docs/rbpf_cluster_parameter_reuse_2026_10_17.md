# RBPF cluster parameter-row reuse (D262, 2026-10-17)

cluster有効時の全scoreは元CPU比12.652%、元GPU経路比6.865%短縮。前回D261比の追加短縮はCPU 3.894%、GPU経路 2.491%。これはCPU側のparameter準備の改善であり、新しいCUDA数値処理の移行ではない。既定・製品は維持する。

## 実装と精度契約

凍結D261のeligibility関数を同じ関数objectとして継承し、selection/counter/fallbackを含むprepare全文ASTは同じ。closureのpacking helperだけを別実装へ渡す。同一call内でselected stateが同じobjectなら、初回の完成済み12-double rowをコピーする。別objectの同値stateは別計算し、Noneも独自の初回rowを持つ。初回のNumPy演算・代入式は元ASTと同じ。元の数値globalsは同じlive dictで、追加id intrinsicだけを検証してclosureに捕まえる。selected listがstate参照を保持するため、call内でidentityは再利用されない。

再利用はcall内で完結する。観測・仮説・粒子・prior・FP64・初回計算の演算順序・threshold・RNG・FGOモデルは同じ。既存のcode/binding/native export/ordinary-field/type/有限bounded位置/float64 array/errpolicy guardと未知filename fallbackをすべて保持する。guard対象外は元parameter blockを1回実行する。外部threadによる同時input変更の保証はない。

実入力cluster1の197call、4063rowのうち初回計算は537row、既存rowコピーは3526row。offline独立auditで全選択state identityとparameter bytesを元版へ照合した。logical canonical/parent要求数は同じで、batch_callsのみ新C API実呼出し数を表す。

## 全scoreの速度

全394保存実入力（各cluster197）、original/off/on CPU/GPU、D261 CPU/GPU、CPU/GPU A/Aの10mode、5round×3group、59100timed/3940warm。各roundは197件integer-ns合計のpaired短縮率の3group中央値、採否は5round中央値。元版と同じwrapperの無効版へ>=10%、D261へ追加>=3%、各対比>=4改善round、対応A/A絶対中央値<5%を事前要求した。GPU jointはさらにonCPUへの>=10%と両A/Aも要求する。全値を保持し、完了済みtimingの再試行はない。

正は短縮、負は遅化。

全59100timed/3940warmの数値照合と記録後、driver末尾の件数assertに旧9modeの3546/53190を残したためouter runはexit1だった。source・log・failed run・raw記録をすべて保持し、別checked auditが事前planどおりの10mode全63040記録の順序・件数・欠損/重複・数値hash・logical countを独立検証して集計JSONを復元した。計測済みsourceは修正せず、速度計測の再試行も行っていない。元のdriverを正常終了したrunとは扱わない。

| 対比 | cluster0 | cluster1 |
|---|---:|---:|
| cpu_prior | +1.582% | +12.652% |
| cpu_control | -1.742% | +12.350% |
| gpu_prior | -0.205% | +6.865% |
| gpu_control | -0.596% | +6.797% |
| gpu_vs_cpu | -175.765% | -121.555% |
| cpu_AA | -1.442% | -0.615% |
| gpu_AA | +0.222% | +0.332% |
| cpu_increment | +2.015% | +3.894% |
| gpu_increment | +0.909% | +2.491% |

CPU component 1/2、同一GPU経路 0/2、GPU joint 0/2。全warmのscore/status/135診断値/input bitsと全timed result/logical countを凍結same-backend referenceへ照合し、raw順序・欠損・重複・全round値を独立auditした。

## PF-stepと実走精度

完全PF-stepの8条件で1200timed/80warmを実施。全状態・RNG・graph/prior/caller metadataは同一backendの元版とbits一致。CPU 0/8、同一GPU経路 0/8、GPU joint 0/8が事前基準を通過した。

| 条件: particles/unique/rows/groups | CPU元比 | GPU元比 | A/A CPU/GPU | gate CPU/GPU/joint |
|---|---:|---:|---|---|
| 0: 64/6/64/4, c0, feedback=True | -20.807% | +6.724% | False/True | False/False/False |
| 1: 64/32/64/4, c0, feedback=True | +3.083% | -0.956% | False/True | False/False/False |
| 2: 64/64/64/4, c0, feedback=True | +7.905% | -3.366% | True/False | False/False/False |
| 3: 128/32/64/4, c0, feedback=True | -2.734% | -6.990% | False/True | False/False/False |
| 4: 64/6/128/8, c1, feedback=True | +7.997% | +16.581% | False/True | False/False/False |
| 5: 128/128/128/8, c1, feedback=True | +20.489% | +25.715% | False/True | False/False/False |
| 6: 64/6/64/4, c0, feedback=False | -2.103% | +12.811% | False/False | False/False/False |
| 7: 64/6/64/4, c1, feedback=True | +1.183% | +13.268% | False/False | False/False/False |

A/A不成立や遅化を含む全roundを保持し、短いstepのばらつきを改善として採用しない。step timerはD236の完全PF処理を含む。観測抽出・整数探索・時刻/衛星入力・graph solveはfixture境界であり、実RTK/FGO全体のthroughputではない。復元・hash・検証はtimer外。各stepは同じ入力から復元し、copied RTKにfeedback callbackをbindして元templateの不変性を確認した。

158tests PASS（既存config warning1）。従来144testsはimport先以外を変えず、追加14testsで初回演算回数・同値別object・NaN payload/符号付きzero/subnormal/infinity prior・C/F/negative stride・参照寿命・call間変更・closure/live globals・空入力を検証した。CPU/GPUの全score、54数値fixture、custom callback/strict/error fallback、3seed×4条件×4epochの保存PF/RNG/graph/prior metadata照合を含む。

実GPU run3/seed20260710、cluster0/1各400epoch prefixで各200保存PF状態（計400）と全trajectory配列のdtype/shape/bytes、cfg・prior caller metadataが凍結D236と一致。各197有効score、scalar fallback0、cluster1 batch197/cluster0元block197を確認した。検証wrapperはconstructorがbindしたcandidate scoreそのものを捕まえてから回数/全診断値を加える。実走のwall/step時間は診断overheadを含むため速度の根拠に使わない。新CPU実走、全route、多seedの実走、fault injection、sanitizer/CPU ASanは今回追加していない。以前の失敗fieldと遅化timingも過去archiveで保存する。

## メモリと保存

4timing contextsの既存numeric stagingはhost/device各4573408B（各context1143352B）。実走は1context。新CUDA bufferはない。新row mapはcall内の異なるselected identityごとの整数keyとrow indexのみを保持し、return前に破棄する。元のselected listと出力parameter matrixを再利用し、追加parameter row arrayを保持しない。Python heap peakは未計測。36製品source・5installed PE・PF build PE・84過去reports・41archives6831filesの不変性を照合した。

[機械可読結果](../benchmarks/rbpf_cluster_parameter_reuse_20261017.json)、[完了確認](../benchmarks/rbpf_cluster_parameter_reuse_completion_check_20261017.json)。既定/製品変更、実RTK/FGO全体の速度・GPU化率の主張はない。
